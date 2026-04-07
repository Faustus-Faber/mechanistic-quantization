import os
import json
import torch
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ==========================================
# PHASE 2: MEMORY-CONSTRAINED LAYER-WISE CAUSAL TRACING
# ==========================================
# Native PyTorch Execution: No TransformerLens overhead/shape mismatches.
# Strict 8GB VRAM adherence via sequential precision swapping.

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "gemma-4-e2b-it")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "contrastive_failure_dataset.json")
RAW_PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw_prompts.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "layer_attribution_scores.json")

def cleanup_vram():
    """Violently garbage collect and wipe CUDA memory to maintain the 8GB constraint."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
def load_data():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        failed_cases = json.load(f)
    with open(RAW_PROMPTS_PATH, "r", encoding="utf-8") as f:
        raw_prompts = json.load(f)
        
    prompt_map = {item['id']: item['prompt'] for item in raw_prompts}
    
    dataset = []
    for case in failed_cases:
        p_id = case['id']
        dataset.append({
            'id': p_id,
            'divergence_type': case['divergence_type'],
            'prompt': prompt_map.get(p_id, "")
        })
    return dataset

def run_atp_pipeline():
    dataset = load_data()
    if not dataset:
        print("Dataset is empty. Run Phase 1 first.")
        return

    fp16_cpu_cache = {}
    target_tokens = {}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    print("========================================")
    print("STEP 1: NATIVE FP16 CLEAN CACHING")
    print("========================================")
    
    cleanup_vram()
    hf_fp16 = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map="cuda"
    )
    
    # Locate the layer sequence container natively
    layer_container = hf_fp16.model.language_model.layers
    n_layers = len(layer_container)

    for data in tqdm(dataset, desc="Caching clean fp16 states to RAM"):
        input_ids = tokenizer(data['prompt'], return_tensors="pt").input_ids.to("cuda")
        
        cache = {}
        hooks = []
        
        def make_fwd_hook(name):
            def hook(module, inp, output):
                val = output[0] if isinstance(output, tuple) else output
                # Send to CPU to save VRAM
                cache[name] = val.detach().cpu().clone()
            return hook
            
        for i, layer in enumerate(layer_container):
            hooks.append(layer.register_forward_hook(make_fwd_hook(f"layer_{i}")))
            
        with torch.no_grad():
            logits = hf_fp16(input_ids).logits
            
        # Target token from FP16 truth
        target_token = logits[0, -1].argmax().item()
        target_tokens[data['id']] = target_token
        fp16_cpu_cache[data['id']] = cache
        
        for h in hooks:
            h.remove()
            
        del logits
        cleanup_vram()

    # WIPE VRAM OF THE ENTIRE FP16 MODEL
    print("\nTeardown FP16 architecture...")
    del hf_fp16
    del layer_container
    cleanup_vram()
    print(f"[VRAM Check] Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\n========================================")
    print("STEP 2: NATIVE NF4 BACKPROP & ATP SCORING")
    print("========================================")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    hf_nf4 = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        quantization_config=bnb_config,
        device_map="cuda"
    )
    layer_container = hf_nf4.model.language_model.layers
    
    total_attribution = {f"layer_{i}": 0.0 for i in range(n_layers)}

    for data in tqdm(dataset, desc="NF4 Attribution Patching"):
        p_id = data['id']
        input_ids = tokenizer(data['prompt'], return_tensors="pt").input_ids.to("cuda")
        target_tok = target_tokens[p_id]
        
        fwd_cache = {}
        bwd_cache = {}
        hooks = []
        
        def make_fwd_hook(name):
            def hook(module, inp, output):
                val = output[0] if isinstance(output, tuple) else output
                fwd_cache[name] = val.detach()
            return hook
            
        def make_bwd_hook(name):
            def hook(module, grad_inp, grad_out):
                val = grad_out[0]
                if val is not None:
                    bwd_cache[name] = val.detach()
            return hook

        for i, layer in enumerate(layer_container):
            hooks.append(layer.register_forward_hook(make_fwd_hook(f"layer_{i}")))
            hooks.append(layer.register_full_backward_hook(make_bwd_hook(f"layer_{i}")))
            
        logits = hf_nf4(input_ids).logits
        
        # Loss Targeting
        log_probs = torch.nn.functional.log_softmax(logits[:, -1, :], dim=-1)
        loss = -log_probs[0, target_tok]
        
        hf_nf4.zero_grad()
        loss.backward()
        
        # Calculate Causal taylor effect
        clean_cache = fp16_cpu_cache[p_id]
        
        for k in total_attribution.keys():
            if k in fwd_cache and k in bwd_cache and k in clean_cache:
                A_clean = clean_cache[k].to("cuda")
                A_corrupt = fwd_cache[k]
                grad_A_corrupt = bwd_cache[k]
                
                # Formula: Effect = \nabla A_corrupt \cdot (A_clean - A_corrupt)
                delta_A = A_clean - A_corrupt
                # Ensure shapes match precisely, compute dot product approximations
                score = (grad_A_corrupt * delta_A).sum(dim=-1).mean().item()
                total_attribution[k] += score
                
                del A_clean
                del delta_A
                
        for h in hooks:
            h.remove()
            
        hf_nf4.zero_grad()
        del logits
        del loss
        del fwd_cache
        del bwd_cache
        cleanup_vram()

    dataset_len = len(dataset)
    for k in total_attribution:
        total_attribution[k] /= dataset_len
        
    print(f"\nFinal VRAM State: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(total_attribution, f, indent=2)
        
    print(f"\nSuccessfully mapped causal attributions to {OUTPUT_PATH}")

if __name__ == "__main__":
    run_atp_pipeline()
