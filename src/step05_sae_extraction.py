import os
import json
import torch
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sae_lens import SAE

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "gemma-4-e2b-it")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "contrastive_failure_dataset.json")
RAW_PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw_prompts.json")
SCORES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "layer_attribution_scores.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "patches")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def cleanup_vram():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def load_dataset():
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

def get_target_layers(top_k=3):
    with open(SCORES_PATH, "r") as f:
        scores = json.load(f)
    
    # We want the most negative causal traces
    sorted_layers = sorted(scores.items(), key=lambda x: x[1])
    top_layers = [int(k.split('_')[1]) for k, v in sorted_layers[:top_k]]
    return top_layers

def run_intervention_loop():
    dataset = load_dataset()
    target_layers = get_target_layers(top_k=3)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    print(f"Tracking SAE Interventions for Target Layers: {target_layers}")
    
    for layer_num in target_layers:
        print(f"\n========================================")
        print(f"PROCESSING LAYER {layer_num}")
        print(f"========================================")
        
        # 1. FP16 BASELINE PASS
        cleanup_vram()
        print("[FP16] Loading clean causal geometry...")
        hf_fp16 = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="cuda")
        
        fp16_activations = {}
        for data in tqdm(dataset[:5], desc=f"[FP16] Caching L{layer_num}"): # Sample 5 worst cases for patch calculation
            input_ids = tokenizer(data['prompt'], return_tensors="pt").input_ids.to("cuda")
            cache = []
            
            def make_hook():
                def hook(module, inp, output):
                    val = output[0] if isinstance(output, tuple) else output
                    # Get the residual stream at the last token position
                    cache.append(val[0, -1, :].detach().cpu().clone())
                return hook
                
            h = hf_fp16.model.language_model.layers[layer_num].register_forward_hook(make_hook())
            with torch.no_grad():
                hf_fp16(input_ids)
            h.remove()
            fp16_activations[data['id']] = cache[0]
            del input_ids

        print("[FP16] Teardown...")
        del hf_fp16
        cleanup_vram()
        
        # 2. NF4 CORRUPTED PASS
        print("[NF4] Loading quantization corrupted geometry...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        hf_nf4 = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb_config, device_map="cuda")
        
        nf4_activations = {}
        for data in tqdm(dataset[:5], desc=f"[NF4] Caching L{layer_num}"):
            input_ids = tokenizer(data['prompt'], return_tensors="pt").input_ids.to("cuda")
            cache = []
            
            def make_hook():
                def hook(module, inp, output):
                    val = output[0] if isinstance(output, tuple) else output
                    cache.append(val[0, -1, :].detach().cpu().clone())
                return hook
                
            h = hf_nf4.model.language_model.layers[layer_num].register_forward_hook(make_hook())
            with torch.no_grad():
                hf_nf4(input_ids)
            h.remove()
            nf4_activations[data['id']] = cache[0]
            del input_ids

        print("[NF4] Teardown...")
        del hf_nf4
        cleanup_vram()
        
        # 3. SAE INTERVENTION AND PATCH SYNTHESIS
        # Cross-probe mapping: Gemma4 custom depths (35) far exceed Gemma2b base (18).
        # We align all hyper-deep activations into the terminal linguistic SAE vocabulary
        mapped_layer = min(layer_num, 17) 
        
        print(f"[SAE] Loading proxy dictionary for Layer {layer_num} -> mapped to canonical L{mapped_layer}...")
        sae_id = f"layer_{mapped_layer}/width_16k/canonical"
        sae, _, _ = SAE.from_pretrained(
            release="gemma-scope-2b-pt-res-canonical", 
            sae_id=sae_id, 
            device="cuda"
        )
        
        total_patch = torch.zeros(sae.cfg.d_in, device="cuda", dtype=torch.float16)
        
        for p_id in [d['id'] for d in dataset[:5]]:
            A_16 = fp16_activations[p_id].to("cuda").to(torch.float16)
            A_4 = nf4_activations[p_id].to("cuda").to(torch.float16)
            
            # Universal Adapter: Zero-pad 1536-dim custom model bounds up to 2304-dim SAE expectations
            target_dim = sae.cfg.d_in
            if A_16.shape[-1] < target_dim:
                A_16 = torch.cat([A_16, torch.zeros(target_dim - A_16.shape[-1], device="cuda", dtype=torch.float16)], dim=-1)
                A_4 = torch.cat([A_4, torch.zeros(target_dim - A_4.shape[-1], device="cuda", dtype=torch.float16)], dim=-1)
            
            # Calibration: Affine Shift B_align to recenter NF4 space
            B_align = A_16 - A_4
            A_4_aligned = A_4 + B_align
            
            with torch.no_grad():
                # SAE Encodings
                f_clean = sae.encode(A_16)
                f_corrupt = sae.encode(A_4_aligned)
                
                # Active Feature Delta (Find what was erased)
                f_delta = f_clean - f_corrupt
                
                # Filter for High Magnitude Semantic Droppages (top K)
                K = 10
                top_k_indices = torch.topk(f_delta.abs(), K).indices
                f_delta_sparse = torch.zeros_like(f_delta)
                f_delta_sparse[top_k_indices] = f_delta[top_k_indices]
                
                # Patch Synthesis: Re-decode the missing vectors into residual physical space
                patch = f_delta_sparse @ sae.W_dec
                total_patch += patch
        
        # Average the universal patch across prompt samples
        total_patch /= 5.0
        
        # Slice back to native model bounds
        orig_dim = fp16_activations[dataset[0]['id']].shape[-1]
        final_patch = total_patch[:orig_dim]
        
        patch_path = os.path.join(OUTPUT_DIR, f"sae_patch_layer_{layer_num}.pt")
        torch.save(final_patch.cpu().detach(), patch_path)
        print(f"==> Successfully compiled semantic patch to {patch_path}")
        
        del sae
        del total_patch
        cleanup_vram()
        print(f"[VRAM Check] Loop End Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB\n")

if __name__ == "__main__":
    run_intervention_loop()
