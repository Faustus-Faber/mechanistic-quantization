import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
from step06_bias_injector import apply_sae_patches

def test_inference(model, tokenizer, prompt, mode_name):
    print(f"\n--- TESTING {mode_name.upper()} GENERATION ---")
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256,
            do_sample=False
        )
        
    gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(gen_text)
    print("-" * 40)

def main():
    print("Loading purely isolated dual Gemma generation test...")
    MODEL_PATH = "models/gemma-4-e2b-it"
    prompt = "Return valid Python source code that parses the following deeply nested JSON structure into typed dataclasses.\n\n<input>\n{\"Sensor\": {\"uid\": 34, \"metadata\": {\"active\": true}}}\n</input>\n\n<constraint>\nProvide only the Python code.\n</constraint>"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # 1. TEST BFLOAT16 BASELINE
    print("\nLoading BFloat16 model...")
    bf16_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="cuda:0",
        dtype=torch.bfloat16
    )
    test_inference(bf16_model, tokenizer, prompt, "BFloat16")
    
    # Clear VRAM
    del bf16_model
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. TEST NF4 QUANTIZATION
    print("\nLoading NF4 Quantized model...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    nf4_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="cuda:0",
        quantization_config=quantization_config
    )
    test_inference(nf4_model, tokenizer, prompt, "NF4 Quantized")
    
    # 3. TEST PATCHED NF4 (WEIGHT-FOLDED)
    print("\nApplying SAE Patches to NF4 Model...")
    patched_model = apply_sae_patches(nf4_model, layers_to_patch=(29, 30, 31))
    test_inference(patched_model, tokenizer, prompt, "Patched NF4")


if __name__ == "__main__":
    main()
