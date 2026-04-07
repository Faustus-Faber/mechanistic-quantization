import os
import json
import torch
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import argparse
import csv

# ==========================================
# PHASE 1: DUAL-MODEL INFERENCE ENGINE
# ==========================================
# This script executes the same 1,500 deterministic prompts across two 
# identically structured representations of Gemma 4 E2B:
# 1. fp16 Uncompressed Baseline
# 2. bnb nf4 4-bit Quantized Baseline
# The resulting outputs are saved sequentially to avoid VRAM overload.

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "gemma-4-e2b-it")

def load_data(limit=None):
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw_prompts.json")
    with open(data_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    if limit:
        return prompts[:limit]
    return prompts

def run_inference_loop(model, tokenizer, prompts):
    results = []
    
    for item in tqdm(prompts, desc=f"Generating"):
        prompt_text = item["prompt"]
        
        try:
            messages = [{"role": "user", "content": prompt_text}]
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
            
            # Deterministic greedy generation
            with torch.no_grad():
                model.config.pad_token_id = tokenizer.eos_token_id
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150, 
                    do_sample=False, 
                    repetition_penalty=1.15,
                    temperature=0.0
                )
            
            # Extract only the generated response
            generated_token_ids = outputs[0][inputs["input_ids"].shape[-1]:]
            response = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()
            
            results.append({
                "id": item["id"],
                "language": item["language"],
                "task_type": item["task_type"],
                "output": response
            })
            
        except Exception as e:
            import traceback
            print(f"\nCRASH on prompt {item['id']}: {traceback.format_exc()}\n")
            results.append({
                "id": item["id"],
                "error": str(e)
            })
            
        # Aggressive memory clearing per prompt
        torch.cuda.empty_cache()
        gc.collect()
        
    return {r["id"]: r.get("output", "ERROR") for r in results}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5, help="Number of prompts to run for prototyping")
    args = parser.parse_args()

    prompts = load_data(limit=args.limit)
    print(f"Loaded {len(prompts)} prompts. Setting up CUDA...")
    
    final_results = {p["id"]: {"prompt": p["prompt"], "language": p["language"], "task": p["task_type"]} for p in prompts}

    # ==========================
    # RUN 1: FP16 BASELINE
    # ==========================
    print("\n--- INITIALIZING FP16 BASELINE MODEL ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    fp16_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="cuda:0",
        dtype=torch.bfloat16
    )
    
    fp16_outputs = run_inference_loop(fp16_model, tokenizer, prompts)
    for p_id, out in fp16_outputs.items():
        if p_id in final_results:
            final_results[p_id]["fp16_output"] = out
            
    # Purge VRAM for safety mapping
    del fp16_model
    torch.cuda.empty_cache()
    gc.collect()
    print("Flushed FP16 Model from GPU.")

    # ==========================
    # RUN 2: NF4 QUANTIZED MODEL
    # ==========================
    print("\n--- INITIALIZING NF4 QUANTIZED MODEL ---")
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
    
    nf4_outputs = run_inference_loop(nf4_model, tokenizer, prompts)
    for p_id, out in nf4_outputs.items():
        if p_id in final_results:
            final_results[p_id]["nf4_output"] = out
            
    # Save the paired dataset primarily as JSON
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "inference_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(list(final_results.values()), f, ensure_ascii=False, indent=2)
        
    # Save synchronously as CSV for spreadsheet analysis
    csv_output_path = os.path.join(os.path.dirname(__file__), "..", "data", "inference_results.csv")
    with open(csv_output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "language", "task", "prompt", "fp16_output", "nf4_output"])
        for p_id, item in final_results.items():
            writer.writerow([
                p_id, 
                item.get("language", ""), 
                item.get("task", ""), 
                item.get("prompt", ""), 
                item.get("fp16_output", "ERROR"), 
                item.get("nf4_output", "ERROR")
            ])
            
    print(f"\nSuccessfully evaluated {len(final_results)} sequences! Pairs saved to {output_path} and {csv_output_path}")

if __name__ == "__main__":
    main()
