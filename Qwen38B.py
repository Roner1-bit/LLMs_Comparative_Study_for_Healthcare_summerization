import os
import argparse
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name: str):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()
    return tokenizer, model

def summarize_text(tokenizer, model, text: str, 
                   max_new_tokens: int = 4096,
                   temperature: float = 0.7,  
                   top_p: float = 0.8,
                   top_k: int = 20):
    """
    Generate a concise summary for the given text.
    Uses non-thinking mode for efficiency.
    """
  
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes transcripts."},
        {"role": "user", "content": f"Please provide a concise summary of the following transcript:\n\n{text}"}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    # Generate summary
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=True
    )
    # Extract only the newly generated tokens
    output_ids = generated[0][inputs.input_ids.shape[-1]:].tolist()
    summary = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return summary

def main():

    input_csv = "QA_test_set.csv"
    output_csv = "QA_test_set_with_summaries_of_QWEN_3_8B.csv"
    model_name = "Qwen/Qwen3-8B"


    # Load data
    df = pd.read_csv(input_csv)


    # Load model & tokenizer
    print(f"Loading model {model_name} ...")
    tokenizer, model = load_model(model_name)

    # Prepare output file mode/header
    write_header = not os.path.exists(output_csv)
    mode = "w" if write_header else "a"

    # Process and summarize each transcript
    records = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Summarizing"):
        text = str(row["transcript"])
        summary = summarize_text(tokenizer, model, text)
        records.append({
            **row.to_dict(),
            "summary": summary
        })

    # Write results
    out_df = pd.DataFrame(records)
    out_df.to_csv(
        output_csv,
        mode=mode,
        header=write_header,
        index=False
    )
    print(f"Summaries written to {output_csv}.")

if __name__ == "__main__":
    main()