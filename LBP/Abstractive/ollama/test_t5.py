import pandas as pd
from rouge_score import rouge_scorer
from tqdm import tqdm
import logging
import time
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

def summarize_text(text):
    preprocessed_text = text.strip().replace('\n','')
    t5_input_text = 'summarize: ' + preprocessed_text
    tokenized_text = tokenizer.encode(t5_input_text, return_tensors='pt', max_length=512).to(device)
    summary_ids = model.generate(tokenized_text, min_length=30, max_length=120)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary.strip()

def compute_rouge(reference, generated, scorer):
    try:
        return scorer.score(reference, generated)
    except Exception as e:
        logging.error(f"Error computing ROUGE: {e}")
        return {"rouge1": 0, "rouge2": 0, "rougeL": 0}

def process_dataset(input_path, output_path, scorer, num_articles=None):
    start_time = time.time()

    data = pd.read_excel(input_path)


    # Check required columns
    if not all(col in data.columns for col in ["article", "highlights"]):
        raise ValueError("Input CSV must contain 'article' and 'highlights' columns")

    results = []

    for _, row in tqdm(data.iterrows(), total=min(num_articles, len(data)) if num_articles else len(data), desc="Processing articles"):
        if num_articles and len(results) >= num_articles:
            break
        
        article = row.get("article", "")
        reference_summary = row.get("highlights", "")

        generated_summary = summarize_text(article)

        score = compute_rouge(reference_summary, generated_summary, scorer)

        # Append results as a dictionary
        results.append({
            "article": article,
            "reference_summary": reference_summary,
            "generated_summary": generated_summary,
            "rouge1": score["rouge1"].fmeasure,
            "rouge2": score["rouge2"].fmeasure,
            "rougeL": score["rougeL"].fmeasure
        })

    results_df = pd.DataFrame(results)

    # Save results to CSV
    results_df.to_csv(output_path, index=False)
    logging.info(f"Summarization and ROUGE evaluation results saved to {output_path}")

    end_time = time.time()
    logging.info(f"Total processing time: {end_time - start_time:.2f} seconds")

# Run the script
if __name__ == "__main__":
    input_path = "repeated_sentences.xlsx"  # Input file
    output_path = "summary_with_rouge.csv"  # Output file
    num_articles = 100  # Adjust as needed
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    process_dataset(input_path, output_path, scorer, num_articles=num_articles)
