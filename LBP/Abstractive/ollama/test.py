import pandas as pd
from langchain_ollama import OllamaLLM
from rouge_score import rouge_scorer
from tqdm import tqdm
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

llm = OllamaLLM(model="phi3", device="cuda")  

def summarize_text(text, llm):
    try:
        prompt = f"Summarize the following text:\n\n{text}\n\nSummary:"
        response = llm.invoke(prompt)  
        return response.strip()
    except Exception as e:
        logging.error(f"Error summarizing text: {e}")
        return ""

def compute_rouge(reference, generated, scorer):
    try:
        return scorer.score(reference, generated)
    except Exception as e:
        logging.error(f"Error computing ROUGE: {e}")
        return {"rouge1": 0, "rouge2": 0, "rougeL": 0}

def process_dataset(input_path, output_path, llm, scorer, num_articles=None):
    start_time = time.time()

    data = pd.read_csv(input_path)
    # data = pd.read_excel(input_path)

    # Check required columns
    if not all(col in data.columns for col in ["article", "highlights"]):
        raise ValueError("Input CSV must contain 'article' and 'highlights' columns")

    results = []

    for _, row in tqdm(data.iterrows(), total=min(num_articles, len(data)) if num_articles else len(data), desc="Processing articles"):
        if num_articles and len(results) >= num_articles:
            break
        
        article = row.get("article", "")
        article = str(article).strip().replace('\n','') if isinstance(article, str) else ''

        reference_summary = row.get("highlights", "")

        generated_summary = summarize_text(article, llm)
        print(generated_summary)
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
    input_path = "test.csv"
    output_path = "summary_with_rouge.csv"  # Output file
    num_articles = 100 # Adjust as needed
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    process_dataset(input_path, output_path, llm, scorer, num_articles=num_articles)
