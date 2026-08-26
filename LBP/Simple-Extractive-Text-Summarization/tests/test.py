import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from tqdm import tqdm
import logging
import time
from rouge_score import rouge_scorer

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    sentences = sent_tokenize(text)
    clean_sentences = []
    
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        clean_words = [word for word in words if word.isalnum() and word not in stop_words]
        clean_sentences.append(" ".join(clean_words))
        
    return clean_sentences, sentences

def summarize_text(text, num_sentences=3):
    try:
        clean_sentences, original_sentences = preprocess_text(text)
        
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(clean_sentences)
        
        cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        sentence_scores = cosine_similarities.sum(axis=1)
        
        top_sentence_indices = heapq.nlargest(num_sentences, range(len(sentence_scores)), key=sentence_scores.take)
        
        summary = [original_sentences[i] for i in sorted(top_sentence_indices)]
        
        return " ".join(summary)
    except Exception as e:
        logging.error(f"Error summarizing text: {e}")
        return ""

def compute_rouge(reference, generated, scorer):
    try:
        scores = scorer.score(reference, generated)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }
    except Exception as e:
        logging.error(f"Error computing ROUGE: {e}")
        return {"rouge1": 0, "rouge2": 0, "rougeL": 0}

def process_dataset(input_path, output_path, scorer, num_sentences=3):
    start_time = time.time()

    data = pd.read_csv(input_path)

    if "article" not in data.columns or "summary" not in data.columns:
        raise ValueError("Input CSV must contain 'article' and 'summary' columns")

    results = []

    for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing articles"):
        article = row.get("article", "")
        reference_summary = row.get("summary", "")
        
        generated_summary = summarize_text(article, num_sentences)

        rouge_scores = compute_rouge(reference_summary, generated_summary, scorer)

        results.append({
            "article": article,
            "reference_summary": reference_summary,
            "generated_summary": generated_summary,
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "rougeL": rouge_scores["rougeL"]
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(output_path, index=False)
    logging.info(f"Summarization and ROUGE evaluation results saved to {output_path}")

    end_time = time.time()
    logging.info(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    input_path = "cnn_articles.csv"  # Input file
    output_path = "tfidf_summary_with_rouge.csv"  # Output file
    num_sentences = 3  # Number of sentences in the summary

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    process_dataset(input_path, output_path, scorer, num_sentences=num_sentences)
