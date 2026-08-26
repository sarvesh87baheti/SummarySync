import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import heapq

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    sentences = sent_tokenize(text)
    clean_sentences = []
    
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        clean_words = [word for word in words if word.isalnum() and word not in stop_words]
        clean_sentences.append(" ".join(clean_words))
        
    return clean_sentences, sentences

# Summarization function
def summarize_text(text, num_sentences=3):
    clean_sentences, original_sentences = preprocess_text(text)
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(clean_sentences)
    
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    sentence_scores = cosine_similarities.sum(axis=1)
    
    top_sentence_indices = heapq.nlargest(num_sentences, range(len(sentence_scores)), key=sentence_scores.take)
    
    summary = [original_sentences[i] for i in sorted(top_sentence_indices)]
    
    return " ".join(summary)

# Take user input
text = input("Enter the text to summarize:\n")
num_sentences = int(input("Enter the number of sentences for summary: "))

# Generate and print summary
summary = summarize_text(text, num_sentences)
print("\nSummary:\n", summary)
