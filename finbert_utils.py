from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple, List

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the FinBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
labels = ["positive", "negative", "neutral"]

def estimate_sentiment(news: List[str], neutral_threshold: float = 0.6) -> Tuple[float, str]:
    if not news:
        return 0.0, "neutral"
    
    total_probs = torch.zeros(3, device=device)  # Sum probabilities for each class

    for headline in news:
        tokens = tokenizer(headline, return_tensors="pt", padding=True, truncation=True).to(device)
        logits = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]
        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()  # Apply softmax to get probabilities
        
        total_probs += probs  # Sum probabilities for all headlines

    # Average the probabilities
    avg_probs = total_probs / len(news)
    max_prob, max_idx = torch.max(avg_probs, dim=-1)
    
    # Assign "neutral" only if the probability for the other classes is not strong
    if labels[max_idx] == "neutral" and max_prob < neutral_threshold:
        # If max class is "neutral" and probability is below threshold, classify as neutral
        sentiment = "neutral"
    else:
        # Otherwise, choose the class with the highest average probability
        sentiment = labels[max_idx]
    
    return max_prob.item(), sentiment

# Test the function with sample news headlines
if __name__ == "__main__":
    test_news = ['markets responded negatively to the news!', 'traders were displeased!']
    probability, sentiment = estimate_sentiment(test_news)
    print("Sentiment Probability:", probability, "Sentiment:", sentiment)
    print("CUDA Available:", torch.cuda.is_available())
