import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple, List

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the FinBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
labels = ["positive", "negative", "neutral"]

def clean_headline(headline: str) -> str:
    """ Clean and standardize headlines for sentiment analysis """
    # Remove punctuation, lowercase text, and strip whitespace
    return re.sub(r"[^\w\s]", "", headline.lower()).strip()

def estimate_sentiment(news: List[str], neutral_threshold: float = 0.4, pos_neg_threshold: float = 0.3) -> Tuple[float, str]:
    # """ Estimate sentiment with added preprocessing steps for API-fetched news """
    # if not news:
    #     return 0.0, "neutral"
    
    # total_probs = torch.zeros(3, device=device)  # Sum probabilities for each class

    # for i, headline in enumerate(news):
    #     # Preprocess headline for consistency
    #     cleaned_headline = clean_headline(headline)
    #     tokens = tokenizer(cleaned_headline, return_tensors="pt", padding=True, truncation=True).to(device)
        
    #     # Check tokenization result for debugging purposes
    #     # print(f"Original headline: {headline}")
    #     # print(f"Cleaned headline: {cleaned_headline}")
    #     # print(f"Tokenized input: {tokens['input_ids']}")

    #     logits = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]
    #     probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()
        
    #     total_probs += probs  # Sum probabilities for all headlines

    # # Average the probabilities
    # avg_probs = total_probs / len(news)
    # max_prob, max_idx = torch.max(avg_probs, dim=-1)
    
    # # Classify sentiment based on threshold
    # if labels[max_idx] == "neutral" and max_prob < neutral_threshold:
    #     sentiment = "neutral"
    # elif max_prob >= pos_neg_threshold:
    #     sentiment = labels[max_idx]
    # else:
    #     sentiment = "neutral"
    
    # return max_prob.item(), sentiment
    if news:
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)

        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])[
            "logits"
        ]
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)
        probability = result[torch.argmax(result)]
        sentiment = labels[torch.argmax(result)]
        return probability, sentiment
    else:
        return 0, labels[-1]

# Test function with API-fetched news for debugging
if __name__ == "__main__":
    # test_news = [
    #     "Google plans to announce its next Gemini model soon, creating competition with OpenAI.",
    #     "The market responded negatively to new tech policies.",
    #     "A recent acquisition by Amazon could affect the e-commerce sector."
    # ]
    # probability, sentiment = estimate_sentiment(test_news)
    # print("Sentiment Probability:", probability, "Sentiment:", sentiment)
    pass