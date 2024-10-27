# # from transformers import AutoTokenizer, AutoModelForSequenceClassification
# # import torch
# # from typing import Tuple, List

# # device = "cuda:0" if torch.cuda.is_available() else "cpu"

# # # Load the FinBERT model and tokenizer
# # tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
# # model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
# # labels = ["positive", "negative", "neutral"]

# # def estimate_sentiment(news: List[str], neutral_threshold: float = 0.6) -> Tuple[float, str]:
# #     if not news:
# #         return 0.0, "neutral"
    
# #     total_probs = torch.zeros(3, device=device)  # Sum probabilities for each class

# #     for headline in news:
# #         tokens = tokenizer(headline, return_tensors="pt", padding=True, truncation=True).to(device)
# #         logits = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]
# #         probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()  # Apply softmax to get probabilities
        
# #         total_probs += probs  # Sum probabilities for all headlines

# #     # Average the probabilities
# #     avg_probs = total_probs / len(news)
# #     max_prob, max_idx = torch.max(avg_probs, dim=-1)
    
# #     # Assign "neutral" only if the probability for the other classes is not strong
# #     if labels[max_idx] == "neutral" and max_prob < neutral_threshold:
# #         # If max class is "neutral" and probability is below threshold, classify as neutral
# #         sentiment = "neutral"
# #     else:
# #         # Otherwise, choose the class with the highest average probability
# #         sentiment = labels[max_idx]
    
# #     return max_prob.item(), sentiment

# # # Test the function with sample news headlines
# # if __name__ == "__main__":
# #     test_news = ['markets responded negatively to the news!', 'traders were displeased!']
# #     probability, sentiment = estimate_sentiment(test_news)
# #     print("Sentiment Probability:", probability, "Sentiment:", sentiment)
# #     print("CUDA Available:", torch.cuda.is_available())

# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# from typing import Tuple, List

# device = "cuda:0" if torch.cuda.is_available() else "cpu"

# # Load the FinBERT model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
# model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
# labels = ["positive", "negative", "neutral"]

# def estimate_sentiment(news: List[str], neutral_threshold: float = 0.4, pos_neg_threshold: float = 0.35) -> Tuple[float, str]:
#     if not news:
#         return 0.0, "neutral"
    
#     total_probs = torch.zeros(3, device=device)  # Sum probabilities for each class

#     for i, headline in enumerate(news):
#         # Weigh more recent headlines more heavily with a higher weight factor
#         weight = 1 + (0.2 * (len(news) - i))  # Adjusted weight factor
#         tokens = tokenizer(headline, return_tensors="pt", padding=True, truncation=True).to(device)
#         logits = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]
#         probs = torch.nn.functional.softmax(logits, dim=-1).squeeze() * weight  # Apply weight
        
#         total_probs += probs  # Sum weighted probabilities for all headlines

#     # Average the weighted probabilities
#     avg_probs = total_probs / sum([1 + (0.2 * (len(news) - i)) for i in range(len(news))])  # Adjusted weight factor
#     max_prob, max_idx = torch.max(avg_probs, dim=-1)
    
#     # Use thresholds to classify sentiment
#     if labels[max_idx] == "neutral" and max_prob < neutral_threshold:
#         sentiment = "neutral"
#     elif max_prob >= pos_neg_threshold:  # Favor positive or negative if threshold met
#         sentiment = labels[max_idx]
#     else:
#         sentiment = "neutral"  # Default to neutral if no strong signal
    
#     return max_prob.item(), sentiment

# # Test the function with sample news headlines
# if __name__ == "__main__":
#     test_news = [
#         "markets responded negatively to the news!", 
#         "traders were displeased!", 
#         "stock prices are expected to rise"
#     ]
#     probability, sentiment = estimate_sentiment(test_news)
#     print("Sentiment Probability:", probability, "Sentiment:", sentiment)
#     print("CUDA Available:", torch.cuda.is_available())


# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# from typing import Tuple, List

# device = "cuda:0" if torch.cuda.is_available() else "cpu"

# # Load the FinBERT model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
# model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
# labels = ["positive", "negative", "neutral"]

# # Updated threshold and weighting adjustments
# def estimate_sentiment(
#     news: List[str], 
#     neutral_threshold: float = 0.15,    # Lowered neutral threshold for sensitivity
#     pos_neg_threshold: float = 0.8,    # Adjusted pos/neg threshold for greater variance
#     recency_weight: float = 1.2        # Increased weight for recent headlines
# ) -> Tuple[float, str]:
#     if not news:
#         return 0.0, "neutral"
    
#     total_probs = torch.zeros(3, device=device)  # Sum probabilities for each class

#     for i, headline in enumerate(news):
#         # Enhanced weighting based on recency
#         weight = recency_weight + (0.05 * (len(news) - i))  # Higher weight increment per item
#         tokens = tokenizer(headline, return_tensors="pt", padding=True, truncation=True).to(device)
#         logits = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]
#         probs = torch.nn.functional.softmax(logits, dim=-1).squeeze() * weight  # Apply weight

#         # Keyword sensitivity boost
#         if any(kw in headline.lower() for kw in ["surge", "drop", "rally", "crisis"]):
#             probs[labels.index("positive")] += 0.05
#             probs[labels.index("negative")] += 0.05

#         total_probs += probs  # Sum weighted probabilities for all headlines

#     # Average the weighted probabilities
#     avg_probs = total_probs / sum([recency_weight + (0.05 * (len(news) - i)) for i in range(len(news))])
#     max_prob, max_idx = torch.max(avg_probs, dim=-1)
    
#     # Use thresholds to classify sentiment
#     if labels[max_idx] == "neutral" and max_prob < neutral_threshold:
#         sentiment = "neutral"
#     elif max_prob >= pos_neg_threshold:  # Favor positive or negative if threshold met
#         sentiment = labels[max_idx]
#     else:
#         sentiment = "neutral"  # Default to neutral if no strong signal
    
#     return max_prob.item(), sentiment

# # Test the function with sample news headlines
# if __name__ == "__main__":
#     test_news = [
#         "Markets rallied strongly on the latest earnings report!",
#         "Investors are optimistic despite recent challenges.",
#         "A crisis could impact stocks heavily.",
#         "Surge in demand expected as the holiday season approaches."
#     ]
#     tensor, sentiment = estimate_sentiment(['Dollar General Aims To Draw Customers From Walmart, Amazon With Holiday Sales Events'])
#     print(tensor, sentiment)
#     print("Sentiment Probability:", tensor, "Sentiment:", sentiment)
#     print("CUDA Available:", torch.cuda.is_available())


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
    """ Estimate sentiment with added preprocessing steps for API-fetched news """
    if not news:
        return 0.0, "neutral"
    
    total_probs = torch.zeros(3, device=device)  # Sum probabilities for each class

    for i, headline in enumerate(news):
        # Preprocess headline for consistency
        cleaned_headline = clean_headline(headline)
        tokens = tokenizer(cleaned_headline, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # Check tokenization result for debugging purposes
        print(f"Original headline: {headline}")
        print(f"Cleaned headline: {cleaned_headline}")
        print(f"Tokenized input: {tokens['input_ids']}")

        logits = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]
        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()
        
        total_probs += probs  # Sum probabilities for all headlines

    # Average the probabilities
    avg_probs = total_probs / len(news)
    max_prob, max_idx = torch.max(avg_probs, dim=-1)
    
    # Classify sentiment based on threshold
    if labels[max_idx] == "neutral" and max_prob < neutral_threshold:
        sentiment = "neutral"
    elif max_prob >= pos_neg_threshold:
        sentiment = labels[max_idx]
    else:
        sentiment = "neutral"
    
    return max_prob.item(), sentiment

# Test function with API-fetched news for debugging
if __name__ == "__main__":
    test_news = [
        "Google plans to announce its next Gemini model soon, creating competition with OpenAI.",
        "The market responded negatively to new tech policies.",
        "A recent acquisition by Amazon could affect the e-commerce sector."
    ]
    probability, sentiment = estimate_sentiment(test_news)
    print("Sentiment Probability:", probability, "Sentiment:", sentiment)
    print("CUDA Available:", torch.cuda.is_available())
