from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load FinBERT model for tone analysis
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

def classify_document(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    # Softmax to get probabilities for each class (e.g., positive, negative, neutral)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probabilities.tolist()  # Convert tensor to list for easier use
