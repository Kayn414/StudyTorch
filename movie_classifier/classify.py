import sys
import pickle
import re
import numpy as np

# Sentiment words
positive_words = {"good", "great", "amazing", "fantastic", "love", "excellent", "brilliant", "masterpiece"}
negative_words = {"bad", "boring", "terrible", "hate", "awful", "dull", "mediocre", "forgettable", "poor"}

def sentiment(review):
    words = re.findall(r'\b\w+\b', str(review).lower())
    score = sum(1 for w in words if w in positive_words) - sum(1 for w in words if w in negative_words)
    return score

def scale_score(arr_score):
    arr_score = np.array(arr_score, dtype=float)
    if arr_score.max() == arr_score.min():
        return np.zeros_like(arr_score)
    return (arr_score - arr_score.min()) / (arr_score.max() - arr_score.min())

def extract_rating(text):
    matches = re.findall(r'\b([0-9](?:\.[0-9])?)\b', text)
    if matches:
        rating = float(matches[0])
        return max(0, min(10, rating))
    return 5.0

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 classify.py model.dat test.txt")
        sys.exit(1)

    model_f = sys.argv[1]
    doc_f = sys.argv[2]

    # Load model
    with open(model_f, "rb") as f:
        clf = pickle.load(f)

    # Load document
    with open(doc_f, "r", encoding="utf-8") as f:
        text = f.read()

    # Features
    # Optional: year feature (if known, default 2000)
    year_norm = scale_score([2000])[0] 
    user_rating = extract_rating(text)
    user_score = user_rating / 10.0          # keep meaning (0–1)
    sentiment_score = sentiment(text)        # keep raw integer
    review_length = len(text.split())
    review_length_norm = np.log1p(review_length)  # log scaling helps shrink outliers
    # year_norm = (year - year_min) / (year_max - year_min)  # min–max scale

    
    
    X_test = np.array([[user_score, sentiment_score, review_length_norm, year_norm]])

    # Predict class
    pred = clf.predict(X_test)[0]
    print(pred)

    label = "Oscar candidate" if pred == 1 else "Not an Oscar candidate"
    print(f"Document classification: {label}")

if __name__ == "__main__":
    main()
