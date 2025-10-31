import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

INPUT_CSV = "imdb_reviews.csv"    # change if your file has a different path
OUTPUT_CSV = "imdb_reviews_with_oscars.csv"

HEADERS = {"User-Agent": "Mozilla/5.0"}

def has_oscar_win(imdb_id):
    """Check if a movie with given IMDb ID has at least one Oscar win."""
    url = f"https://www.imdb.com/title/{imdb_id}/awards"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"Error fetching {imdb_id}: {e}")
        return "Unknown"

    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text(" ", strip=True).lower()

    # Look for Academy Awards wins
    if "academy awards" in text or "academy award" in text or "oscar" in text:
        if "winner" in text:
            return "Yes"
        else:
            return "No"
    return "No"

def main():
    df = pd.read_csv(INPUT_CSV)
    unique_ids = df["imdb_id"].dropna().unique()
    print(f"Checking {len(unique_ids)} unique IMDb IDs...")

    results = {}
    for i, imdb_id in enumerate(unique_ids, 1):
        print(f"[{i}/{len(unique_ids)}] {imdb_id}...", end=" ")
        result = has_oscar_win(imdb_id)
        print(result)
        results[imdb_id] = result
        time.sleep(random.uniform(1, 2))  # polite delay

    # Add results back into dataframe
    df["oscar_winner"] = df["imdb_id"].map(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n Saved results to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()