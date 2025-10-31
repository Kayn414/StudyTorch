 

import pandas as pd
import pickle
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.utils import resample
from sklearn.metrics import classification_report



# def info(reviews_df):
# 	print(reviews_df.head())
# 	print(reviews_df.columns.tolist())
	# reviews_df.describe()
	# reviews_df.info()
	# print(reviews_df.isnull().sum())  
    # print(reviews_df.dtypes)

positive_words = {"good", "great", "amazing", "fantastic", "love", "excellent", "brilliant", "masterpiece"}
negative_words = {"bad", "boring", "terrible", "hate", "awful", "dull", "mediocre", "forgettable", "poor"}

def sentiment(review):
	words = re.findall(r'\b\w+\b', review.lower())
	score = sum(1 for w in words if w in positive_words) - sum(1 for w in words if w in negative_words)
	return score

def scale_score(arr_score):
	return (arr_score - arr_score.min()) / (arr_score.max() - arr_score.min())


def merge_datasets(reviews_df, movie_df):
    merged_df = reviews_df.merge(
        movie_df[['id', 'title','year']],   # Keep only id and title from movies
        left_on='imdb_id',
        right_on='id',
        how='left'
    )
    return merged_df

def train(reviews_df):
    # Fill missing values
    reviews_df.fillna(0, inplace=True)
    reviews_df.rename(columns={'review title': 'review_title', 'review_rating': 'user_rating'}, inplace=True)

    # Compute sentiment score
    reviews_df['sentiment_score'] = reviews_df['review'].apply(sentiment)

    # Numeric features
    user_score= reviews_df['user_rating']
    review_length_norm = scale_score(reviews_df['review'].apply(lambda x: len(str(x).split())))
    year = reviews_df['year']

    # Combine features
    X_features = np.column_stack((
        user_score,
        reviews_df['sentiment_score'],
        review_length_norm,
        year
    ))

    # Label
    y_label = ((user_score >= 0.7) & 
               (reviews_df['sentiment_score'] >= 2) & 
               (reviews_df['oscar_winner'] == 'Yes')).astype(int)

    # Imbalance handling
    df_majority = reviews_df[y_label == 1]
    df_minority = reviews_df[y_label == 0]

    if len(df_minority) < len(df_majority):
        df_minority_upsampled = resample(df_minority,
                                         replace=True,
                                         n_samples=len(df_majority),
                                         random_state=42)
        balanced_df = pd.concat([df_majority, df_minority_upsampled])
    else:
        balanced_df = reviews_df.copy()

    # Recompute features after upsampling
    X_features_bal = np.column_stack((
        scale_score(balanced_df['user_rating']),
        balanced_df['sentiment_score'],
        scale_score(balanced_df['review'].apply(lambda x: len(str(x).split()))),
        scale_score(balanced_df['year'])
    ))
    y_label_bal = ((scale_score(balanced_df['user_rating']) >= 0.8) & 
                   (balanced_df['sentiment_score'] >= 2) & 
                   (balanced_df['oscar_winner'] == 'Yes')).astype(int)

   
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_features_bal, y_label, test_size=0.4, random_state=42,)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)

    
    clf = SGDClassifier(loss='log_loss',
                    penalty='l2',
                    max_iter=500,
                    class_weight='balanced',
                    random_state=42)

    clf.fit(X_train, y_train)

    with open("model.dat", "wb") as f:
        pickle.dump(clf, f)

    return clf, X_dev, y_dev, X_test, y_test



def main():
	# Use the absolute path if you're unable to load the dataset.
	# reviews_path = r"path/to/file" 
    try:
        reviews_df = pd.read_csv("imdb_reviews_with_oscars.csv", encoding="utf-8")
        movie_df = pd.read_csv("imdb_list.csv", encoding="utf-8")
        print("loaded dataset")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    

    # Merge datasets
    merged_df = merge_datasets(reviews_df, movie_df)

    # info(merged_df)

    # Train classifier
    clf, X_dev, y_dev, X_test, y_test = train(merged_df)

    # Evaluate
    print("Dev accuracy:", clf.score(X_dev, y_dev))
    print("Test accuracy:", clf.score(X_test, y_test))
    # y_pred = clf.predict(X_test)
    # print(classification_report(y_test, y_pred))
    

if __name__ == "__main__":
	main()
