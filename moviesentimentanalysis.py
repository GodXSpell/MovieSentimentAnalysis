import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import fetch_20newsgroups, load_iris
import seaborn as sns
import matplotlib.pyplot as plt

def sentiment_analysis():
    """
    Performs sentiment analysis on a dataset of movie reviews.
    """
    print("\n" + "="*50)
    print("🎬 Sentiment Analysis of Movie Reviews")
    print("="*50 + "\n")

    # Creating a simple dataset of movie reviews (Synthetic)
    # data = {
    #     'review': [
    #         'this movie was fantastic and amazing',
    #         'I absolutely loved this film',
    #         'a masterpiece of cinema',
    #         'great acting and a wonderful story',
    #         'simply brilliant',
    #         'a terrible and boring movie',
    #         'I hated every moment of it',
    #         'the plot was predictable and dull',
    #         'a complete waste of time',
    #         'I would not recommend this to anyone'
    #     ],
    #     'sentiment': [
    #         'positive', 'positive', 'positive', 'positive', 'positive',
    #         'negative', 'negative', 'negative', 'negative', 'negative'
    #     ]
    # }

    # Using a csv file of movies
    # Note the csv is clean and thus it does not need data cleanup
    data = pd.read_csv('movie.csv')
    df = pd.DataFrame(data)
    print("Dataset:")
    print(df)

    for col in df.columns:
      print(f"Column: {col}")
      # print(df[col].unique())
    df['label'].unique()

    # Doing EDA before ML training
    plt.figure()
    df['label'].value_counts().plot(kind='bar')
    plt.show()
    plt.figure()
    df['len'] = df['text'].apply(len)
    df['len'].plot(kind = 'hist', bins = 60)
    plt.show()
    plt.figure()
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['word_count'].plot(kind='hist', bins=60, edgecolor='black')
    plt.show()

    # Preprocessing: Map labels to binary values
    # If this had string sentiment given we would have done this
    # df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=1)

    # Feature Extraction using TF-IDF
    # TF-IDF gives more weight to words that are important to a document.
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Training the model
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    print("\nSentiment analysis model trained!")

    # Predictions and Evaluation
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Test with new reviews
    print("\n--- Testing with new reviews ---")
    new_reviews = [
        "The story was incredible and the visuals were stunning.",
        "It was a very dull and slow-paced film.",
        "A decent movie, not great but not bad either."
    ]
    new_reviews_tfidf = tfidf_vectorizer.transform(new_reviews)
    predictions = model.predict(new_reviews_tfidf)

    for review, prediction in zip(new_reviews, predictions):
        print(f'Review: "{review}" -> Sentiment: {"Positive" if prediction == 1 else "Negative"}')


sentiment_analysis()
