import pandas as pd
import re
import nltk
import string
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

data = pd.read_csv("Imdb_Movies.csv")

data = data[['Series_Title', 'Overview']]

def preprocess_text(text):
    text = text.lower()  
    text = re.sub(r'\d+', '', text)  
    text = text.translate(str.maketrans("", "", string.punctuation))  
    tokens = word_tokenize(text)  
    tokens = [word for word in tokens if word not in stopwords.words('english')]  
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  
    return " ".join(tokens)


data['Cleaned_Overview'] = data['Overview'].apply(preprocess_text)

def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "positive"
    elif analysis.sentiment.polarity < 0:
        return "negative"
    else:
        return "neutral"

data['Sentiment'] = data['Cleaned_Overview'].apply(get_sentiment)

vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(data['Cleaned_Overview']).toarray()
y = data['Sentiment'].map({"negative": 0, "neutral": 1, "positive": 2})  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, target_names=["Negative", "Neutral", "Positive"]))

data['Predicted_Sentiment'] = model.predict(vectorizer.transform(data['Cleaned_Overview']).toarray())

data.to_csv("sentiment_analysis_results.csv", index=False)
print("Sentiment analysis completed. Results saved to 'sentiment_analysis_results.csv'.")
