import streamlit as st
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Initialize the Sentiment Intensity Analyzer
analyzer = SentimentIntensityAnalyzer()

# Load the pre-trained LDA model
lda_model = joblib.load('lda_model.pkl')

# Load the vectorizer used during training
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = joblib.load(file)

# Get the vocabulary indices for 'israel' and 'palestine' in your vectorizer
israel_word_index = vectorizer.vocabulary_.get('israel')
palestine_word_index = vectorizer.vocabulary_.get('palestine')

if israel_word_index is not None and palestine_word_index is not None:
    topic_term_distribution = lda_model.components_
    israel_topic_index = topic_term_distribution[:, israel_word_index].argmax()
    palestine_topic_index = topic_term_distribution[:, palestine_word_index].argmax()
else:
    st.write("Words 'israel' and 'palestine' not found in the vocabulary.")
    st.stop()

# Function to clean text
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove HTML tags
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    
    # Remove special characters and lowercase the text
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Lemmatization and removing stopwords
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    # Join the tokens back into a sentence
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

# Function to determine geopolitical stance
def determine_geopolitical_stance(comment):
    cleaned_comment = clean_text(comment)
    sentiment_score = analyzer.polarity_scores(cleaned_comment)["compound"]
    topic_distribution = lda_model.transform(vectorizer.transform([cleaned_comment]))[0]
    
    # Determine stance based on sentiment score and dominant topic
    if sentiment_score <= -0.5:
        return 'Against Israel/Palestine'
    elif sentiment_score >= 0.5 and topic_distribution.argmax() == israel_topic_index:
        return 'Supports Israel'
    elif sentiment_score >= 0.5 and topic_distribution.argmax() == palestine_topic_index:
        return 'Supports Palestine'
    else:
        return 'Neutral/Stance Not Clear'

# Streamlit app
st.title('Geopolitical Stance Detector')

# User input
comment = st.text_area('Enter your comment:')
if st.button('Determine Stance'):
    if comment:
        stance = determine_geopolitical_stance(comment)
        st.write('Geopolitical Stance:', stance)
    else:
        st.write('Please enter a comment to determine its geopolitical stance.')
