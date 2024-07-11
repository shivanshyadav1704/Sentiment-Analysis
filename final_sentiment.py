import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import streamlit as st
import seaborn as sns
import re
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from deep_translator import GoogleTranslator


#Rmoving warnings from stramlit UI
st.set_option('deprecation.showPyplotGlobalUse', False)



st.write("""
# Sentiment Analysis on Twitter Tweets
         

""")

st.write("-----------------------------------------------------------------------------------------------------------")

# File Input
uploaded_file = st.file_uploader('Upload your file here \n',type=["csv"])

st.subheader("OR")

# Text input
tweet_txt = st.text_input("Enter your tweet:")

if uploaded_file is not None:
    st.write("#### Provided Dataset :")
    # Load the data into a DataFrame
    df = pd.read_csv(uploaded_file)
    st.write(df.head(10))

    st.write("-----------------------------------------------------------------------------------------------------------")


# Preprocessing of the dataset
    
    def data_processing(text):
        text = text.lower()
        text = re.sub(r"https\S+|www\S+https\S+", '',text, flags=re.MULTILINE)
        text = re.sub(r'\@w+|\#','',text)
        text = re.sub(r'[^\w\s]','',text)
        text_tokens = word_tokenize(text)
        filtered_text = [w for w in text_tokens if not w in stop_words]
        return " ".join(filtered_text)

    new_df = pd.DataFrame()
    new_df["Tweets"] = df["Tweets"].apply(data_processing)
    new_df = new_df.drop_duplicates("Tweets")

    stemmer = PorterStemmer()
    def stemming(data):
        text = [stemmer.stem(word) for word in data]
        return data
    
    new_df["Tweets"] = new_df["Tweets"].apply(lambda x: stemming(x))

    st.write('#### Data after converting it to lowercase, removing URLs, Twitter handles, hashtags, punctuation marks, and stop words, and tokenizing the text into individual words.')
    st.write(new_df.head(10))

    st.write("-----------------------------------------------------------------------------------------------------------")


    def polarity(text):
        return TextBlob(text).sentiment.polarity

    new_df["Polarity"] = new_df["Tweets"].apply(polarity)

    st.write('#### After applying polarity to the data')
    st.write(new_df.head(10))

    st.write("-----------------------------------------------------------------------------------------------------------")


    def sentiment(label):
        if label <0:
            return "Negative"
        elif label ==0:
            return "Neutral"
        elif label>0:
            return "Positive"
        
    new_df["Sentiment"] = new_df["Polarity"].apply(sentiment)


    st.write('#### Assigning Sentiment to the data according to the polarity')
    st.write(new_df.head(10))

    st.write("-----------------------------------------------------------------------------------------------------------")

    st.write("#### Count of Sentiment in the Data  ")
    fig = plt.figure(figsize=(8,6))
    # plt.title("Count of Sentiment in the Data ")
    # sns.set_palette("Set2")
    sns.countplot(x='Sentiment', data = new_df,palette=["red", "blue", "green"])

    st.pyplot(fig)


    st.write("-----------------------------------------------------------------------------------------------------------")
    
    st.write("#### Distribution of sentiments in the dataset")
    fig = plt.figure(figsize=(8,8))
    colors = ("green", "blue", "red")
    wp = {'linewidth':2, 'edgecolor':"black"}
    tags = new_df['Sentiment'].value_counts()
    explode = (0.1,0.1,0.1)
    tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors = colors, startangle=90, wedgeprops = wp, explode = explode, label='')
    # plt.title('Distribution of sentiments')

    st.pyplot(fig)











elif tweet_txt:
    to_translate = tweet_txt
    translated = GoogleTranslator(source='auto', target='english').translate(to_translate)
    st.write(f"##### Your text was : {translated}")
    # preprocess tweet
    tweet_words = []
    for word in translated.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)
    tweet_proc = " ".join(tweet_words)
    
    # Load model and tokenizer
    roberta = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)
    labels = ['Negative', 'Neutral', 'Positive']

    # Sentiment analysis
    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
    output = model(**encoded_tweet)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # Display sentiment analysis results
    st.write("### Sentiment Analysis Results:")
    for i in range(len(scores)):
        st.write(f"- {labels[i]}: {scores[i]}")

    plt.figure(figsize=(8, 6))
    colors = ("red", "blue", "green")
    explode = (0.1, 0.1, 0.1)
    wp = {'linewidth': 1.5, 'edgecolor': "black"}  # Adjust border properties here

# Plot the pie chart
    plt.pie(scores, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors,
        explode=explode, wedgeprops=wp)
    
    plt.title("Sentiment Percentage")

    st.pyplot()


