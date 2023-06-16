import streamlit as st
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from time import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from pandas import DataFrame
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Load data
data = pd.read_csv('Data-hasil-stemming-revisi.csv')

# membuat slidebar
st.subheader("ANALISIS SENTIMEN PENILAIAN MASYARAKAT")
st.sidebar.title("Pengaturan")
method = st.sidebar.selectbox( "Metode Analisis", ["Naive Bayes"])
text_input = st.sidebar.text_input("Masukkan Teks untuk Analisis", "")

if st.sidebar.button("Analisis Sentimen"):
    if method == "Naive Bayes":
        
        # Split data
        X = data['tweet']
        y = data['Label']

        bow_transformer = CountVectorizer().fit(data['tweet'])
        tokens = bow_transformer.get_feature_names_out()
        text_bow = bow_transformer.transform(data['tweet'])
        X = text_bow.toarray()
        tfidf_transformer = TfidfTransformer().fit(text_bow)
        tweet_tfidf = tfidf_transformer.transform(text_bow)

        # Split the data 8:2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the Naive Bayes model
        model = MultinomialNB()
        model.fit(X_train, y_train)
        test_1_unseen=bow_transformer.transform([text_input])
        data=test_1_unseen.toarray()
        prediction = model.predict(data)
        
        # evaluasi hitung
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        score1 = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, average='weighted')
        error_rate = 1 - score1


        # Display sentiment analysis results
        st.write("Teks Yang diinputkan :", text_input)
        st.write("Prediksi Sentimen Untuk Kata Yang di inputkan Adalah :", prediction[0])

        st.subheader("Confusion Matrix")
        st.write(pd.DataFrame(cm, columns=['Negatif', 'Positif'], index=['Negatif', 'Positif']))
        # Display classification report
        st.subheader("Classification Report")
        st.write("Akurasi:", score1)
        st.write("Error Rate:", error_rate)
        st.text(report)
