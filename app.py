import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import streamlit as st
import pickle

ps=PorterStemmer()
def clear(txt):
    txt = txt.lower()
    txt = nltk.word_tokenize(txt)

    y = []

    for i in txt:
        if i.isalnum():
            y.append(i)

    txt = y[:]
    y.clear()

    for i in txt:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    txt = y[:]
    y.clear()

    for i in txt:
        y.append(ps.stem(i))

    return " ".join(y)


model=pickle.load(open('model.pkl','rb'))
vectoriser=pickle.load(open('vectorise.pkl','rb'))

st.title('Email/SMS Spam Classifier ')
input_txt=st.text_input("Enter the message")


if st.button('Predict'):
    input_transform=clear(input_txt)
    input=vectoriser.transform([input_transform])

    pred=model.predict(input)
    print(pred)
    if pred == 1:
        st.header("Spam")

    else:
        st.header("Not Spam")

