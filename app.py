# -*- coding: utf-8 -*-

import streamlit as st
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences


## loading saved model
from tensorflow.keras.models import load_model
loaded_model = load_model('spam_model.h5')


ps = PorterStemmer()

def transform_text(messages):
    review = re.sub('[^a-zA-Z]', ' ', messages)  # removes all special characters other than a-z,A-Z
    review = review.lower()  # all characters sat to lowercase

    # word_tokenization method
    review = review.split()  # sentences to words to apply stopwords

    # stemming is applied for words that which words are not in stop words
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]

    # then again joined into sentences
    review = ' '.join(review)

    # onehot representing
    voc_size = 5000
    one_hot_ex = [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0, 0, 0,  0, 0]
    onehot_repr = one_hot(review, voc_size)
    
    list = []
    list.append(one_hot_ex)
    list.append(onehot_repr)
    
    
    
    sent_length=40
    embedded_docs=pad_sequences(list,padding='pre',maxlen=sent_length)
    
    return embedded_docs

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")


if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. predict
    result = loaded_model.predict(transformed_sms)[1]
    result = np.round(result).astype(int)
    
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")