
"""
@author: Gokulraj
"""
import numpy as np
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
import pandas as pd
import webbrowser
import streamlit as st 

from PIL import Image
MODEL_PATH ='stock_pre_lstm.h5'
model = load_model(MODEL_PATH)
voc_size=16000
sent_length = 200


def predict_message(message):
    text = message
    onehot_rep = [one_hot(text,voc_size)]
    embedded_doc = pad_sequences(onehot_rep, padding='pre', maxlen=sent_length)
    test = np.array(embedded_doc)
    prediction=model.predict(test)
    if prediction < 0.5:
        prediction = 0
    else:
        prediction = 1
    if prediction == 0:
        ans = "Stock price stayed same or decreased"
        
    else:
        ans = "Stock price increased."
    return ans



def main():
    st.text("@author: Gokulraj.T")
    st.text("Machine Learning & Deep learning App Built with Streamlit")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Stock Sentiment Analysis using News Headlines</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    message = st.text_input("Enter the news headline")
    result=""
    if st.button("Predict"):
        result=predict_message(message)
        
    
    st.success(result)
    if st.button("About"):
        st.text("This model can predict stock price increase or decrease based on the News headlines")
        st.text("Used Algorithm : LSTM(Long short term memory")
        st.text("Accuracy : 87%")
        link = '[Code](https://github.com/gokulvm/NLP_deep_learning_stock_sentiment_analysis)'
        st.markdown(link, unsafe_allow_html=True)
       
if __name__=='__main__':
    main()
    
       
