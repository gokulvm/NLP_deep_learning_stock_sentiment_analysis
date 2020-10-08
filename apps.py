
"""
@author: Gokulraj
"""
import numpy as np
import pickle
import pandas as pd
import webbrowser
import streamlit as st

from PIL import Image

model=pickle.load(open('stockanalys2.pkl', 'rb'))
transformer=pickle.load(open('stock_trans.pkl', 'rb'))



def predict_message(message):
    message = [message]
    message = transformer.transform(message).toarray()
    prediction = model.predict(message)
    if prediction == 0:
        ans = "Stock price stayed same or decreased"
        
    else:
        ans = "Stock price increased."
    return ans



def main():
    st.text("@author: Gokulraj.T")
    st.text("Machine Learning App Built with Streamlit")
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
        st.text("Used Algorithm : Multinomial Naive Bayes")
        st.text("Accuracy : 84%")
        link = '[Code](https://github.com/gokulvm/NLP_deep_learning_stock_sentiment_analysis)'
        st.markdown(link, unsafe_allow_html=True)
       
if __name__=='__main__':
    main()
    
       
