import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')

from nltk.stem.porter import PorterStemmer
import  sklearn

import base64
import joblib

ps = PorterStemmer()


def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
 
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "lav1.png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
side_bg = 'lav3.jpeg'
set_bg_hack(side_bg)


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))



st.title(":green[Email/SMS Spam Classifier]")
# st.markdown('''
#     :red[Email/SMS Spam Classifier]''')

input_sms = st.text_area(":violet[Enter the message]")

if st.button(':black[Predict]'):
    if not input_sms:
        st.header(":red[Please enter a message]")
    else:

    # 1. preprocess
        transformed_sms = transform_text(input_sms)
    # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
    # 3. predict
        result = model.predict(vector_input)[0]
        # st.header(result)
    # 4. Display
        if result == 1:
            st.header(":red[Spam]")
        else:
            st.header(":green[Not Spam]")