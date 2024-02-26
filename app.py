import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import  sklearn

import base64
import joblib
nltk.download('punkt')

ps = PorterStemmer()

# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
 
# def set_bg_hack_url():
#     '''
#     A function to unpack an image from url and set as bg.
#     Returns
#     -------
#     The background.
#     '''
        
#     st.markdown(
#          f"""
#          <style>
#          .stApp {{
#              background: url("https://cdn.pixabay.com/photo/2020/06/19/22/33/wormhole-5319067_960_720.jpg");
#              background-size: cover
#          }}
#          </style>
#          """,
#          unsafe_allow_html=True
#      )
    
# set_bg_hack_url()


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



# # set background, use base64 to read local file
# def get_base64_of_bin_file(bin_file):
#     """
#     function to read png file 
#     ----------
#     bin_file: png -> the background image in local folder
#     """
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# def set_png_as_page_bg(png_file):
#     """
#     function to display png as bg
#     ----------
#     png_file: png -> the background image in local folder
#     """
#     bin_str = get_base64_of_bin_file(png_file)
#     page_bg_img = '''
#     <style>
#     st.App {
#     background-image: url("data:image/png;base64,%s");
#     background-size: cover;
#     }
#     </style>
#     ''' % bin_str
    
#     st.markdown(page_bg_img, unsafe_allow_html=True)
#     return

# set_png_as_page_bg(image)



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



st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if not input_sms:
        st.header("Please enter a message")
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
            st.header("Spam")
        else:
            st.header("Not Spam")
