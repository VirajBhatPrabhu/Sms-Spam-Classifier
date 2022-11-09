import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import VotingClassifier


ps = PorterStemmer()

tf = pickle.load(open('TFIDF.pkl','rb'))
model = pickle.load(open('mnbmodel.pkl','rb'))


def Preprocess(text):
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


st.markdown("<h1 style='text-align: center; '>SMS & Email Spam Classifier</h1>", unsafe_allow_html=True)

input_sms = st.text_area("Enter Your Text Here")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = Preprocess(input_sms)
    # 2. vectorize
    vector_input = tf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.markdown(f"<h2 style='text-align: center;'>This is a Spam Text</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 style='text-align: center;'>This is Not a Spam Text</h2>", unsafe_allow_html=True)