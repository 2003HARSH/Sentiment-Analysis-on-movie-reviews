import streamlit as st
import pickle
import gensim
from gensim.utils import simple_preprocess
from utils import conv_lower,remove_special,remove_stopwords,join_back,document_vector

st.set_page_config(
        page_title="Sentiment Analysis on movie reviews",
)

clf = pickle.load(open('clf.pkl', 'rb'))
model = gensim.models.Word2Vec.load("word2vec.model")

st.title('Sentiment Analysis on movie reviews')

input_text=st.text_input("Enter your review")

if st.button('Predict'):
    text=conv_lower(input_text)
    text=remove_special(text)
    text=remove_stopwords(text)
    text=join_back(text)
    text=simple_preprocess(text)
    text=join_back(text)

    X=document_vector(text,model)
    output=clf.predict(X.reshape(1,100))[0]
    if output==1:
        st.header('Postitve review')
    else:
        st.header('Negative review')


