import streamlit as st
import pickle
import gensim
import base64
from gensim.utils import simple_preprocess
from utils import conv_lower,remove_special,remove_stopwords,join_back,document_vector

st.set_page_config(
        page_title="Sentiment Analysis on movie reviews",
)

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background("./bg_image.avif")


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


