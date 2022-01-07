import streamlit as st
import bz2
import pickle
import _pickle as cPickle
import pickle
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import re
import time
import warnings
import nltk
#import seaborn as sns
from sklearn.preprocessing import normalize
import joblib
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import SGDClassifier
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
warnings.filterwarnings("ignore")
#nltk.download('stopwords')

st.write("""
# Personalized-Cancer-Diagnosis Prediction App
This app predicts the **Personalized-Cancer-Diagnosis** 

Data obtained from the [Kaggle](https://www.kaggle.com/c/msk-redefining-cancer-treatment/data)
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://github.com/Bharathk075/Personalized-Cancer-Diagnosis/blob/main/test_data.zip)
""")


#@st.cache
def load_pickle_file(filename):
    pickle_obj = joblib.load(f"pickle_files/{filename}")
    return pickle_obj

#@st.cache
# Load any compressed pickle file
def decompress_pickle(file):
 data = bz2.BZ2File(file, 'rb')
 data = cPickle.load(data)
 return data

@st.cache
def nlp_preprocessing(total_text,stop_words): #, index, column)
    if type(total_text) is not int:
        string = ""
        # replace every special char with space
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', total_text)
        # replace multiple spaces with single space
        total_text = re.sub('\s+',' ', total_text)
        # converting all the chars into lower-case.
        total_text = total_text.lower()
        
        for word in total_text.split():
        # if the word is a not a stop word then retain that word from the data
            if not word in stop_words:
                string += word + " "
    return string

@st.cache
def clean_data(mytest_data,stop_words):
    #text processing stage.
    for index, row in mytest_data.iterrows():
        if type(row['TEXT']) is str:
            clean_string = nlp_preprocessing(row['TEXT'],stop_words) #, index, 'TEXT'
            mytest_data['TEXT'][index] = clean_string
    return mytest_data

def predict_data(model,vectorizer_Gene,vectorizer_Variation,vectorizer_TEXT,mytest_data):
    test_input =  hstack((vectorizer_Gene.transform(mytest_data['Gene']), vectorizer_Variation.transform(mytest_data['Variation']), vectorizer_TEXT.transform(mytest_data['TEXT']))).tocsr()
    pred = model.predict(test_input)
    prob = model.predict_proba(test_input)
    probabilities = []
    for i in prob[0]:
        probabilities.append("%.3f" % i)
        #print("%.3f" % i)
    return pred, probabilities


def load_test_data(sample):
    t1 = pd.read_csv("test_data.zip")
    test = t1.sample(sample)
    #test = t1[t1['ID'].isin([3316,3317,3318,3319,3320])]
    return test


def user_input_features(sample):
    sample_size = st.sidebar.slider('Test Sample', sample,5,1)
    mytest_data = load_test_data(sample)

        # island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        # sex = st.sidebar.selectbox('Sex',('male','female'))
        # bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
        # bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
        # flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
        # body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
        # data = {'island': island,
        #         'bill_length_mm': bill_length_mm,
        #         'bill_depth_mm': bill_depth_mm,
        #         'flipper_length_mm': flipper_length_mm,
        #         'body_mass_g': body_mass_g,
        #         'sex': sex}
        # features = pd.DataFrame(data, index=[0])
    return mytest_data

#mytest_data = user_input_features(sample)

def main():
    sample_size = st.sidebar.slider('Test Sample', 1,5,1)
    mytest_data = load_test_data(sample_size)
    st.write(mytest_data)
    st.write("A sample data selected from test data file")
    if st.button("predict",key="data"):
        #st.write(mytest_data)
        clf = load_pickle_file("model.pkl")   #clf = joblib.load("pickle_files/")
        vectorizer_Gene = load_pickle_file("vectorizer_Gene.pkl") #joblib.load('pickle_files/vectorizer_Gene.pkl') #decompress_pickle('pickle_files/vectorizer_Gene.pkl.pbz2') #
        vectorizer_Variation = load_pickle_file("vectorizer_Variation.pkl") #joblib.load('pickle_files/vectorizer_Variation.pkl') #decompress_pickle('pickle_files/vectorizer_Variation.pkl.pbz2') #
        vectorizer_TEXT = decompress_pickle('pickle_files/vectorizer_TEXT.pkl.pbz2') ##vectorizer_TEXT =  joblib.load('pickle_files/vectorizer_TEXT.zip') #load_pickle_file("vectorizer_TEXT.pkl") #
        stop_words = set(stopwords.words('english'))
        mytest_data2 = clean_data(mytest_data,stop_words)
        #st.write(mytest_data)
        #st.write("data after cleansing")
        pred, probabilities = predict_data(clf,vectorizer_Gene,vectorizer_Variation,vectorizer_TEXT,mytest_data2)
        st.write(pred)
        st.write(probabilities)





    

if __name__ == "__main__":
    main()
    #st.write(os.listdir("pickle_files"))
    # clf = load_pickle_file("model.pkl")   #clf = joblib.load("pickle_files/")
    # vectorizer_Gene = decompress_pickle('pickle_files/vectorizer_Gene.pkl.pbz2') #load_pickle_file("vectorizer_Gene.pkl") #joblib.load('pickle_files/vectorizer_Gene.pkl')
    # vectorizer_Variation = decompress_pickle('pickle_files/vectorizer_Variation.pkl.pbz2') #load_pickle_file("vectorizer_Variation.pkl") #joblib.load('pickle_files/vectorizer_Variation.pkl')
    # vectorizer_TEXT = decompress_pickle('pickle_files/vectorizer_TEXT.pkl.pbz2') ##vectorizer_TEXT =  joblib.load('pickle_files/vectorizer_TEXT.zip') #load_pickle_file("vectorizer_TEXT.pkl") #
    # stop_words = set(stopwords.words('english'))
    # mytest_data = user_input_features(sample)
    # df = load_test_data()
    # mytest_data = df.sample(1)
    # clean_data(mytest_data,stop_words)
    #st.write(df)
    # st.write(mytest_data)
    # pred, probabilities = predict_data(clf,vectorizer_Gene,vectorizer_Variation,vectorizer_TEXT,mytest_data)
    # st.write(pred)
    # st.write(probabilities)
