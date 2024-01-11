import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# from pandas_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report

# Import model
knn = pickle.load(open('KNN_model.pkl', 'rb'))

# Load dataset
data = pd.read_csv('Heart Dataset.csv')

st.title('Aplikasi prediksi penyakit Jantung')

html_layout1 = """
<br>
    <div style="background-color: purple ; padding:2px">
        <h2 style="color:white;text-align:center;font-size:35px"><b>Check Here</b></h2>
    </div>
<br>
<br>
"""
st.markdown(html_layout1, unsafe_allow_html=True)

activities = ['KNN']
option = st.sidebar.selectbox('Pilihan mu ?', activities)
st.sidebar.header('Data Pasien')

if st.checkbox("About Dataset"):
    html_layout2 ="""
    <br>
    <p>Ini adalah dataset Kaggle</p>
    """
    st.markdown(html_layout2, unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

# if st.checkbox('EDa'):
#     pr = ProfileReport(data, explorative=True)
#     st.header('**Input Dataframe**')
#     st.write(data)
#     st.write('---')
#     st.header('**Profiling Report**')
#     st_profile_report(pr)

# Train test split
X_new = data[['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']]
y_new = data['output']
X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.20, random_state=42)
knn.fit(X_train, y_train)
# SIMPAN MODEL BARU
pickle.dump(knn, open('KNN_update.pkl', 'wb'))

# Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():
    age = st.sidebar.slider('Umur', 1, 100, 20)
    sex = st.sidebar.slider('Jenis Kelamin', 0, 1, 1)
    cp = st.sidebar.slider('Nyeri Dada', 0, 3, 0)
    trtbps = st.sidebar.slider('Tekanan Darah', 0, 500, 120)
    chol = st.sidebar.slider('Kolestrol', 0, 500, 200)
    fbs = st.sidebar.slider('Gula Darah', 0, 1, 0)
    restecg = st.sidebar.slider('Resting Electrocardiographic', 0, 1, 0)
    thalachh = st.sidebar.slider('Denyut Jantung Maksimal', 0, 500, 90)
    exng = st.sidebar.slider('Angina', 0, 1, 0)
    oldpeak = st.sidebar.slider('Oldpeak', 0, 10, 0)
    slp = st.sidebar.slider('Kemiringan segmen ST latihan', 0, 2, 0)
    caa = st.sidebar.slider('caa', 0, 2, 0)
    thall = st.sidebar.slider('thall', 0, 3, 0)
    
    user_report_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trtbps': trtbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalachh': thalachh,
        'exng': exng,
        'oldpeak': oldpeak,
        'slp': slp,
        'caa': caa,
        'thall': thall
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# Data Pasien
user_data = user_report()
st.subheader('Data Pasien')
st.write(user_data)

user_result = knn.predict(user_data)
knn_score = accuracy_score(y_test, knn.predict(X_test))

# Output
st.subheader('Hasilnya adalah : ')
output = ''
if user_result[0] == 0:
    output = 'Kamu Sehat'
else:
    output = 'Kamu terkena Penyakit Jantung'
st.title(output)
st.subheader('Model yang digunakan : \n' + option)
st.subheader('Accuracy : ')
st.write(str(knn_score * 100) + '%')