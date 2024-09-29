import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

st.title("LSP Data Science: CRISP-DM Approach Portfolio")
st.write("### Nama: Muhamad Ibnu Khaidar Hafiz")
st.write("### Kelas: 4IA15")
st.write("### NPM: 50421867")

st.header("Latihan Portofolio menggunakan Pendekatan CRISP-DM")
st.markdown("""
### CRISP-DM Steps:
1. Data Understanding
2. Data Preparation
3. Modeling
4. Evaluation
5. Deployment
""")

st.write('Data asli atau dataset yang pakai di kaggle ini adalah dari Kaggle ini: [Link](https://www.kaggle.com/datasets/pramudyadika/yogyakarta-housing-price-ndonesia)')
st.write('Dataset yang saya tampilkan disini sudah bisa dilihat proses pembuatannya di [github saya](https://github.com/Cleign1/Porto_data-science)')

# Load Dataset directly from 'olahan.csv'
st.header("Data Understanding")
data = pd.read_csv('olahan.csv')
st.write("### First 5 rows of the dataset")
st.write(data.head())

st.write("### Cek shape dari dataset")
st.write(data.shape)

st.write("### Cek tipe data dari dataset")
st.write(data.dtypes)

st.write('### Statistika Dasar')
# Display df.info() output
buffer = io.StringIO()
data.info(buf=buffer)
info_str = buffer.getvalue()
st.subheader("Informasi Dataset")
st.text(info_str)

st.header('Data Preparation')

st.write('### Cek missing Value')
st.text(data.isna().any())
st.write('### Cek missing value lebih jelas')
st.text(data.isna().sum())
st.write('sudah tidak ada lagi missing data karena sudah di olah data tersebut.')

st.write('### Cek outlier')
df_outlier = data.select_dtypes(exclude=['object'])
for column in df_outlier:
    st.subheader(f'Boxplot for {column}')
    
    # Create a figure for each boxplot
    fig, ax = plt.subplots(figsize=(20, 2))
    sns.boxplot(data=df_outlier, x=column, ax=ax)
    
    # Display the boxplot in Streamlit
    st.pyplot(fig)