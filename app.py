import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
from PIL import Image

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
st.write('Cek Dataset secara keseluruhan')
st.dataframe(data)
st.write("### .head() dataset")
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

st.write('sudah tidak ada outlier untuk data olahan ini')    

st.write('## Visualisasi')

correlation_matrix = data.corr()
# Buat heatmap korelasi menggunakan seaborn
plt.figure(figsize=(10, 6))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)

# Tampilkan heatmap di Streamlit
st.write("Correlation Heatmap Dataset:")
st.pyplot(plt)

st.subheader("Harga Rumah vs Luas Tanah")
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data['luas_tanah'], data['harga_juta'], alpha=0.7, color='blue')
ax.set_xlabel('Luas Tanah (m²)')
ax.set_ylabel('Harga Rumah (Juta)')
ax.set_title('Harga Rumah vs Luas Tanah')
st.pyplot(fig)

st.subheader("Harga Rumah vs Luas Bangunan")
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data['luas_bangunan'], data['harga_juta'], alpha=0.7, color='green')
ax.set_xlabel('Luas Bangunan (m²)')
ax.set_ylabel('Harga Rumah (Juta)')
ax.set_title('Harga Rumah vs Luas Bangunan')
st.pyplot(fig)

# Combined scatter plot for both Luas Tanah and Luas Bangunan vs Harga Rumah
plt.figure(figsize=(10, 6))
plt.scatter(data['luas_tanah'], data['harga_juta'], color='blue', alpha=0.5, label='Land Area', marker='x')
plt.scatter(data['luas_bangunan'], data['harga_juta'], color='red', alpha=0.5, label='Building Area', marker='x')
plt.title('Harga Rumah vs. Luas Tanah dan Luas Bangunan')
plt.xlabel('Luas (m²)')
plt.ylabel('Harga (Juta)')
plt.legend()
plt.grid(True)
st.pyplot(plt)

# Bar Plot for Bed vs Harga Rumah
plt.figure(figsize=(8, 6))
sns.barplot(x='bed', y='harga_juta', data=data, ci=None, palette='Blues_d')
plt.title('Rata - Rata Harga rumah vs Jumlah Kamar')
plt.xlabel('Jumlah Kamar')
plt.ylabel('Harga (Jutaan)')
st.pyplot(plt)

plt.figure(figsize=(8, 6))
sns.barplot(x='bath', y='harga_juta', data=data, ci=None, palette='Reds_d')
plt.title('Rata - Rata Harga rumah vs Jumlah Kamar Mandi')
plt.xlabel('Jumlah Kamar Mandi')
plt.ylabel('Harga (Jutaan)')
st.pyplot(plt)

plt.figure(figsize=(8, 6))
sns.barplot(x='carport', y='harga_juta', data=data, ci=None, palette='Greens_d')
plt.title('Rata - Rata Harga rumah vs Jumlah Carport')
plt.xlabel('Jumlah Carport')
plt.ylabel('Harga (Jutaan)')
st.pyplot(plt)

# Load the image
image = Image.open('Tensorflow_actual_vs_predicted_prices.png')

# Display the image in Streamlit with a caption
st.image(image, caption='Hasil Asli vs Prediksi Harga Linear Regression pake Tensorflow', use_column_width=True)

image = Image.open('randomforest_actual_vs_predicted_prices.png')
st.image(image, caption='Hasil Asli vs Prediksi Harga Linear Regressionpakai Scikit-Learn Random Forest', use_column_width=True)