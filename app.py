import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "product_recommendation_dataser.csv"

df = pd.read_csv(DATA_PATH)

df["combined_features"] = (
    df["Category"] + " " +
    df["Brand"] + " " +
    df["Description"]
)

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df["combined_features"])

similarity_matrix = cosine_similarity(tfidf_matrix)

def recommend_products(product_name, num_recommendations=5):
    product_index = df[df["Product_Name"] == product_name].index[0]

    similarity_scores= similarity_matrix[product_index]

    similar_products= sorted(
        enumerate(similarity_scores),
        key=lambda x :x[1],
        reverse=True
    )

    recommendations=[]

    for index, score in similar_products[1:num_recommendations + 1]:
        recommendations.append(df.iloc[index]["Product_Name"])

    return recommendations

st.title("customer recommendation product system")

product_name= st.selectbox(
    "select a product",
    df["Product_Name"].tolist()
)

if st.button("Recommend Products"):
    recommendations= recommend_products(product_name)

    st.write("Recommend Products:")
    
    for product in recommendations:
        st.write(product)