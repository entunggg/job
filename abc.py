import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import re

# Step 1: Load Job Data
file_url = "https://raw.githubusercontent.com/entunggg/job/f53f0535bc618fe1ded19eab93d3b9ab4f50d386/Job_List.csv"
job_data = pd.read_csv(file_url, encoding='utf-8')

# Step 2: Preprocess Data (文字處理)
def preprocess_text(text):
    """基本的文字預處理：小寫化並去除標點符號與多餘的空格"""
    text = re.sub(r'[^\w\s]', '', str(text).lower())  # 移除標點符號
    text = re.sub(r'\s+', ' ', text)  # 移除多餘空格
    return text.strip()  # 去掉開頭和結尾的空格

# 對技能需求欄位進行預處理
job_data['技能需求'] = job_data['技能需求'].apply(preprocess_text)

# Step 3: Keyword Extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=20)
tfidf_matrix = vectorizer.fit_transform(job_data['技能需求'])
keywords = list(vectorizer.get_feature_names_out())

# Step 4: Maintain Keyword Order with Session State
if "shuffled_keywords" not in st.session_state:
    st.session_state.shuffled_keywords = random.sample(keywords, len(keywords))

shuffled_keywords = st.session_state.shuffled_keywords

# Step 5: Initialize Score in session_state
if "score" not in st.session_state:
    st.session_state.score = 0

# Step 6: App UI
st.title("職缺推薦系統")
st.write("請點擊您感興趣的關鍵字！")

# Create a dictionary in session_state to track selected keywords
if "selected_keywords" not in st.session_state:
    st.session_state.selected_keywords = []

def toggle_keyword(keyword):
    """Toggle the selection state of a keyword."""
    if keyword in st.session_state.selected_keywords:
        st.session_state.selected_keywords.remove(keyword)
    else:
        st.session_state.selected_keywords.append(keyword)
        # Increase the score when a keyword is selected
        st.session_state.score += 1

# Display keywords as buttons
cols = st.columns(5)  # Adjust the number of columns to control layout
for idx, keyword in enumerate(shuffled_keywords):
    with cols[idx % 5]:
        if st.button(keyword):
            toggle_keyword(keyword)

# Show selected keywords
st.write("已選擇的關鍵字：", ", ".join(st.session_state.selected_keywords))
st.write(f"當前積分：{st.session_state.score}")

# Step 7: Recommend Jobs based on score
if st.button("推薦職缺"):
    if not st.session_state.selected_keywords:
        st.warning("請至少選擇一個關鍵字！")
    else:
        # Recommend jobs based on the user's score
        num_jobs_to_recommend = 1 if st.session_state.score < 5 else 3

        # Calculate Similarity
        user_vector = vectorizer.transform([' '.join(st.session_state.selected_keywords)])
        similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

        # Recommend Jobs
        job_data['相似度'] = similarity_scores
        recommended_jobs = job_data.sort_values(by='相似度', ascending=False).head(num_jobs_to_recommend)

        # Display Results
        st.subheader(f"推薦結果 ({num_jobs_to_recommend} 個職缺)")
        for _, row in recommended_jobs.iterrows():
            st.write(f"**{row['職位名稱']}** - {row['公司名稱']} ({row['地點']})")

