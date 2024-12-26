import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import re
import pickle
import os

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

# Step 5: Initialize Score in session_state (use session_state to persist score)
if "score" not in st.session_state:
    # Try loading the score from a file
    if os.path.exists("score.pkl"):
        with open("score.pkl", "rb") as f:
            st.session_state.score = pickle.load(f)
    else:
        st.session_state.score = 0  # Default to 0 if no previous score

# Step 6: App UI
st.markdown("<h1 style='text-align: center;'>職缺推薦系統</h1>", unsafe_allow_html=True)
st.write("")
st.write("請點擊您感興趣的關鍵字！")

# Create a dictionary in session_state to track selected keywords
if "selected_keywords" not in st.session_state:
    st.session_state.selected_keywords = []

def toggle_keyword(keyword):
    """Toggle the selection state of a keyword and update score."""
    if keyword in st.session_state.selected_keywords:
        st.session_state.selected_keywords.remove(keyword)
        st.session_state.score -= 1  # Decrease score if keyword is deselected
    else:
        st.session_state.selected_keywords.append(keyword)
        st.session_state.score += 1  # Increase score if keyword is selected

    # Save score to a file
    with open("score.pkl", "wb") as f:
        pickle.dump(st.session_state.score, f)

# Display keywords as buttons
cols = st.columns(5)  # Adjust the number of columns to control layout
for idx, keyword in enumerate(shuffled_keywords):
    with cols[idx % 5]:
        if st.button(keyword):
            toggle_keyword(keyword)

# Show selected keywords
st.write("已選擇的關鍵字：", ", ".join(st.session_state.selected_keywords))
st.write("")
st.write(f"當前積分：{st.session_state.score}")

# Step 7: Recommend Jobs based on score
if st.button("推薦職缺"):
    if not st.session_state.selected_keywords:
        st.warning("請至少選擇一個關鍵字！")
    else:
        # Enhanced reward mechanism based on score
        if st.session_state.score < 5:
            num_jobs_to_recommend = 1
        elif st.session_state.score < 10:
            num_jobs_to_recommend = 3
        else:
            num_jobs_to_recommend = 5

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
            st.write(f"關鍵技能需求：{row['技能需求']}")
            st.write("---")

# Step 8: Reset Button for Score and Selected Keywords
def reset_progress():
    """Reset the score and selected keywords."""
    st.session_state.score = 0
    st.session_state.selected_keywords = []
    with open("score.pkl", "wb") as f:
        pickle.dump(st.session_state.score, f)  # Reset score in the file

# Add a button to reset the score and keywords, styled in three lines
st.write("\n")  # Add empty lines for spacing
st.write("\n")
if st.button("重置"):
    reset_progress()
    st.success("積分與選擇已重置！請刷新網頁！")
st.write("\n")
