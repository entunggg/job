import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import re

# Step 1: Load Job Data
file_url = "https://raw.githubusercontent.com/entunggg/job/f53f0535bc618fe1ded19eab93d3b9ab4f50d386/Job_List.csv"  # CSV 檔案的原始網址
job_data = pd.read_csv(file_url, encoding='utf-8')  # 讀取 CSV 檔案，確保使用 UTF-8 編碼

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

# Initialize or get the user's score from session state
if "user_score" not in st.session_state:
    st.session_state.user_score = 0

# Initialize or get the user's selected keywords from session state
if "selected_keywords" not in st.session_state:
    st.session_state.selected_keywords = []

# Step 5: App UI
st.title("職缺推薦系統")
st.write("請點擊您感興趣的關鍵字！")

def toggle_keyword(keyword):
    """Toggle the selection state of a keyword and add score."""
    if keyword in st.session_state.selected_keywords:
        st.session_state.selected_keywords.remove(keyword)
        st.session_state.user_score -= 1  # Deduct points when unselecting
    else:
        st.session_state.selected_keywords.append(keyword)
        st.session_state.user_score += 1  # Add points when selecting

# Display keywords as buttons
cols = st.columns(5)  # Adjust the number of columns to control layout
for idx, keyword in enumerate(shuffled_keywords):
    with cols[idx % 5]:
        if st.button(keyword):
            toggle_keyword(keyword)

# Show selected keywords
st.write("已選擇的關鍵字：", ", ".join(st.session_state.selected_keywords))

# Show the user's current score
st.write(f"您的當前積分：{st.session_state.user_score}")

# Step 6: Recommend Jobs
if st.button("推薦職缺"):
    if st.session_state.user_score < 5:
        # Recommend one job initially
        # Calculate Similarity for the first recommendation (this could be based on a random or fixed initial recommendation)
        user_vector = vectorizer.transform([' '.join(st.session_state.selected_keywords)])
        similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

        # Recommend one job
        job_data['相似度'] = similarity_scores
        recommended_job = job_data.sort_values(by='相似度', ascending=False).head(1)

        # Display the recommended job
        st.subheader("推薦結果")
        for _, row in recommended_job.iterrows():
            st.write(f"**{row['職位名稱']}** - {row['公司名稱']} ({row['地點']})")

    elif st.session_state.user_score >= 5:
        # Recommend three jobs when score is >= 5
        user_vector = vectorizer.transform([' '.join(st.session_state.selected_keywords)])
        similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

        # Recommend three jobs
        job_data['相似度'] = similarity_scores
        recommended_jobs = job_data.sort_values(by='相似度', ascending=False).head(3)

        # Display the recommended jobs
        st.subheader("推薦結果")
        for _, row in recommended_jobs.iterrows():
            st.write(f"**{row['職位名稱']}** - {row['公司名稱']} ({row['地點']})")
        
        # Award bonus points for receiving multiple recommendations
        st.session_state.user_score += 5  # Bonus points for getting multiple recommendations
        st.write(f"獎勳：您獲得了 5 點積分，現在您的總積分是 {st.session_state.user_score}！")
