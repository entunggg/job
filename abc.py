import streamlit as st
import pandas as pd
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

# Step 5: Initialize Score and Operation Log in session_state
if "score" not in st.session_state:
    # Try loading the score from a file
    if os.path.exists("score.pkl"):
        with open("score.pkl", "rb") as f:
            st.session_state.score = pickle.load(f)
    else:
        st.session_state.score = 0  # Default to 0 if no previous score

if "operation_log" not in st.session_state:
    st.session_state.operation_log = []

# Step 6: App UI
st.markdown("<h1 style='text-align: center;'>職缺推薦系統</h1>", unsafe_allow_html=True)
st.write("請點擊您感興趣的關鍵字！")

# Create a dictionary in session_state to track selected keywords
if "selected_keywords" not in st.session_state:
    st.session_state.selected_keywords = []

def toggle_keyword(keyword):
    """Toggle the selection state of a keyword and update score."""
    if keyword in st.session_state.selected_keywords:
        st.session_state.selected_keywords.remove(keyword)
        st.session_state.score -= 1  # Decrease score if keyword is deselected
        operation = f"取消選擇關鍵字: {keyword}"
    else:
        st.session_state.selected_keywords.append(keyword)
        st.session_state.score += 1  # Increase score if keyword is selected
        operation = f"選擇關鍵字: {keyword}"
    
    # Log the operation
    st.session_state.operation_log.append(operation)
    with open("score.pkl", "wb") as f:
        pickle.dump(st.session_state.score, f)  # Save score in the file

# Display keywords as buttons
cols = st.columns(5)  # Adjust the number of columns to control layout
for idx, keyword in enumerate(shuffled_keywords):
    with cols[idx % 5]:
        if st.button(keyword):
            toggle_keyword(keyword)

# Show selected keywords
st.write("已選擇的關鍵字：", ", ".join(st.session_state.selected_keywords))
st.write(f"當前積分：{st.session_state.score}")

# Step 9: User Feedback Form
feedback_file = r"feedback.txt"
# 顯示用戶回饋輸入區域
st.markdown("<h2 style='text-align: center;'>用戶回饋</h2>", unsafe_allow_html=True)

feedback_text = st.text_area("請輸入您的建議或回饋：")

# 提交回饋按鈕
if st.button("提交回饋"):
    if feedback_text.strip():
        try:
            with open(feedback_file, "a", encoding="utf-8") as f:
                f.write(feedback_text + "\n")
            st.success("感謝您的回饋！")
        except Exception as e:
            st.error(f"處理回饋時發生錯誤: {str(e)}")
    else:
        st.warning("請輸入回饋內容後再提交！")

# 添加復選框選項來控制是否顯示評論
show_feedback = st.checkbox("顯示其他用戶回饋")

# 如果用戶選擇顯示評論，讀取並顯示評論
if show_feedback:
    try:
        if os.path.exists(feedback_file):
            with open(feedback_file, "r", encoding="utf-8") as f:
                feedback_list = f.readlines()

            st.markdown("<h3 style='text-align: center;'>其他用戶回饋</h3>", unsafe_allow_html=True)

            if feedback_list:
                for i, feedback in enumerate(feedback_list, 1):
                    st.write(f"{i}. {feedback.strip()}")
            else:
                st.info("目前沒有用戶回饋。")
        else:
            st.info("目前沒有回饋檔案。")
    except Exception as e:
        st.error(f"讀取回饋時發生錯誤: {str(e)}")

st.markdown("<h2 style='text-align: center;'>操作記錄</h2>", unsafe_allow_html=True)
if "operation_log" in st.session_state and st.session_state.operation_log:
    st.write("以下是您的操作記錄：")
    for log in st.session_state.operation_log:
        st.write(log)
else:
    st.write("目前無操作記錄。")
