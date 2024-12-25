import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import re

# Step 1: File Upload - User Uploads the Job Data CSV
uploaded_file = st.file_uploader("請上傳職缺資料 CSV 檔案", type=["csv"])

if uploaded_file is not None:
    # Load job data
    job_data = pd.read_csv(uploaded_file)

    # Step 2: Preprocess Data
    def preprocess_text(text):
        """Basic text preprocessing: lowercasing and removing punctuation."""
        return re.sub(r'[^\w\s]', '', str(text).lower())

    job_data['技能需求'] = job_data['技能需求'].apply(preprocess_text)

    # Step 3: Keyword Extraction using TF-IDF
    vectorizer = TfidfVectorizer(max_features=20)
    tfidf_matrix = vectorizer.fit_transform(job_data['技能需求'])
    keywords = list(vectorizer.get_feature_names_out())

    # Step 4: Maintain Keyword Order with Session State
    if "shuffled_keywords" not in st.session_state:
        st.session_state.shuffled_keywords = random.sample(keywords, len(keywords))

    shuffled_keywords = st.session_state.shuffled_keywords

    # Step 5: App UI
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

    # Display keywords as buttons dynamically in columns
    num_columns = 5  # Number of columns to display buttons
    cols = st.columns(num_columns)
    for idx, keyword in enumerate(shuffled_keywords):
        with cols[idx % num_columns]:
            if st.button(keyword):
                toggle_keyword(keyword)

    # Show selected keywords
    st.write("已選擇的關鍵字：", ", ".join(st.session_state.selected_keywords))

    # Step 6: Recommend Jobs
    if st.button("推薦職缺"):
        if not st.session_state.selected_keywords:
            st.warning("請至少選擇一個關鍵字！")
        else:
            # Calculate Similarity
            user_vector = vectorizer.transform([' '.join(st.session_state.selected_keywords)])
            similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

            # Recommend Jobs
            job_data['相似度'] = similarity_scores
            recommended_jobs = job_data.sort_values(by='相似度', ascending=False).head(5)

            # Display Results
            st.subheader("推薦結果")
            for _, row in recommended_jobs.iterrows():
                st.write(f"**職位名稱**: {row['職位名稱']}")
                st.write(f"**公司名稱**: {row['公司名稱']}")
                st.write(f"**地點**: {row['地點']}")
                st.write(f"**相似度分數**: {row['相似度']:.2f}")
                st.write("---")

else:
    st.warning("請上傳職缺資料 CSV 檔案！")
