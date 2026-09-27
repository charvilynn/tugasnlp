import streamlit as st
import pandas as pd
import numpy as np
import re
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi Halaman Web
st.set_page_config(
    page_title="Medical Assistant Chatbot",
    page_icon="🩺",
    layout="centered"
)

# Unduh resource NLTK yang dibutuhkan
@st.cache_resource
def download_nltk_data():
    for res in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
        nltk.download(res, quiet=True)

download_nltk_data()

# Pipeline Preprocessing Text
lemmatizer = WordNetLemmatizer()
english_stopwords = set(stopwords.words("english"))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in english_stopwords and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# Load Dataset & Training TF-IDF Matrix (di-cache agar cepat dan ringan di memori)
@st.cache_resource
def init_chatbot(csv_path="train_data_chatbot1.csv"):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    
    # Filter label valid jika ada
    if 'label' in df.columns:
        df = df[df['label'] == 1.0].copy()
        
    column_mapping = {}
    for col in df.columns:
        if col in ["question", "questions", "short_question", "query"]:
            column_mapping[col] = "question"
        elif col in ["answer", "answers", "short_answer", "response"]:
            column_mapping[col] = "answer"
        elif col in ["category", "categories", "tag", "tags", "class"]:
            column_mapping[col] = "category"
            
    df.rename(columns=column_mapping, inplace=True)
    
    if "category" not in df.columns:
        df["category"] = "General"
    else:
        df["category"] = df["category"].astype(str).str.replace(r"[\[\]']", "", regex=True).str.strip()
        df["category"] = df["category"].replace("", "General")
        
    df = df.dropna(subset=["question", "answer"]).drop_duplicates(subset=["question"]).reset_index(drop=True)
    df["question"] = df["question"].astype(str)
    df["answer"] = df["answer"].astype(str)
    df["processed_question"] = df["question"].apply(preprocess_text)
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(df['processed_question'])
    return df, vectorizer, tfidf_matrix

# Inisialisasi engine
try:
    df, vectorizer, tfidf_matrix = init_chatbot()
except Exception as e:
    st.error(f"Gagal memuat dataset: {e}")
    st.stop()

# Rules / Deteksi Intent Manual
RULES = {
    'emergency': {
        'patterns': [
            r'(severe.*chest pain|cannot.*breathe|can\'t.*breathe|unconscious|heavy.*bleeding)',
        ],
        'responses': [
            "🚨 **MEDICAL EMERGENCY!** Call 911 (or your local emergency number 119) or go to the nearest emergency room immediately!"
        ]
    },
    'greeting': {
        'patterns': [
            r'\b(hi|hello|hey|good morning|good evening)\b'
        ],
        'responses': [
            "👋 Hello! How can I assist you with your health questions today?"
        ]
    }
}

def check_rules(text):
    for intent, data in RULES.items():
        for pattern in data['patterns']:
            if re.search(pattern, text.lower()):
                return random.choice(data['responses'])
    return None

def get_bot_response(user_input, threshold=0.15):
    if not user_input.strip():
        return "Please enter a health-related question."
        
    rule_resp = check_rules(user_input)
    if rule_resp:
        return rule_resp

    processed = preprocess_text(user_input)
    vec = vectorizer.transform([processed])
    scores = cosine_similarity(vec, tfidf_matrix).flatten()
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    
    if best_score < threshold:
        return "🤔 I couldn't find a sufficiently relevant answer in the database. Please consult a qualified doctor for advice."
        
    row = df.iloc[best_idx]
    
    return f"""**[Category: {row['category']} | Confidence: {best_score:.2f}]**\n\n{row['answer']}\n\n---\n*⚠️ For informational purposes only. Consult a doctor for clinical concerns.*"""

# --- TAMPILAN STREAMLIT ---
st.title("🩺 Medical Assistant AI")
st.caption("🟢 Online · Medical Question-Answering Retrieval System")
st.info("⚠️ For educational and informational purposes only. Not a substitute for professional medical advice. Emergency: 911 / 119.")

# Inisialisasi riwayat chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your **Medical Assistant**.\n\nAsk me medical questions in English (e.g., *'can antibiotics cause a rash?'*, *'dietary restrictions for celiac disease'*)."
        }
    ]

# Tampilkan pesan sebelumnya
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🩺" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])

# Input pengguna
if user_prompt := st.chat_input("Ask a medical question..."):
    # Tampilkan & simpan input user
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Respon bot
    with st.chat_message("assistant", avatar="🩺"):
        with st.spinner("Searching medical database..."):
            bot_reply = get_bot_response(user_prompt)
            st.markdown(bot_reply)
            
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})