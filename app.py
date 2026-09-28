import os
import re
import random
import time
import pandas as pd
import numpy as np
import streamlit as st

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. SETUP RESOURCE NLTK & LEMMATIZER
# ==========================================
for res in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
    try:
        nltk.download(res, quiet=True)
    except Exception:
        pass

lemmatizer = WordNetLemmatizer()

try:
    english_stopwords = set(stopwords.words("english"))
except Exception:
    english_stopwords = {'the', 'is', 'are', 'was', 'what', 'how', 'why', 'when', 'where', 'a', 'an'}

indonesian_stopwords = {
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk',
    'pada', 'adalah', 'atau', 'juga', 'dalam', 'tidak', 'akan', 'ada',
    'saya', 'kamu', 'anda', 'ia', 'mereka', 'kami', 'kita', 'bisa',
    'sudah', 'bila', 'jika', 'maka', 'oleh', 'karena', 'apa',
    'bagaimana', 'berapa', 'kapan', 'dimana', 'siapa', 'apakah'
}
all_stopwords = english_stopwords | indonesian_stopwords


def preprocess_text(text: str) -> str:
    """Membersihkan, menormalisasi, menyaring stopwords, dan lematisasi."""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in all_stopwords and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# ==========================================
# 2. CLASS CHATBOT ENGINE
# ==========================================
class MedicalChatbotEngineV3:
    def __init__(self, dataframe: pd.DataFrame, sbert_model=None, threshold=0.35, top_k=3):
        self.df = dataframe.copy()
        self.top_k = top_k
        self.sbert_model = sbert_model
        self.use_sbert = sbert_model is not None
        self.threshold = threshold if self.use_sbert else 0.25

        self.vectorizer = None
        self.tfidf_matrix = None
        self.sbert_embeddings = None

        self._define_rules()
        self._build_index()

    def _build_index(self):
        # 1. Matriks TF-IDF
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['processed_question'])

        # 2. Embedding Sentence-BERT
        if self.use_sbert:
            self.sbert_embeddings = self.sbert_model.encode(
                self.df['question'].tolist(),
                convert_to_tensor=True,
                show_progress_bar=False
            )

    def _define_rules(self):
        self.rules = {
            'emergency': {
                'patterns': [
                    r'(severe.*chest pain|cannot.*breathe|can\'t.*breathe|unconscious|heavy.*bleeding|serangan jantung|henti napas|pingsan)',
                ],
                'responses': [
                    "🚨 **MEDICAL EMERGENCY!** Call 911 (or local emergency 119) or go to the nearest emergency room immediately!"
                ]
            },
            'greeting': {
                'patterns': [
                    r'\b(hi|hello|hey|good morning|good evening|halo|hai)\b'
                ],
                'responses': [
                    "👋 **Hello!** How can I assist you with your medical or health questions today?"
                ]
            },
            'capability': {
                'patterns': [
                    r'(what can you do|capabilities|help me with|features|topik apa|bisa apa)'
                ],
                'responses': [
                    "ℹ️ I am a medical QA assistant trained on curated doctor-patient dialogues. You can ask me about symptoms, diseases, medications, and general health guidelines."
                ]
            }
        }

    def _check_rules(self, text: str):
        for intent, data in self.rules.items():
            for pattern in data['patterns']:
                if re.search(pattern, text.lower()):
                    return random.choice(data['responses'])
        return None

    def _search_sbert(self, query: str):
        from sentence_transformers import util
        emb = self.sbert_model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(emb, self.sbert_embeddings)[0].cpu().numpy()
        top_results = np.argsort(-scores)[:self.top_k]
        return [(idx, float(scores[idx])) for idx in top_results]

    def _search_tfidf(self, query: str):
        processed = preprocess_text(query)
        vec = self.vectorizer.transform([processed])
        scores = cosine_similarity(vec, self.tfidf_matrix).flatten()
        top_results = np.argsort(scores)[::-1][:self.top_k]
        return [(idx, float(scores[idx])) for idx in top_results]

    def _build_context_query(self, user_input: str, conversation_history: list) -> str:
        if len(conversation_history) > 0:
            last_user_query = conversation_history[-1]
            return f"{last_user_query} {user_input}"
        return user_input

    def get_response(self, user_input: str, conversation_history: list) -> dict:
        if not user_input.strip():
            return {
                "answer": "Please enter a health-related question.",
                "category": "Notice",
                "score": 0.0,
                "method": "None",
                "is_rule": True
            }

        rule_resp = self._check_rules(user_input)
        if rule_resp:
            return {
                "answer": rule_resp,
                "category": "Direct Rule",
                "score": 1.0,
                "method": "Rule Engine",
                "is_rule": True
            }

        query = self._build_context_query(user_input, conversation_history)

        if self.use_sbert:
            results = self._search_sbert(query)
            method = "Sentence-BERT"
        else:
            results = self._search_tfidf(query)
            method = "TF-IDF"

        best_idx, best_score = results[0]

        if best_score < self.threshold:
            return {
                "answer": "🤔 I couldn't find a sufficiently relevant answer in the database. Please consult a qualified doctor or clinic for clinical diagnosis.",
                "category": "Unknown / Out-of-Scope",
                "score": round(best_score, 2),
                "method": method,
                "is_rule": False
            }

        # Keyword Re-Ranking (+0.05 per kecocokan kata)
        boosted = []
        user_words = [w for w in user_input.lower().split() if w not in all_stopwords and len(w) > 2]
        for idx, score in results:
            text = self.df.iloc[idx]['question'].lower()
            bonus = sum(1 for word in user_words if word in text)
            boosted.append((idx, score + 0.05 * bonus))

        best_idx = sorted(boosted, key=lambda x: x[1], reverse=True)[0][0]
        row = self.df.iloc[best_idx]

        return {
            "answer": str(row['answer']),
            "category": str(row['category']),
            "matched_question": str(row['question']),
            "score": round(best_score, 2),
            "method": method,
            "is_rule": False
        }


# ==========================================
# 3. STREAMLIT CACHING (LOAD SEKALI SAJA)
# ==========================================
@st.cache_resource(show_spinner="Sedang memuat Dataset & Model NLP (SBERT)...")
def load_chatbot_engine():
    csv_file = "train_data_chatbot1.csv"
    if not os.path.exists(csv_file):
        for candidate in ["train_data_chatbot.csv", "data/train_data_chatbot1.csv"]:
            if os.path.exists(candidate):
                csv_file = candidate
                break

    if not os.path.exists(csv_file):
        st.error(f"File `{csv_file}` tidak ditemukan di direktori saat ini.")
        st.stop()

    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip().str.lower()

    if 'label' in df.columns:
        df = df[df['label'] == 1.0].copy()

    col_map = {}
    for col in df.columns:
        if col in ["question", "questions", "short_question", "query"]:
            col_map[col] = "question"
        elif col in ["answer", "answers", "short_answer", "response"]:
            col_map[col] = "answer"
        elif col in ["category", "categories", "tag", "tags", "class"]:
            col_map[col] = "category"
    df.rename(columns=col_map, inplace=True)

    if "category" not in df.columns:
        df["category"] = "General"
    else:
        df["category"] = df["category"].astype(str).str.replace(r"[\[\]']", "", regex=True).str.strip()
        df["category"] = df["category"].replace("", "General")

    df = df.dropna(subset=["question", "answer"]).drop_duplicates(subset=["question"]).reset_index(drop=True)
    df["question"] = df["question"].astype(str)
    df["answer"] = df["answer"].astype(str)
    df["processed_question"] = df["question"].apply(preprocess_text)

    # Inisialisasi SBERT
    sbert_model = None
    try:
        from sentence_transformers import SentenceTransformer
        sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    except Exception as e:
        print(f"SBERT tidak tersedia: {e}")

    engine = MedicalChatbotEngineV3(df, sbert_model=sbert_model)
    return engine


# ==========================================
# 4. STREAMLIT UI CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="MedBot AI - Smart Medical Assistant",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS Tampilan Modern
st.markdown("""
<style>
    .med-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f766e;
        margin-bottom: 0.2rem;
    }
    .med-subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .alert-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 10px 14px;
        border-radius: 6px;
        color: #92400e;
        font-size: 0.88rem;
        margin-bottom: 1rem;
    }
    .badge-cat {
        background-color: #ccfbf1;
        color: #0f766e;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-method {
        background-color: #f1f5f9;
        color: #475569;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Inisialisasi Engine
engine = load_chatbot_engine()

# Inisialisasi Session State Percakapan
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am **MedBot v3**, your medical QA assistant powered by **Sentence-BERT**.\n\nFeel free to ask health-related questions or select one of the suggested topics below."
        }
    ]

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# ==========================================
# 5. SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/medical-doctor.png", width=64)
    st.title("🩺 MedBot System")
    st.caption("Context-Aware Semantic Medical Assistant")
    
    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.metric("Total Q&A Valid", f"{len(engine.df):,}")
    st.metric("Topik Kategori", len(engine.df['category'].unique()))
    st.markdown(f"**Model Mesin:** `{'Sentence-BERT (Active)' if engine.use_sbert else 'TF-IDF Fallback'}`")
    
    st.markdown("---")
    st.markdown("""
    <div class="alert-box">
        🚨 <strong>Darurat Medis?</strong><br>
        Jika mengalami sesak napas akut atau nyeri dada berat, segera hubungi <strong>911 / 119</strong> atau kunjungi IGD terdekat.
    </div>
    """, unsafe_allow_html=True)
    
    # Tombol Reset Percakapan
    if st.button("🔄 Reset Percakapan", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat telah direset. Silakan tanyakan keluhan atau pertanyaan kesehatan baru Anda!"
            }
        ]
        st.session_state.conversation_history = []
        st.rerun()

# ==========================================
# 6. KONTEN UTAMA
# ==========================================
st.markdown('<div class="med-title">Medical Assistant AI</div>', unsafe_allow_html=True)
st.markdown('<div class="med-subtitle">Temu Balik Informasi Medis berbasis Semantik Sentence-BERT & TF-IDF</div>', unsafe_allow_html=True)

# Tombol Rekomendasi Cepat (Quick Chips)
st.markdown("**⚡ Coba Pertanyaan Cepat:**")
chip_cols = st.columns(5)
quick_queries = [
    ("💊 Antibiotic Rash", "can an antibiotic through an iv give you a rash?"),
    ("🌾 Celiac Diet", "what are the dietary restrictions for celiac disease gluten?"),
    ("🫁 Walking Pneumonia", "is walking pneumonia contagious?"),
    ("🩸 Blood Pressure", "how to treat high blood pressure?"),
    ("🚨 Chest Pain", "severe chest pain cannot breathe")
]

selected_chip = None
for i, (label, query_text) in enumerate(quick_queries):
    with chip_cols[i]:
        if st.button(label, use_container_width=True, key=f"chip_{i}"):
            selected_chip = query_text

# Render Riwayat Pesan
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🩺"):
        st.markdown(msg["content"])
        if "meta" in msg:
            meta = msg["meta"]
            if not meta.get("is_rule", False):
                st.markdown(
                    f'<span class="badge-cat">Kategori: {meta["category"]}</span>'
                    f'<span class="badge-method">{meta["method"]} (Similarity: {meta["score"]})</span>',
                    unsafe_allow_html=True
                )

# Input Box Chat Pengguna
user_prompt = st.chat_input("Ketik pertanyaan kesehatan di sini...")
active_input = selected_chip if selected_chip else user_prompt

if active_input:
    # 1. Tampilkan pesan user
    st.session_state.messages.append({"role": "user", "content": active_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(active_input)

    # 2. Proses jawaban bot
    with st.chat_message("assistant", avatar="🩺"):
        with st.spinner("Mencari jawaban dokter yang paling relevan..."):
            result = engine.get_response(active_input, st.session_state.conversation_history)
            
            # Update riwayat context pertanyaan user
            st.session_state.conversation_history.append(active_input)
            
            st.markdown(result["answer"])
            if not result.get("is_rule", False):
                st.markdown(
                    f'<span class="badge-cat">Kategori: {result["category"]}</span>'
                    f'<span class="badge-method">{result["method"]} (Similarity: {result["score"]})</span>',
                    unsafe_allow_html=True
                )

    # 3. Simpan ke session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "meta": result
    })