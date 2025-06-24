import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv("emotion_text_dataset.csv")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])

# Streamlit UI
st.title("Emotion Recognition from Text")
st.markdown("How you feeling today? let me guess 🤔 ")

# User input
user_input = st.text_area("Your sentence", "")

if user_input:
    # Vectorize user input
    user_vec = vectorizer.transform([user_input])

    # Compute similarity with dataset
    similarities = cosine_similarity(user_vec, X).flatten()
    best_match_idx = similarities.argmax()

    # Predicted emotion
    predicted_emotion = df.iloc[best_match_idx]['emotion']
    matched_text = df.iloc[best_match_idx]['text']

    st.success(f"**Predicted Emotion:** `{predicted_emotion}`")
    st.info(f"Closest matching sentence in dataset:\n\n_{matched_text}_")

# Show sample data
if st.checkbox("Show Dataset"):
    st.dataframe(df.sample(10))

    # ...existing code...

# ...existing code...

st.markdown(
    """
    <style>
    .stApp {
        background-color: #e3f0ff;
        color: #000000;
    }
    .stTextInput > div > div > input,
    .stTextArea > div > textarea {
        background-color: #cce4ff;
        color: #000000;
    }
    .st-bb, .st-at, .st-cq, .st-cv, .st-cw, .st-cx, .st-cy, .st-cz, .st-da, .st-db, .st-dc, .st-dd, .st-de, .st-df, .st-dg, .st-dh, .st-di, .st-dj, .st-dk, .st-dl, .st-dm, .st-dn, .st-do, .st-dp, .st-dq, .st-dr, .st-ds, .st-dt, .st-du, .st-dv, .st-dw, .st-dx, .st-dy, .st-dz, .st-e0, .st-e1, .st-e2, .st-e3, .st-e4, .st-e5, .st-e6, .st-e7, .st-e8, .st-e9, .st-ea, .st-eb, .st-ec, .st-ed, .st-ee, .st-ef, .st-eg, .st-eh, .st-ei, .st-ej, .st-ek, .st-el, .st-em, .st-en, .st-eo, .st-ep, .st-eq, .st-er, .st-es, .st-et, .st-eu, .st-ev, .st-ew, .st-ex, .st-ey, .st-ez, .st-f0, .st-f1, .st-f2, .st-f3, .st-f4, .st-f5, .st-f6, .st-f7, .st-f8, .st-f9, .st-fa, .st-fb, .st-fc, .st-fd, .st-fe, .st-ff, .st-fg, .st-fh, .st-fi, .st-fj, .st-fk, .st-fl, .st-fm, .st-fn, .st-fo, .st-fp, .st-fq, .st-fr, .st-fs, .st-ft, .st-fu, .st-fv, .st-fw, .st-fx, .st-fy, .st-fz {
        color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ...existing code...