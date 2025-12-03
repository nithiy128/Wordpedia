import streamlit as st
import google.generativeai as genai
import pandas as pd
import matplotlib.pyplot as plt
import requests
import numpy as np
import json

# ---- Page Config ----
st.set_page_config(page_title="Wordpedia", page_icon="📚", layout="wide")

# ---- Custom CSS ----
st.markdown("""
<style>

body {
    background-color: #f8f9fc;
    font-family: "Segoe UI", sans-serif;
}

/* Title styling */
.wordpedia-title {
    font-size: 60px;
    font-weight: 800;
    color: #4A4A4A;
    text-align: center;
    margin-top: -30px;
    letter-spacing: 2px;
}

/* Subheader */
.wordpedia-sub {
    font-size: 24px;
    text-align: center;
    color: #6C6C6C;
    margin-bottom: 10px;
}
            
/* Caption styling */
.wordpedia-caption {
    font-size: 16px;
    text-align: center;
    color: #8C8C8C;
    margin-bottom: 30px;
}

/* Card container */
.card {
    background: white;
    padding: 25px 30px;
    border-radius: 18px;
    box-shadow: 0px 8px 30px rgba(0, 0, 0, 0.06);
    margin-top: 25px;
}

/* Glow highlight */
.card:hover {
    box-shadow: 0px 12px 40px rgba(0, 0, 0, 0.12);
}

/* Paragraph styling */
p {
    font-size: 18px;
    color: #333333;
    line-height: 1.6;
}

</style>
""", unsafe_allow_html=True)

# ---- Page Content ----
st.markdown('<div class="wordpedia-title">📘 Wordpedia</div>', unsafe_allow_html=True)
st.markdown('<div class="wordpedia-sub">Your Ultimate Word Reference Tool</div>', unsafe_allow_html=True)
st.caption('<div class="wordpedia-caption">Discover meanings, synonyms, antonyms, and more!</div>', unsafe_allow_html=True)

st.markdown("""
<div class="card">
    <p>
    Welcome to <b>Wordpedia</b>, your go-to application for exploring the fascinating world of words!  
    Whether you're a writer, student, or just a curious mind, Wordpedia has something for everyone.
    </p> 
    Start using Wordpedia by entering your api code, then enter a word in the search bar below this box.  
    You can look up definitions, origin of the word, how often it's used, and even find synonyms and antonyms to expand your vocabulary.
    
    Disclaimer: It might take a few seconds to fetch the data, so please be patient!
           
</div>
""", unsafe_allow_html=True)

def fetch_ngram_data(word):
        url = ("https://books.google.com/ngrams/json?"
        f"content={word}&year_start=1800&year_end=2019&corpus=26&smoothing=7")

        response = requests.get(url)
        if response.status_code != 200:
            return None
        json_data = response.json()
        if not json_data:
            return None
        
        timeseries = json_data[0]['timeseries']
        years = list(range(1800, 2020))
        df = pd.DataFrame({'Year': years, 'Frequency': timeseries})
        return df
    
api_container = st.empty()
next_input_container = st.empty()

if "api_validated" not in st.session_state:
    st.session_state.api_validated = False
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

def validate_api_key():
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        model.generate_content("hello")
        return True
    except Exception as e:
        st.error(f"API Key validation failed: {e}")
        return False

def get_emnbedding(word):
    result = genai.embed_content(
        model="text-embedding-004",
        content=word
    )
    return np.array(result["embedding"])

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def generate_syn_ant_tables(word):
    prompt = f"""Give me 10 synonyms and 10 antonyms for the word '{word}'.
    Return  STRICT JSON with this structure:
      {{
        "synonyms": ["word1", "word2", ...],
        "antonyms": ["word1", "word2", ...]
    }}
    Do NOT add any extra information, ONLY return the JSON."""

    response = genai.GenerativeModel("gemini-2.5-flash-lite").generate_content(prompt)

    json_text = response.text.strip().replace("```json", "").replace("```", "")
    data = json.loads(json_text)
    synonyms = data.get("synonyms", [])
    antonyms = data.get("antonyms", [])

    word_vec = get_emnbedding(word)
    syn_rows = []
    for s in synonyms:
        score = cosine_similarity(word_vec, get_emnbedding(s))
        syn_rows.append((s, round(float(score), 4)))
    syn_df = pd.DataFrame(syn_rows, columns=["Word", "Similarity"])

    ant_rows = []
    for a in antonyms:
        score = cosine_similarity(word_vec, get_emnbedding(a))
        ant_rows.append((a, round(float(score), 4)))
    ant_df = pd.DataFrame(ant_rows, columns=["Word", "Similarity"])

    return syn_df.sort_values(by="Similarity", ascending=False), ant_df.sort_values(by="Similarity", ascending=False)
  

if not st.session_state.api_validated:
    with api_container.container():
        api_key = st.text_input("Enter your Google Generative AI API Key:", type="password")
        if api_key:
            if validate_api_key():
                st.session_state.api_validated = True
                st.session_state.api_key = api_key
                st.success("API Key validated successfully!")
                api_container.empty()
            else:
                st.error("Invalid API Key. Please try again.")
                
if st.session_state.api_validated:
    with next_input_container.container():
        user_input = st.text_input("Insert a word to search:", "")
    if user_input:
        st.write("Wordpedia of ", user_input)

        genai.configure(api_key=st.session_state.api_key)
        
        prompt = f"""
        Provide a detailed wordpedia for the word '{user_input}' including:

        - Definition from Cambridge Dictionary  
        - Origin of the word  

        Format the response as:  
        1. Definition (paragraph)  
        2. Origin (paragraph)  
        """
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        
        response = model.generate_content(prompt)
        
        st.markdown(response.text)

        df = fetch_ngram_data(user_input)
        if df is not None:
            st.subheader("📈 Usage Frequency (Google Ngram)")
        
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(df["Year"], df["Frequency"], color="#4A90E2", linewidth=2)
            ax.fill_between(df["Year"], df["Frequency"], color="#4A90E2", alpha=0.1)
            ax.set_xlabel("Year", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.set_title(f"Ngram usage for '{user_input}'", fontsize=14)
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        else:
            st.warning("No Ngram data found for this word.")

        st.markdown("### 🔄 Synonyms and Antonyms")
        syn_table, ant_table = generate_syn_ant_tables(user_input)
        syn_table = syn_table.reset_index(drop=True)
        ant_table = ant_table.reset_index(drop=True)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Synonyms")
            st.dataframe(syn_table)
        with col2:
            st.subheader("Antonyms")
            st.dataframe(ant_table)