# app.py

import streamlit as st
import pandas as pd
import os
import re
import pickle
import warnings

# --- ML/NLP Libraries ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

warnings.filterwarnings('ignore')

# This class contains the most advanced detection logic. It is defined here
# to avoid import errors and must be identical to the class in train_model.py.
class PhishingDetector:
    def __init__(self):
        self.vectorizer = None
        self.all_models = {}
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def enhanced_preprocess_text(self, text: str) -> str:
        if pd.isna(text): return ""
        original_text_lower = text.lower()
        feature_tokens = []
        emails = re.findall(r'\S+@\S+', original_text_lower)
        if emails:
            high_risk_tlds = ['.ru', '.xyz', '.top', '.club', '.site', '.link', '.tk', '.biz']
            freemail_providers = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com', 'mail.ru']
            for email in emails:
                try:
                    domain = email.split('@')[1]
                    if any(tld in domain for tld in high_risk_tlds):
                        feature_tokens.append('highriskdomain')
                    if domain in freemail_providers:
                        feature_tokens.append('freemailprovider')
                except IndexError:
                    continue
        urls = re.findall(r'http[s]?://\S+', original_text_lower)
        if urls:
            feature_tokens.append('urlexist')
            if any(url.startswith('http://') for url in urls):
                feature_tokens.append('insecurehttp')
            suspicious_keywords = ['login', 'verify', 'secure', 'account', 'reset', 'confirm', 'update', 'signin', 'bank']
            if any(keyword in url for url in urls for keyword in suspicious_keywords):
                feature_tokens.append('suspiciouslink')
        urgency_words = ['urgent', 'immediately', 'expired', 'warning', 'revoked', 'suspension', 'lockout', 'action required']
        if any(word in original_text_lower for word in urgency_words):
            feature_tokens.append('urgencyword')
        generic_greetings = ['dear user', 'valued customer', 'dear client', 'hello user']
        if any(greet in original_text_lower for greet in generic_greetings):
            feature_tokens.append('genericgreet')
        symbols = re.findall(r'[!👉🔐™@#$*]', text)
        if len(symbols) > 2:
            feature_tokens.append('excessivesymbols')
        clean_text = re.sub(r'http[s]?://\S+', '', original_text_lower)
        clean_text = re.sub(r'<[^>]+>', '', clean_text)
        clean_text = re.sub(r'\S+@\S+', '', clean_text)
        clean_text = re.sub(r'[^a-zA-Z\s]', '', clean_text)
        clean_text = ' '.join(clean_text.split())
        tokens = word_tokenize(clean_text)
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words and len(token) > 2]
        return ' '.join(stemmed_tokens + feature_tokens)

    def predict_with_all_models(self, email_text: str) -> list:
        if not self.all_models: raise ValueError("Models not loaded!")
        processed_text = self.enhanced_preprocess_text(email_text)
        text_tfidf = self.vectorizer.transform([processed_text])
        results = []
        for name, model in self.all_models.items():
            prediction = model.predict(text_tfidf)[0]
            probabilities = model.predict_proba(text_tfidf)[0]
            verdict, confidence = ("Phishing", probabilities[1] * 100) if prediction == 1 else ("Legitimate", probabilities[0] * 100)
            results.append({"Model": name, "Verdict": verdict, "Confidence": f"{confidence:.2f}%"})
        return results

    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.vectorizer = model_data['vectorizer']
        self.all_models = model_data['all_models']
        print(f"Multi-engine model loaded from {filepath}")

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Forensic Phishing Detector",
    page_icon="🛡️",
    layout="wide"
)

# --- Caching Functions ---
@st.cache_resource
def load_detector():
    """Loads the multi-engine phishing detector from the standard pickle file."""
    detector = PhishingDetector()
    # Standard model filename, no versioning.
    model_path = "assets/phishing_detector_model.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at '{model_path}'. Please run `train_model.py` first to create it.")
        return None
    try:
        detector.load_model(model_path)
        return detector
    except Exception as e:
        st.error(f"Fatal error loading model: {e}")
        return None

@st.cache_data
def load_sample_emails():
    """Loads all .txt sample emails from the 'sample_emails' directory."""
    sample_dir = "sample_emails"
    email_map = {"-- Select a Sample --": ""}
    if os.path.isdir(sample_dir):
        for filename in sorted(os.listdir(sample_dir)):
            if filename.endswith(".txt"):
                clean_name = os.path.splitext(filename)[0].replace('_', ' ').replace('-', ' ').title()
                filepath = os.path.join(sample_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        email_map[clean_name] = f.read()
                except Exception as e:
                    print(f"Could not load sample file {filename}: {e}")
    return email_map

# --- Main App Interface ---
st.title("🛡️ Forensic Phishing Detector")
st.markdown("This scanner uses multiple machine learning models and advanced forensic analysis to detect sophisticated phishing attempts.")

detector = load_detector()
sample_emails = load_sample_emails()

if detector:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("### Email Input")
        selected_sample = st.selectbox(
            "Load a sample email:",
            options=list(sample_emails.keys()),
            label_visibility="collapsed"
        )
        
        email_input = st.text_area(
            "Or paste the full email content here:",
            value=sample_emails[selected_sample],
            height=450,
            label_visibility="collapsed"
        )
        
        analyze_button = st.button("Analyze with All Engines", type="primary", use_container_width=True)

    with col2:
        st.write("### Analysis Dashboard")
        
        if analyze_button and email_input.strip():
            # Perform prediction with all models
            results = detector.predict_with_all_models(email_input)
            df = pd.DataFrame(results)
            
            # --- Final Verdict based on Majority Vote ---
            phishing_votes = (df['Verdict'] == 'Phishing').sum()
            total_votes = len(df)
            
            st.write("**Final Verdict**")
            if phishing_votes > total_votes / 2:
                st.error(f"**PHISHING DETECTED** ({phishing_votes}/{total_votes} engines agree)", icon="🚨")
            else:
                st.success(f"**LIKELY LEGITIMATE** ({total_votes - phishing_votes}/{total_votes} engines agree)", icon="✅")

            # --- Engine Breakdown Table ---
            st.write("**Engine Breakdown**")
            
            def highlight_verdict(val):
                color = 'red' if val == 'Phishing' else 'green'
                return f'color: {color}; font-weight: bold;'
            
            st.dataframe(
                df.style.applymap(highlight_verdict, subset=['Verdict']),
                use_container_width=True
            )

            # --- Confidence Chart ---
            st.write("**Confidence Levels**")
            # Convert confidence from string "98.81%" to float 98.81
            df_chart = df.copy()
            df_chart['Confidence'] = df_chart['Confidence'].str.replace('%', '').astype(float)
            
            st.bar_chart(df_chart.set_index('Model')['Confidence'])
        
        elif analyze_button and not email_input.strip():
            st.warning("Please enter or load email content to analyze.")
        else:
            st.info("Analysis results from all engines will be displayed here.")
else:
    st.warning("Detector model could not be loaded. Please follow the training instructions carefully.")
