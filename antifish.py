# Made with 🖤 by Viem

import pandas as pd
import re
import pickle
import warnings
import os

# ML/NLP Libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# --- Download NLTK data if not present ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading required NLTK data packages (punkt, stopwords)...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("Downloads complete.")

warnings.filterwarnings('ignore')

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

    def load_and_preprocess_data(self, filepath: str) -> pd.DataFrame:
        print(f"Loading dataset from {filepath}...")
        try:
            df = pd.read_csv(filepath, encoding='latin1')
        except FileNotFoundError:
            print(f"[ERROR] The file at {filepath} was not found.")
            return None
        subject_col = next((col for col in df.columns if col.lower() == 'subject'), 'Subject')
        body_col = next((col for col in df.columns if col.lower() == 'body'), 'Body')
        label_col = next((col for col in df.columns if col.lower() == 'label'), 'Label')
        if not all(col in df.columns for col in [subject_col, body_col, label_col]):
             print(f"[ERROR] Dataset must contain 'Subject', 'Body', and 'Label' columns.")
             return None
        df['full_text'] = df[subject_col].fillna('') + ' ' + df[body_col].fillna('')
        print("Preprocessing text with advanced forensic logic...")
        df['processed_text'] = df['full_text'].apply(self.enhanced_preprocess_text)
        df['numerical_label'] = df[label_col].apply(lambda x: 1 if str(x).lower() in ['phishing', 'spam', '1'] else 0)
        df.dropna(subset=['processed_text'], inplace=True)
        df = df[df['processed_text'].str.len() > 0].copy()
        return df

    def train(self, df: pd.DataFrame):
        print("\nStarting model training process...")
        X = df['processed_text']
        y = df['numerical_label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=5)
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        models_to_train = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'SVM': SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=120, random_state=42, class_weight='balanced')
        }
        print("\n" + "="*50 + "\nMODEL EVALUATION ON TEST SET\n" + "="*50)
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_tfidf, y_train)
            self.all_models[name] = model
            y_pred = model.predict(X_test_tfidf)
            print(f"  -> Test Accuracy for {name}: {accuracy_score(y_test, y_pred):.4f}")

    def save_model(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {'vectorizer': self.vectorizer, 'all_models': self.all_models}
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\n✅ Model saved successfully to {filepath}")

def main():
    detector = PhishingDetector()
    model_path = "assets/phishing_detector_model.pkl"
    print("--- Phishing Detector Model Training Script ---")
    try:
        dataset_path = "assets/CEAS_08.csv"
    except FileNotFoundError:
        print("Dataset not found in assets folder.")
        dataset_path = input("Enter the path to your dataset CSV file: ")
    if not os.path.exists(dataset_path):
        print(f"\n[ERROR] Dataset file not found: '{dataset_path}'")
        return
    df = detector.load_and_preprocess_data(dataset_path)
    if df is not None:
        detector.train(df)
        detector.save_model(model_path)
        print("\n--- Training complete. You can now run the Streamlit app. ---")

if __name__ == "__main__":
    main()
