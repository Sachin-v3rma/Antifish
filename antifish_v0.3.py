# train_model.py

import pandas as pd
import re
import pickle
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# --- Download NLTK data if not present ---
# This ensures that the script can run in any environment without manual setup.
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading required NLTK data packages (punkt, stopwords)...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("Downloads complete.")

warnings.filterwarnings('ignore')

# This class definition MUST BE IDENTICAL to the one in your app.py file
# for the saved model (pickle file) to load correctly.
class PhishingDetector_v3:
    """
    A class to handle the training of a multi-engine phishing detector.
    It preprocesses text data, trains multiple classifiers, and saves them.
    """
    def __init__(self):
        self.vectorizer = None
        self.all_models = {}
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def enhanced_preprocess_text(self, text: str) -> str:
        """
        Processes text by cleaning it and engineering features as special tokens.
        This enhanced method identifies URLs, urgency, and generic greetings.
        """
        if pd.isna(text):
            return ""

        original_text_lower = text.lower()
        feature_tokens = []

        # Feature 1: URL Presence and Suspicion
        urls = re.findall(r'http[s]?://\S+', original_text_lower)
        if urls:
            feature_tokens.append('urlexist')
            suspicious_keywords = ['login', 'verify', 'secure', 'account', 'reset', 'confirm', '.io', '.xyz', '.club', '.top']
            if any(keyword in url for url in urls for keyword in suspicious_keywords):
                feature_tokens.append('suspiciouslink')

        # Feature 2: Urgency Detection
        urgency_words = ['urgent', 'immediately', 'expired', 'warning', 'revoked', 'action required', 'limited time']
        if any(word in original_text_lower for word in urgency_words):
            feature_tokens.append('urgencyword')

        # Feature 3: Generic Greeting Detection
        generic_greetings = ['dear user', 'valued customer', 'dear client', 'hello user']
        if any(greet in original_text_lower for greet in generic_greetings):
            feature_tokens.append('genericgreet')

        # --- Standard Text Cleaning ---
        # Clean up the text after feature extraction
        clean_text = re.sub(r'http[s]?://\S+', '', original_text_lower)
        clean_text = re.sub(r'<[^>]+>', '', clean_text)  # Remove HTML tags
        clean_text = re.sub(r'\S+@\S+', '', clean_text)  # Remove email addresses
        clean_text = re.sub(r'[^a-zA-Z\s]', '', clean_text)  # Remove special characters/digits
        clean_text = ' '.join(clean_text.split())  # Normalize whitespace

        # Tokenize, remove stopwords, and stem the remaining words
        tokens = word_tokenize(clean_text)
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words and len(token) > 2]

        # The final processed text is a combination of stemmed words and feature tokens
        return ' '.join(stemmed_tokens + feature_tokens)

    def load_and_preprocess_data(self, filepath: str) -> pd.DataFrame:
        """
        Loads a CSV dataset and applies the enhanced preprocessing.
        It intelligently finds the required columns regardless of case.
        """
        print(f"Loading dataset from {filepath}...")
        try:
            df = pd.read_csv(filepath, encoding='latin1')
        except FileNotFoundError:
            print(f"[ERROR] The file at {filepath} was not found.")
            return None

        # Intelligently find column names, ignoring case
        subject_col = next((col for col in df.columns if col.lower() == 'subject'), 'Subject')
        body_col = next((col for col in df.columns if col.lower() == 'body'), 'Body')
        label_col = next((col for col in df.columns if col.lower() == 'label'), 'Label')

        if subject_col not in df.columns or body_col not in df.columns or label_col not in df.columns:
             print(f"[ERROR] Dataset must contain columns for 'Subject', 'Body', and 'Label'. Check your CSV file.")
             return None

        df['full_text'] = df[subject_col].fillna('') + ' ' + df[body_col].fillna('')
        print("Preprocessing text with enhanced feature logic...")
        df['processed_text'] = df['full_text'].apply(self.enhanced_preprocess_text)

        # Convert text labels to binary (1 for phishing/spam, 0 for legitimate)
        df['numerical_label'] = df[label_col].apply(lambda x: 1 if str(x).lower() in ['phishing', 'spam', '1'] else 0)
        
        # Clean up the dataframe
        df.dropna(subset=['processed_text'], inplace=True)
        df = df[df['processed_text'].str.len() > 0].copy()
        
        print("\nDataset preprocessing complete.")
        print(f"Class distribution:\n{df['numerical_label'].value_counts(normalize=True)}")
        return df

    def train(self, df: pd.DataFrame):
        """
        Trains all specified machine learning models and evaluates them.
        """
        print("\nStarting model training process...")
        X = df['processed_text']
        y = df['numerical_label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Initialize the TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=3)
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        # Define the models to be trained
        models_to_train = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'SVM': SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        }

        print("\n" + "="*50)
        print("MODEL EVALUATION ON TEST SET")
        print("="*50)
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_tfidf, y_train)
            self.all_models[name] = model  # Store the trained model
            
            y_pred = model.predict(X_test_tfidf)
            print(f"  -> Test Accuracy for {name}: {accuracy_score(y_test, y_pred):.4f}")
            # Uncomment below to see a detailed report for each model during training
            # print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

    def save_model(self, filepath: str):
        """
        Saves the vectorizer and all trained models to a single pickle file.
        """
        # Ensure the 'assets' directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'vectorizer': self.vectorizer,
            'all_models': self.all_models
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\n✅ Multi-engine model saved successfully to {filepath}")

def main():
    """
    The main function to orchestrate the training process.
    """
    detector = PhishingDetector_v3()
    model_path = "assets/phishing_detector_model_v3.pkl"
    
    print("--- Phishing Detector Model Training Script (v3) ---")
    dataset_path = input("Enter the path to your dataset CSV file (e.g., assets/CEAS_08.csv): ")

    if not os.path.exists(dataset_path):
        print(f"\n[ERROR] Dataset file not found: '{dataset_path}'")
        return

    df = detector.load_and_preprocess_data(dataset_path)
    
    if df is not None:
        detector.train(df)
        detector.save_model(model_path)
        print("\n--- Training complete. You can now run the Streamlit app using: streamlit run app.py ---")

if __name__ == "__main__":
    main()
