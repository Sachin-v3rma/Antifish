import pandas as pd
import re
import pickle
import warnings
import os
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

warnings.filterwarnings('ignore')

# --- PhishingDetector class remains unchanged ---
class PhishingDetector:
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.best_model = None
        self.best_model_name = None

        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)

    def load_and_preprocess_data(self, filepath):
        print("Loading dataset...")
        try:
            df = pd.read_csv(filepath, encoding='latin1')
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            raise

        print("Checking for required columns...")
        subject_col = 'subject' if 'subject' in df.columns else 'Subject'
        body_col = 'body' if 'body' in df.columns else 'Body'
        label_col = 'label' if 'label' in df.columns else 'Label'

        required_cols = {subject_col, body_col, label_col}
        if not required_cols.issubset(df.columns):
            missing_cols = required_cols - set(df.columns)
            raise KeyError(f"Dataset is missing required columns: {missing_cols}. Please ensure the CSV has 'subject'/'Subject', 'body'/'Body', and 'label'/'Label' columns.")

        df['full_text'] = df[subject_col].fillna('') + ' ' + df[body_col].fillna('')
        print("Preprocessing text...")
        df['processed_text'] = df['full_text'].apply(self.preprocess_text)

        if df[label_col].dtype == 'object':
            print("Mapping text labels to numerical format...")
            df[label_col] = df[label_col].str.lower()
            label_map = {'phishing': 1, 'safe': 0, 'spam': 1, 'ham': 0}
            df['numerical_label'] = df[label_col].map(label_map)
            unmapped_count = df['numerical_label'].isna().sum()
            if unmapped_count > 0:
                print(f"Warning: Found {unmapped_count} rows with unmappable labels. These rows will be dropped.")
                df.dropna(subset=['numerical_label'], inplace=True)
            df['numerical_label'] = df['numerical_label'].astype(int)
        else:
             df['numerical_label'] = df[label_col]

        df = df[df['processed_text'].str.len() > 0].copy()
        print(f"Dataset shape after preprocessing: {df.shape}")
        print(f"Class distribution:\n{df['numerical_label'].value_counts()}")
        return df

    def train_models(self, X_train, y_train):
        print("Training models...")
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.95)
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        models = {
            'Naive Bayes': MultinomialNB(alpha=0.1),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='linear', probability=True, random_state=42)
        }
        results = {}
        for name, model in models.items():
            print(f"Training {name}...")
            cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='accuracy')
            model.fit(X_train_tfidf, y_train)
            self.models[name] = model
            results[name] = {'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(), 'model': model}
            print(f"{name} CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        print(f"\nBest model: {best_model_name}")
        return results

    def evaluate_models(self, X_test, y_test):
        X_test_tfidf = self.vectorizer.transform(X_test)
        print("\n" + "="*50 + "\nMODEL EVALUATION RESULTS\n" + "="*50)
        for name, model in self.models.items():
            y_pred = model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"\n{name}:\nAccuracy: {accuracy:.4f}\n\nClassification Report:\n{classification_report(y_test, y_pred)}")

    def predict_email(self, email_text, return_probabilities=False):
        if not self.best_model or not self.vectorizer:
            raise ValueError("Model not trained yet!")
        processed_text = self.preprocess_text(email_text)
        text_tfidf = self.vectorizer.transform([processed_text])
        prediction = self.best_model.predict(text_tfidf)[0]
        if return_probabilities:
            probabilities = self.best_model.predict_proba(text_tfidf)[0]
            return prediction, probabilities
        return prediction

    def save_model(self, filepath):
        model_data = {'vectorizer': self.vectorizer, 'best_model': self.best_model, 'best_model_name': self.best_model_name, 'all_models': self.models}
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.vectorizer = model_data['vectorizer']
        self.best_model = model_data['best_model']
        self.best_model_name = model_data['best_model_name']
        self.models = model_data['all_models']
        print(f"Model loaded from {filepath}")

class PhishingDetectorGUI:
    def __init__(self, detector):
        self.detector = detector
        self.root = tk.Tk()
        self.root.title("Phishing Email Detection System")
        self.root.geometry("800x600")
        
        # --- NEW: Load sample email files ---
        self.sample_email_dir = "sample_emails"
        self.sample_files = self._load_sample_email_list()
        
        self.setup_gui()
    
    def _load_sample_email_list(self):
        """Finds and prepares the list of sample emails from the directory."""
        if not os.path.isdir(self.sample_email_dir):
            return {}
        
        # Maps user-friendly name to file path
        email_map = {}
        try:
            files = sorted(os.listdir(self.sample_email_dir))
            for filename in files:
                if filename.endswith(".txt"):
                    # Create a clean name for the dropdown
                    clean_name = os.path.splitext(filename)[0] # Remove .txt
                    clean_name = re.sub(r'^\d+[_ ]*', '', clean_name) # Remove leading numbers/underscores
                    clean_name = clean_name.replace('_', ' ').replace('-', ' ').title() # Format it nicely
                    email_map[clean_name] = os.path.join(self.sample_email_dir, filename)
            return email_map
        except Exception as e:
            print(f"Error loading sample emails: {e}")
            return {}

    def setup_gui(self):
        """Setup the GUI interface with a dropdown for samples."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        title_label = ttk.Label(main_frame, text="Phishing Email Detection System", font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        ttk.Label(main_frame, text="Enter Email Content:", font=('Arial', 12)).grid(row=1, column=0, columnspan=3, sticky=tk.W)
        self.email_text = scrolledtext.ScrolledText(main_frame, width=70, height=15, wrap=tk.WORD)
        self.email_text.grid(row=2, column=0, columnspan=3, pady=(5, 10), sticky=(tk.W, tk.E))
        
        # --- MODIFIED: Buttons frame now has a dropdown ---
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="Analyze Email", command=self.analyze_email).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_text).pack(side=tk.LEFT, padx=5)
        
        # --- NEW: Dropdown (Combobox) for samples ---
        if self.sample_files:
            self.sample_dropdown = ttk.Combobox(button_frame, values=list(self.sample_files.keys()), state="readonly")
            self.sample_dropdown.pack(side=tk.LEFT, padx=5)
            self.sample_dropdown.set("Select a Sample...")
            ttk.Button(button_frame, text="Load Sample", command=self.load_selected_sample).pack(side=tk.LEFT, padx=5)

        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="10")
        results_frame.grid(row=4, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        self.result_label = ttk.Label(results_frame, text="", font=('Arial', 12, 'bold'))
        self.result_label.pack()
        self.confidence_label = ttk.Label(results_frame, text="", font=('Arial', 10))
        self.confidence_label.pack()
        self.model_label = ttk.Label(results_frame, text="", font=('Arial', 9))
        self.model_label.pack()
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)

    def analyze_email(self):
        email_content = self.email_text.get(1.0, tk.END).strip()
        if not email_content:
            messagebox.showwarning("Warning", "Please enter email content to analyze.")
            return
        
        try:
            prediction, probabilities = self.detector.predict_email(email_content, return_probabilities=True)
            if prediction == 1:
                result_text = "🚨 PHISHING EMAIL DETECTED"
                result_color = "red"
                confidence = probabilities[1] * 100
            else:
                result_text = "✅ LEGITIMATE EMAIL"
                result_color = "green"
                confidence = probabilities[0] * 100
            
            self.result_label.config(text=result_text, foreground=result_color)
            self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
            self.model_label.config(text=f"Model: {self.detector.best_model_name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during analysis: {str(e)}")

    def clear_text(self):
        self.email_text.delete(1.0, tk.END)
        self.result_label.config(text="")
        self.confidence_label.config(text="")
        self.model_label.config(text="")
    
    # --- NEW: Method to load from the dropdown selection ---
    def load_selected_sample(self):
        """Loads the content of the selected sample email into the text box."""
        selected_key = self.sample_dropdown.get()
        if selected_key == "Select a Sample...":
            messagebox.showinfo("Info", "Please select a sample from the dropdown menu first.")
            return

        filepath = self.sample_files.get(selected_key)
        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.clear_text()
                self.email_text.insert(1.0, content)
            except Exception as e:
                messagebox.showerror("Error", f"Could not load sample file:\n{str(e)}")
        else:
            messagebox.showerror("Error", f"Sample file for '{selected_key}' not found.")

    def run(self):
        self.root.mainloop()

# --- main function remains unchanged ---
def main():
    detector = PhishingDetector()
    model_path = "assets/phishing_detector_model.pkl"
    try:
        detector.load_model(model_path)
        print("Loaded existing model.")
    except FileNotFoundError:
        print("No existing model found. Training a new model.")
        print("-" * 50)
        try:
            dataset_path = "assets/CEAS_08.csv"
        except FileNotFoundError:
            print("Dataset not found in assets folder.")
            dataset_path = input("Enter the path to your dataset CSV file: ")
        try:
            df = detector.load_and_preprocess_data(dataset_path)
            X = df['processed_text']
            y = df['numerical_label']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            detector.train_models(X_train, y_train)
            detector.evaluate_models(X_test, y_test)
            detector.save_model(model_path)
        except FileNotFoundError:
            print(f"\n[ERROR] Dataset file not found at '{dataset_path}'. Please check the file path and try again.")
            return
        except KeyError as e:
            print(f"\n[ERROR] A required column was not found in the dataset. {e}")
            return
        except Exception as e:
            print(f"\n[ERROR] An unexpected error occurred during the training process: {e}")
            return
    except Exception as e:
        print(f"An error occurred before or during GUI launch: {e}")
        return
        
    print("\nLaunching GUI...")
    gui = PhishingDetectorGUI(detector)
    gui.run()

if __name__ == "__main__":
    main()