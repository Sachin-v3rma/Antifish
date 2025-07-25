# 🛡️ Multi-Engine Phishing Detector

An advanced, web-based phishing detection tool that uses multiple machine learning models to analyze emails. Inspired by VirusTotal, this application provides a more robust and reliable verdict by leveraging the consensus of several detection "engines."

## Key Features

-   **Multi-Engine Analysis:** Uses Logistic Regression, SVM, and Random Forest simultaneously to evaluate emails, reducing the risk of a single model's blind spot.
-   **VirusTotal-Style Verdict:** Presents a final verdict based on a majority vote from the trained models.
-   **Interactive Web UI:** A clean and modern user interface built with Streamlit for easy use.
-   **Detailed Reporting:** Includes an "Engine Breakdown" table and a confidence level bar chart for a transparent analysis.
-   **Intelligent Feature Engineering:** Goes beyond simple word matching by creating special feature tokens for URLs (`urlexist`), suspicious links (`suspiciouslink`), and urgent language (`urgencyword`).
-   **Sample Emails:** Comes with a dropdown of pre-loaded sample emails for quick and easy testing.

## Project Structure

A well-organized project structure is key. Ensure your directory looks like this:

```
phishing-detector/
│
├── assets/
│   └── phishing_detector_model_v3.pkl  # The multi-engine model created by the training script
│
├── sample_emails/
│   ├── 01_Sample_Phishing.txt
│   └── 02_Sample_Legitimate.txt
│
├── app.py                         # The main Streamlit web application file
├── antifish.py                 # The script to train the model from a dataset
└── requirements.txt               # List of project dependencies
```

## How to Set Up and Run

Follow these steps to get the phishing detector running on your local machine.

### Prerequisites

-   Python 3.8+
-   `pip` for installing packages

### Step 1: Clone and Install Dependencies

First, clone this repository to your local machine and navigate into the project directory. Then, install all the required libraries using the `requirements.txt` file.

```bash
git clone https://github.com/Sachin-v3rma/Antifish
cd Antifish
pip install -r requirements.txt
```

### Step 2: Train the Model (One-Time Setup)

Before you can run the web app, you must train the machine learning models using your dataset. This process creates the `phishing_detector_model_v3.pkl` file that the app relies on.

Run the training script from your terminal:

```bash
python antifish_v0.3.py
```

The script will prompt you to enter the path to your dataset. A CSV file with 'Subject', 'Body', and 'Label' columns is expected. For example:

```
Enter the path to your dataset CSV file (e.g., assets/CEAS_08.csv): assets/CEAS_08.csv```

Let the script run to completion. It will preprocess the data, train the models, and save the final `.pkl` file in the `assets/` directory.

### Step 3: Run the Streamlit Web App

Once the model file has been created, you can launch the web application.

```bash
streamlit run app.py
```

Your default web browser will automatically open a new tab with the Multi-Engine Phishing Detector running. You can now paste emails, use the samples, and see the analysis dashboard in action!

## How It Works: The Detection Logic

This detector's accuracy is enhanced by **feature engineering**. Instead of just looking at words, the preprocessing pipeline identifies key phishing indicators and converts them into special tokens that the models can learn from:

-   `urlexist`: Added if any URL is found.
-   `suspiciouslink`: Added if a URL contains common phishing keywords (e.g., "login", "verify") or suspicious top-level domains (e.g., ".xyz", ".club").
-   `urgencyword`: Added if the email contains high-pressure words like "urgent", "expired", or "action required".
-   `genericgreet`: Added for impersonal greetings like "Dear user" or "Valued customer".

By training on these tokens in addition to the regular email text, the models become much better at identifying the *tactics* of phishing, not just the vocabulary.

## Dependencies (`requirements.txt`)

```
pandas
scikit-learn
nltk
streamlit
```
