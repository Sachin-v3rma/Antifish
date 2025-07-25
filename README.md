# Antifish

This project provides a comprehensive solution for detecting phishing emails using Natural Language Processing (NLP) and a variety of Machine Learning models. It includes a Python script to train and evaluate multiple classifiers and a graphical user interface (GUI) built with Tkinter to analyze emails in real-time.

## Features

*   **Advanced Text Preprocessing:** Implements a full NLP pipeline including lowercasing, removal of URLs, HTML tags, email addresses, and special characters. It also includes tokenization, stop-word removal, and stemming.
*   **Multiple Machine Learning Models:** Trains and evaluates the following popular classification algorithms:
    *   Naive Bayes
    *   Logistic Regression
    *   Random Forest
    *   Support Vector Machine (SVM)
*   **Best Model Selection:** Automatically selects the best-performing model based on cross-validation accuracy.
*   **Model Persistence:** Saves the trained vectorizer and model to a file, avoiding the need for retraining on subsequent runs.
*   **Interactive GUI:** A user-friendly graphical interface to:
    *   Input and analyze email text.
    *   Display the prediction (phishing or legitimate) with a confidence score.
    *   **Load various email examples** from a dropdown menu to test the model's performance.

## Project Structure

For the script to work correctly, your project must follow this directory structure:

```
your-project-folder/
│
├── assets/
│   └── CEAS_08.csv              # Place your dataset here
│
├── sample_emails/
│   ├── 01_phishing_urgent_account.txt
│   ├── 02_phishing_invoice_scam.txt
│   └── ... (other .txt sample files)
│
├── antifish_v0.1.py             # The main script
└── requirements.txt             # The dependency file
```

## Dependencies

To run this project, you will need Python 3 and the libraries listed in `requirements.txt`.

1.  Create a file named `requirements.txt` in your project directory with the following content:
    ```
    pandas
    scikit-learn
    nltk
    ```2.  Install the dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```
The script will handle the download of necessary NLTK packages (`punkt`, `stopwords`) on its first run.

## How to Use

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Set Up Directories:**
    *   Create a folder named `assets`.
    *   Create a folder named `sample_emails`.
    *   Place your email dataset (e.g., `CEAS_08.csv`) inside the `assets` folder.
    *   Place any `.txt` sample emails you want to test in the `sample_emails` folder.

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Script:**
    ```bash
    python antifish_v0.1.py
    ```

5.  **Training (First Run):**
    *   The script will first check for a pre-trained model at `assets/phishing_detector_model.pkl`.
    *   If not found, it will automatically look for `assets/CEAS_08.csv` to begin the training process.
    *   If that dataset is also not found, it will fall back to asking you for the file path in the console.
    *   The script will preprocess the data, train the models, and save the best one to the `assets` folder.

6.  **Using the GUI:**
    *   Once a model is loaded or trained, the GUI will launch.
    *   Paste email content into the text box and click **Analyze Email**.
    *   To test with pre-saved examples, select one from the **dropdown menu** and click **Load Sample**.

## A Note on Training Time

The training process involves multiple algorithms. The **Support Vector Machine (SVM)** can be computationally intensive and may take a significant amount of time to train, especially on datasets with tens of thousands of emails. If the script seems "stuck" on `Training SVM...`, please be patient as this is normal behavior. For a much faster (though slightly different) implementation, you could modify the script to use `sklearn.svm.LinearSVC` instead of `SVC(kernel='linear')`.
