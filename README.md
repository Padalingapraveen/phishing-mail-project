📧 Phishing Email Detector

A machine learning–based tool to detect phishing emails. Users can paste or upload email text, and the system will analyze it to determine whether the email is Safe or Phishing, along with a confidence score and risk gauge.

🚀 Features

Detects Phishing vs Safe emails using ML.

Supports pasting text or uploading .txt files.

Shows confidence percentage and risk gauge meter.

Lightweight, runs in a browser via Streamlit.

🛠️ Tools & Libraries Used

Python – Programming language

Scikit-Learn – Machine learning model training

Pandas – Data handling and preprocessing

Joblib – Saving/loading ML models

Streamlit – Web application framework

Plotly – Visualizations (risk gauge)

JSON – Label mapping storage

📂 Project Structure
phishing-email-detector/
│── app_streamlit.py        # Streamlit web app
│── model-training.ipynb    # Jupyter notebook (training)
│── phishing_model.pkl      # Trained ML model
│── vectorizer.pkl          # TF-IDF vectorizer
│── label_encoder.pkl       # Label encoder
│── human_label_map.json    # Human-readable label mapping
│── requirements.txt        # Dependencies
│── README.md               # Project documentation

▶️ Run Locally

Clone the repository

git clone https://github.com/yourusername/phishing-email-detector.git
cd phishing-email-detector


Create virtual environment (Windows)

python -m venv venv
venv\Scripts\activate


Install dependencies

pip install -r requirements.txt


Run the app

streamlit run app_streamlit.py
