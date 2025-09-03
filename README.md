# Phishing mail decetor


A machine learning–based tool to detect phishing emails. Users can paste or upload email text, and the system will analyze it to determine whether the email is Safe or Phishing, along with a confidence score and risk gauge.

🚀Features

Detects Phishing vs Safe emails using ML. 

Supports pasting text or uploading .txt files. 

Shows confidence percentage and risk gauge meter. 

Lightweight, runs in a browser via Streamlit.
## Installation

Run Locally

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


    
## Tools & Libraries

Python – Programming language

Scikit-Learn – Machine learning model training

Pandas – Data handling and preprocessing

Joblib – Saving/loading ML models

Streamlit – Web application framework

Plotly – Visualizations (risk gauge)

JSON – Label mapping storage
## Deployment

To deploy this project run

Push repo to GitHub.

Go to Streamlit Cloud.

Connect repo → Select app_streamlit.py → Deploy.
## Future Work
Add support for URL/attachment scanning.

Provide detailed PDF reports.

Extend to browser plugins or email client integration
