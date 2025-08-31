# predict_cli.py
import joblib
import sys

model = joblib.load('phishing_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

text = " ".join(sys.argv[1:]) if len(sys.argv)>1 else None
if not text:
    print("Usage: python predict_cli.py \"<email text>\"")
    sys.exit(1)

vec = vectorizer.transform([text])
pred = model.predict(vec)[0]
print("Predicted label (encoded):", pred)
if hasattr(model, "predict_proba"):
    prob = model.predict_proba(vec)[0].max()
    print("Confidence:", f"{prob*100:.2f}%")
