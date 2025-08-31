# train_model.py
import argparse
import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def find_text_label_columns(df):
    text_cols = [c for c in df.columns if any(k in c.lower() for k in ('text','email','body','message'))]
    label_cols = [c for c in df.columns if any(k in c.lower() for k in ('label','target','class'))]
    return text_cols, label_cols

def load_from_csv(path):
    df = pd.read_csv(path)
    text_cols, label_cols = find_text_label_columns(df)
    if not text_cols:
        raise ValueError(f"No text-like column found. Columns: {df.columns.tolist()}")
    if not label_cols:
        raise ValueError(f"No label-like column found. Columns: {df.columns.tolist()}")
    X = df[text_cols[0]].astype(str).fillna('')
    y = df[label_cols[0]]
    return X, y

def load_hf(dataset_id):
    from datasets import load_dataset
    ds = load_dataset(dataset_id)
    split = 'train' if 'train' in ds else list(ds.keys())[0]
    df = pd.DataFrame(ds[split])
    return load_from_dataframe(df)

def load_from_dataframe(df):
    text_cols, label_cols = find_text_label_columns(df)
    if not text_cols or not label_cols:
        raise ValueError("Couldn't find text/label columns in DataFrame. Columns: " + ", ".join(df.columns))
    X = df[text_cols[0]].astype(str).fillna('')
    y = df[label_cols[0]]
    return X, y

def build_human_label_map(y_unique):
    human_map = {}
    found_phish_label = None
    for val in y_unique:
        sval = str(val).lower()
        if 'phish' in sval or 'malicious' in sval or 'spam' in sval:
            found_phish_label = val
            break
    if found_phish_label is not None:
        for val in y_unique:
            if val == found_phish_label:
                human_map[val] = "Phishing"
            else:
                human_map[val] = "Safe"
    else:
        if set(map(str, y_unique)).issubset({'0','1'}):
            for val in y_unique:
                human_map[val] = "Phishing" if str(val)=='1' else "Safe"
        else:
            counts = pd.Series(y_unique).value_counts()
            if len(counts) == 2:
                majority = counts.index[0]
                for val in y_unique:
                    human_map[val] = "Safe" if val==majority else "Phishing"
            else:
                for i,val in enumerate(sorted(y_unique)):
                    human_map[val] = "Phishing" if i==1 else "Safe"
    return human_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", default="phishing_email_dataset.csv")
    parser.add_argument("--hf_dataset", default=None)
    parser.add_argument("--model_out", default="phishing_model.pkl")
    parser.add_argument("--vectorizer_out", default="vectorizer.pkl")
    parser.add_argument("--label_encoder_out", default="label_encoder.pkl")
    parser.add_argument("--human_map_out", default="human_label_map.json")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    if args.hf_dataset:
        print("Loading HF dataset:", args.hf_dataset)
        X, y = load_hf(args.hf_dataset)
    else:
        if not os.path.exists(args.data_csv):
            raise FileNotFoundError(f"CSV not found at {args.data_csv}. Place your dataset there.")
        print("Loading CSV:", args.data_csv)
        X, y = load_from_csv(args.data_csv)

    mask = X.astype(bool)
    X = X[mask]
    y = y[mask]

    unique_vals = pd.Series(y.unique()).tolist()
    human_map = build_human_label_map(unique_vals)
    print("Human label map (original_label -> Friendly):", human_map)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print("Label classes (LabelEncoder.classes_):", le.classes_.tolist())

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=args.test_size, random_state=42, stratify=y_encoded)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=args.test_size, random_state=42)

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')
    X_train_t = vectorizer.fit_transform(X_train)
    X_test_t = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_t, y_train)

    preds = model.predict(X_test_t)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, target_names=[str(c) for c in le.classes_]))

    joblib.dump(model, args.model_out)
    joblib.dump(vectorizer, args.vectorizer_out)
    joblib.dump(le, args.label_encoder_out)
    encoded_classes = le.transform(le.classes_)
    encoded_human_map = {}
    for enc_val, orig in zip(encoded_classes, le.classes_):
        encoded_human_map[int(enc_val)] = human_map[orig]
    with open(args.human_map_out, "w") as f:
        json.dump(encoded_human_map, f, indent=2)

    print("Saved model:", args.model_out)
    print("Saved vectorizer:", args.vectorizer_out)
    print("Saved label encoder:", args.label_encoder_out)
    print("Saved human label map:", args.human_map_out)

if __name__ == "__main__":
    main()
