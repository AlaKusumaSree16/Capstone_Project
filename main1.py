import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import webbrowser
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.preprocessing import LabelEncoder
from feature_extraction import extract_features
from train_models import train_all_models

# ================================
# 1. Load Dataset
# ================================
data = pd.read_csv("dataset_final.csv")

print("\n===== DATASET SUMMARY =====")
print("Total Samples (Before Cleaning):", len(data))

# Remove duplicates
data = data.drop_duplicates(subset=["payload"])

# Shuffle dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print("Total Samples (After Cleaning):", len(data))
print("Duplicate payloads remaining:", data["payload"].duplicated().sum())

# Fill Normal class properly
data["attack_type"] = data["attack_type"].fillna("Normal")

print("\nAttack Type Distribution:")
print(data["attack_type"].value_counts())

# ================================
# 2. Multi-Class Encoding
# ================================
le = LabelEncoder()
data["Label"] = le.fit_transform(data["attack_type"])

print("\nEncoded Classes:")
for i, cls in enumerate(le.classes_):
    print(i, "->", cls)

# ================================
# 3. Feature Extraction
# ================================
X = np.array(data["payload"].apply(extract_features).tolist())
y = data["Label"].values

# ================================
# 4. Feature Matrix → HTML
# ================================
feature_names = [
    "payload_length",
    "digit_count",
    "special_char_count",
    "sql_keyword_count",
    "xss_keyword_count",
    "cmd_keyword_count"
]

feature_df = pd.DataFrame(X, columns=feature_names)
feature_df.insert(0, "ID", feature_df.index)

html_matrix = f"""
<html>
<head>
    <title>Feature Matrix</title>
    <style>
        body {{
            font-family: Arial;
            padding: 20px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
        }}
        th, td {{
            border: 1px solid black;
            padding: 5px;
            text-align: center;
        }}
        th {{
            background-color: #f0f0f0;
        }}
    </style>
</head>
<body>
    <h2>Feature Matrix (Numerical Representation)</h2>
    {feature_df.to_html(index=False)}
</body>
</html>
"""

matrix_file = "feature_matrix.html"

with open(matrix_file, "w", encoding="utf-8") as f:
    f.write(html_matrix)

print("Feature matrix saved as feature_matrix.html")

# Auto-open in browser
webbrowser.open("file://" + os.path.realpath(matrix_file))

# ================================
# 5. Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ================================
# 6. Train Models
# ================================
models = train_all_models(X_train, y_train)

# ================================
# 7. Evaluate Models (Multi-Class)
# ================================
results = []

for name, model in models.items():

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    results.append([name, acc, prec, rec, f1])

comparison_df = pd.DataFrame(
    results,
    columns=["Algorithm", "Accuracy", "Precision", "Recall", "F1-score"]
)

print("\n===== FINAL MODEL COMPARISON =====")
print(comparison_df)

# ================================
# 8. Bar Chart
# ================================
comparison_df.set_index("Algorithm").plot(kind="bar")
plt.title("Model Performance Comparison (Multi-Class)")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=30)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

