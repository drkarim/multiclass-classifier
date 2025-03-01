# Digits Classifier with Logistic Regression

## 📌 Overview

This project trains a **Logistic Regression Classifier** on the **Digits Dataset**, evaluates it using **ROC-AUC**, determines an **optimal threshold**, and saves the trained model, scaler, and thresholds for future inference.

## 🏗️ Steps Involved

1️⃣ **Train a classifier on the Digits dataset** using Logistic Regression (Softmax for multiclass classification).\
2️⃣ **Check ROC-AUC** curves to evaluate performance per class.\
3️⃣ **Choose the optimal threshold** using **Youden’s J-statistic**.\
4️⃣ **Save the Model, Scaler, and Class Thresholds persistently** in `.pkl` and `.json` files.\
5️⃣ **Load the trained model & thresholds for inference** to classify new digit samples.

## 🚀 Getting Started

### **1️⃣ Clone the Repository**

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/digits-classifier.git
cd digits-classifier
```

### **2️⃣ Install Dependencies**

Ensure you have Python 3.8+ and install required libraries:

```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Notebook**

Launch Jupyter Notebook and open `digits_classifier_notebook.ipynb`:

```bash
jupyter notebook
```

Run the cells in order to train, evaluate, and save the model.

### **4️⃣ Load and Use the Model for Inference**

After training and saving the model, you can use the saved artifacts (`.pkl` and `.json` files) to classify new samples.

To test inference, run:

```python
import joblib
import json
import numpy as np

# Load model, scaler, and thresholds
model = joblib.load("logistic_regression_multiclass.pkl")
scaler = joblib.load("scaler.pkl")
with open("optimal_thresholds.json", "r") as f:
    thresholds = json.load(f)

# Sample new input (assuming 64 features, reshape as needed)
X_new = np.random.rand(1, 64)  # Replace with real data
X_new_scaled = scaler.transform(X_new)
y_proba = model.predict_proba(X_new_scaled)

# Apply optimal thresholds
y_pred_adjusted = np.zeros_like(y_proba)
for i in range(len(thresholds)):
    y_pred_adjusted[:, i] = (y_proba[:, i] >= thresholds[str(i)]).astype(int)

final_prediction = np.argmax(y_pred_adjusted, axis=1)
print("Predicted Class:", final_prediction)
```

## 📂 Files in This Repository

- `digits_classifier_notebook` → Main Jupyter Notebook for training, evaluating, and saving the model.
- `logistic_regression_multiclass.pkl` → Saved trained logistic regression model.
- `scaler.pkl` → Saved standard scaler for preprocessing test data.
- `optimal_thresholds.json` → Saved per-class optimal thresholds.
- `requirements.txt` → List of dependencies required to run the project.

## 📜 License

This project is open-source under the **MIT License**. Feel free to use and contribute!

---

Made with ❤️ by [Rashed Karim]

