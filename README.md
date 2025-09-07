
---

````markdown
# 🩺 MetaHealth: Smart Diabetes Predictor

AI-powered web app for early **diabetes risk detection** using Logistic Regression and Random Forest models trained on the **Pima Indians Diabetes dataset**.  

Built with **Streamlit**, this app allows both **single-patient predictions** and **batch predictions (CSV upload)**.  
It also provides **model evaluation metrics** such as ROC curves and feature importance.  

---
## 🌐 Live Demo

👉 [Hosted App Link][https://metahealth-99.streamlit.app/]

---

## 📌 Features
- 🔹 Predict diabetes risk using **two models**:
  - Logistic Regression
  - Random Forest
- 🔹 Input patient details through an **interactive sidebar form**
- 🔹 Upload a CSV file for **batch predictions**
- 🔹 View **ROC curves, AUC scores, and feature importance**
- 🔹 Clean UI with team credits and footer section

---

## 📂 Project Structure

```bash
METAHEALTH/
│── app.py                 # Main Streamlit app
│── requirements.txt       # Dependencies
│── readme.md              # Project documentation
│
├── assets/                # UI assets
│   ├── logo.png
│   └── styles.css
│
├── data/                  # Dataset
│   └── pima_diabetes.csv
│
├── models/                # Models & training
│   ├── train_model.py     # Script to train & save models
│   ├── logistic_model.pkl
│   ├── random_forest_model.pkl
│   └── saved/             # Saved models & artifacts
│       ├── logistic_model.pkl
│       ├── random_forest_model.pkl
│       ├── scaler.pkl
│       ├── metrics.pkl
│       ├── test_data.pkl
│       └── README.txt
│
└── utils/                 # Helper functions
    ├── preprocessing.py   # Data cleaning & scaling
    └── evaluation.py      # Metrics, ROC curve plotting
````

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/METAHEALTH.git
cd METAHEALTH
```

### 2️⃣ Create Virtual Environment & Install Dependencies

```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

pip install -r requirements.txt
```

### 3️⃣ Train Models (Run Once)

This step generates the pre-trained models inside `models/saved/`.

```bash
python models/train_model.py
```

### 4️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📊 Usage

### 🔹 Single Prediction

* Enter details like **Glucose, BMI, Age, Pregnancies** etc. in the **sidebar form**
* Get predictions from **Logistic Regression** and **Random Forest** models
* View if the models **agree/disagree**

### 🔹 Batch Prediction

* Upload a **CSV file** with these exact column names:

  ```
  Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
  ```
* Download predictions as a **CSV**

### 🔹 Model Evaluation

* View **ROC Curves**, **AUC scores**, and **Feature Importance**

---

## 📈 Example Output

* **Single Prediction**
  Logistic Regression → 72% risk
  Random Forest → 68% risk

* **Batch Prediction**
  Generates predictions for all patients in uploaded CSV

---

## 👨‍💻 Contributors

Developed by **Team MetaMorphs**

* Vishnu V
* Vijay V
* Yukthi Reddy D S
* Tanusrii S

---



## 📝 License

This project is licensed under the **MIT License**.

```

---

⚡ Do you want me to also include **sample CSV files links** in the README (so users can directly test batch prediction)?
```
