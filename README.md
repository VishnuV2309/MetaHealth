

# 🩺 **MetaHealth: Smart Diabetes Predictor**  

🌐 **Live Demo:** 👉 [Click Here to Try MetaHealth](https://metahealth-99.streamlit.app/)  

---

## 👨‍👩‍👧‍👦 **Team MetaMorphs**
💡 *Hackathon Project built with passion and innovation*  
- 👨‍💻 Vishnu V  
- 👨‍💻 Vijay V  
- 👩‍💻 Yukthi Reddy D S  
- 👩‍💻 Tanusrii S  

---

## 📌 **About the Project**  
MetaHealth is an **AI-powered Smart Health Assistant** 🤖 designed to help with **early diabetes risk detection**.  
It leverages **Machine Learning models** trained on the **Pima Indians Diabetes dataset** to provide:  
✅ Instant predictions based on patient data  
✅ CSV batch prediction support  
✅ Intuitive visualization with ROC curves & feature importance  

---

## 📂 **Project Structure**

```bash
METAHEALTH/
│── app.py                 # 🚀 Main Streamlit app
│── requirements.txt       # 📦 Dependencies
│── readme.md              # 📘 Documentation
│
├── assets/                # 🎨 UI assets
│   ├── logo.png
│   └── styles.css
│
├── data/                  # 📊 Dataset
│   └── pima_diabetes.csv
│
├── models/                # 🧠 Models & training
│   ├── train_model.py     # 🔧 Script to train & save models
│   ├── logistic_model.pkl
│   ├── random_forest_model.pkl
│   └── saved/             # 💾 Saved models & artifacts
│       ├── logistic_model.pkl
│       ├── random_forest_model.pkl
│       ├── scaler.pkl
│       ├── metrics.pkl
│       ├── test_data.pkl
│       └── README.txt
│
└── utils/                 # 🛠️ Helper functions
    ├── preprocessing.py   # 🧹 Data cleaning & scaling
    └── evaluation.py      # 📈 Metrics & ROC plotting
````

---

## ⚙️ **Setup Instructions**

### 🔹 1. Clone the Repository


git clone https://github.com/gv-2309/metahealth
cd METAHEALTH


### 🔹 2. Create Virtual Environment & Install Dependencies


python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

pip install -r requirements.txt


### 🔹 3. Train Models (Run Once)

```bash
python models/train_model.py
```

### 🔹 4. Run the App

```bash
streamlit run app.py
```

Or just use the hosted version 👉 [MetaHealth on Streamlit](https://metahealth-99.streamlit.app/)

---

## 🎯 **How It Works**

### 🧍 **Single Prediction**

* Fill in patient details (Glucose, BMI, Age, Pregnancies, etc.)
* Get prediction from:

  * 📊 Logistic Regression
  * 🌲 Random Forest
* See if models **agree/disagree**

### 📑 **Batch Prediction**

* Upload a CSV file with these columns:

  ```
  Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
  ```
* Download results with predictions ✅

### 📈 **Model Evaluation**

* ROC curves & AUC values
* Feature importance visualization

---

## 📊 **Example Results**

* **Single Prediction**

  * Logistic Regression → **72% risk**
  * Random Forest → **68% risk**

* **Batch Prediction**
  Generates predictions for all patients in uploaded CSV file 📂


---

✨ Built with ❤️ by **Team MetaMorphs** for a smarter and healthier future.

