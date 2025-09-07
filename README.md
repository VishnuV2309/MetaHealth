
```markdown
# MetaHealth: Smart Diabetes Predictor 🩺⚡

MetaHealth is a **Streamlit-based web application** that predicts the likelihood of diabetes in women based on the **Pima Indians Diabetes dataset**.  
It allows both **model comparison** and **individual patient predictions** using **Logistic Regression** and **Random Forest** models.

---

## 🚀 Features
- 📊 **Model Comparison**  
  Upload test CSV files and compare Logistic Regression vs Random Forest with:
  - Classification Report
  - Confusion Matrix
  - ROC Curve with AUC

- 🧑‍⚕️ **User Input Prediction**  
  Enter patient details manually and get:
  - Probability of diabetes
  - Risk category (High Risk / No Diabetes)

- 📂 **Reference Test Files**  
  Comes with **5 pre-sampled test case CSV files** (`testcase1.csv` ... `testcase5.csv`)  
  to validate model predictions.

---

## 🏗️ Project Structure
```

MetaHealth/
│── app.py                          # Main entry point (Streamlit app)
│── 1\_Model\_Comparison.py           # Page for model comparison
│── 2\_User\_Input\_Prediction.py      # Page for user input predictions
│── utils/
│   ├── preprocessing.py             # Data cleaning & preprocessing
│   ├── evaluation.py                # Model evaluation helpers
│   ├── inference.py                 # Model loading & prediction functions
│── Test\_Case\_CSV\_Files/
│   ├── testcase1.csv
│   ├── testcase2.csv
│   ├── testcase3.csv
│   ├── testcase4.csv
│   ├── testcase5.csv
│── requirements.txt                 # Dependencies
│── README.md                        # Project documentation

````

---

## ⚙️ Installation & Setup

1. **Clone repository**
   ```bash
   git clone https://github.com/your-username/MetaHealth.git
   cd MetaHealth
````

2. **Create virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate   # for Linux/Mac
   venv\Scripts\activate      # for Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run Streamlit app**

   ```bash
   streamlit run app.py
   ```

---

## 📊 Dataset

We use the **Pima Indians Diabetes Dataset** from [UCI Machine Learning Repository](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv).

Features:

* Pregnancies
* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI
* DiabetesPedigreeFunction
* Age
* Outcome (0 = No Diabetes, 1 = Diabetes)

---

## 👨‍💻 Team MetaMorphs

* Vishnu V
* Vijay V
* Yukthi Reddy D S
* Tanusrii S

---

## 🌐 Deployment

The app can be deployed on:

* [Streamlit Cloud](https://streamlit.io/cloud)
* Heroku
* Azure / AWS / GCP

*(Instructions for deployment can be added based on platform chosen)*

---

## 📜 License

This project is for educational and research purposes.

```



👉 Do you want me to also **add screenshots / demo GIF** section in the README so your hackathon/demo submission looks more attractive?
```
