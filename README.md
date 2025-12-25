# 📉 Expense AI Predictor

A professional Machine Learning application that predicts monthly expenses based on academic, professional, and personal data. This project uses a Multiple Linear Regression (MLR) model with manual preprocessing, including Null-value imputation, Label Encoding, and Feature Scaling.

## 🚀 Features
* **Full ML Pipeline**: Includes data cleaning, median imputation for missing values, and standard scaling.
* **Null-Safe Prediction**: The app handles missing inputs gracefully using pre-trained logic.
* **Beautiful UI**: Built with Streamlit, featuring a modern dark-mode "Glassmorphism" interface.
* **Normalized Model**: Uses `StandardScaler` to ensure all features contribute fairly to the prediction.



## Tech Stack
* **Language:** Python 3.x
* **Frontend:** Streamlit
* **ML Library:** Scikit-Learn
* **Data Handling:** Pandas, Numpy
* **Model Persistence:** Pickle

## How to Run
Clone the repository:

``` Bash

git clone https://github.com/Govind-Pillai/placement_college.git
cd placement_college
```
Install dependencies:

```Bash

pip install -r requirements.txt
```
Train the model (Optional - model is already provided):

```Bash

python main.py
```
Run the Application:

```Bash

python -m streamlit run ui_app.py
```

## Project Structure
```text
├── app.py                     # Streamlit web application
├── train.py                   # Model training and preprocessing script
├── data.csv                   # Dataset used for training
├── expenses_model_final.pkl   # Packaged model, scaler, and imputers
├── requirements.txt           # List of dependencies
└── README.md                  # Project documentation
```

🧠 Model Details

``` Text
The model is a Multiple Linear Regression algorithm.

Features Used: CGPA, IQ, Year of Experience, Dependents, Salary, Gender, and Marital Status.

Target Variable: Expenses.

Preprocessing:

Numerical: SimpleImputer(strategy='median')

Categorical: LabelEncoder

Scaling: StandardScaler
```

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.
