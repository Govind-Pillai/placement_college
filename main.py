import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# 1. Load data
df = pd.read_csv('data.csv')
df = df.dropna(subset=['Expenses'])

# Define features
numeric_features = ['CGPA', 'IQ', 'Year_of_Experience', 'Dependents', 'Salary']
categorical_features = ['Gender', 'Marital_Status']
target = 'Expenses'

X = df[numeric_features + categorical_features]
y = df[target]

# 2. Manual Preprocessing
# Numerical Imputation
num_imputer = SimpleImputer(strategy='median')
X_num = num_imputer.fit_transform(X[numeric_features])

# Categorical Imputation
cat_imputer = SimpleImputer(strategy='most_frequent')
X_cat = cat_imputer.fit_transform(X[categorical_features])

# Label Encoding
label_encoders = {}
X_cat_encoded = np.zeros(X_cat.shape)
for i, col in enumerate(categorical_features):
    le = LabelEncoder()
    X_cat_encoded[:, i] = le.fit_transform(X_cat[:, i])
    label_encoders[col] = le

# Combine Features
X_combined = np.hstack([X_num, X_cat_encoded])

# 3. Scaling (Normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# 4. Train
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Save all components
model_package = {
    'model': model,
    'num_imputer': num_imputer,
    'cat_imputer': cat_imputer,
    'label_encoders': label_encoders,
    'scaler': scaler,  # <--- Added Scaler
    'numeric_features': numeric_features,
    'categorical_features': categorical_features
}

with open('expenses_model_final.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print("Model with Scaling saved!")