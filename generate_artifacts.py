import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

print("Generating model artifacts...")
csv_url = 'https://raw.githubusercontent.com/bianca255/linear_regression/main/linear_regression_model/summative/linear_regression/student-mat.csv'
df = pd.read_csv(csv_url, sep=';')
print(f'✓ Loaded dataset: {df.shape}')

target = 'G3'
X = df.drop(columns=[target]).copy()
y = df[target].copy()

for col in X.columns:
    if X[col].dtype == 'object':
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col].astype(str))

FEATURE_COLUMNS = ['sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'paid', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'goout', 'Dalc', 'Walc', 'health', 'G1', 'G2']
X = X.reindex(columns=FEATURE_COLUMNS, fill_value=0)

# Debug: print dtypes
print('Column dtypes after reindex:')
for col in X.columns:
    print(f'  {col}: {X[col].dtype}')

# Encode again for any columns still object type
for col in X.columns:
    if X[col].dtype == 'object':
        print(f'    Encoding {col}')
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])

# Ensure all numeric
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
print(f'✓ Prepared {X.shape[1]} features')
print(f'✓ Prepared {X.shape[1]} features')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(max_depth=6, random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
}

best_model = None
best_mse = None
best_name = None

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print(f'  {name}: MSE = {mse:.4f}')
    if best_mse is None or mse < best_mse:
        best_mse = mse
        best_model = model
        best_name = name

print(f'✓ Best model: {best_name} (MSE={best_mse:.4f})')

api_dir = 'linear_regression_model/summative/API'
os.makedirs(api_dir, exist_ok=True)

joblib.dump(best_model, os.path.join(api_dir, 'best_model.pkl'))
joblib.dump(scaler, os.path.join(api_dir, 'scaler.pkl'))
with open(os.path.join(api_dir, 'feature_columns.json'), 'w') as f:
    json.dump(FEATURE_COLUMNS, f, indent=2)

print('✓ Saved best_model.pkl')
print('✓ Saved scaler.pkl')
print('✓ Saved feature_columns.json')
print('✅ All artifacts generated successfully!')
