import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def load_and_process_data(file_path_or_df, is_prediction=False):  # ✅ Add `is_prediction`
    """ Load, clean, encode, and split data """
    
    if isinstance(file_path_or_df, str):
        df = pd.read_csv(file_path_or_df)
    else:
        df = file_path_or_df.copy()

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    drop_columns = ['unnamed:_0', 'id']
    df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors='ignore')

    print("✅ Available columns after cleaning:", list(df.columns))

    if not is_prediction:  # ✅ Only check for 'satisfaction' if training
        if "satisfaction" not in df.columns:
            raise KeyError(f"❌ Target column 'satisfaction' not found! Available columns: {list(df.columns)}")

        print("✅ Using 'satisfaction' as the target column.")

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    num_imputer = SimpleImputer(strategy="median")
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    df = df.apply(pd.to_numeric, errors='coerce')

    if not is_prediction:
        X = df.drop(columns=['satisfaction'])
        y = df['satisfaction']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, scaler, label_encoders
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)  # ✅ Preprocess input without 'satisfaction'
        return X_scaled
