# utils/preprocessing.py
import numpy as np
import pandas as pd

COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
           "BMI", "DiabetesPedigreeFunction", "Age"]  # features only; Outcome handled separately

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace impossible zeros in certain medical columns with median of the column.
    Returns a cleaned DataFrame.
    """
    df = df.copy()
    columns_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in columns_with_zeros:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            df[col].fillna(df[col].median(), inplace=True)
    return df

def get_X_y(df: pd.DataFrame):
    """
    Returns X (features DataFrame) and y (target Series). Assumes 'Outcome' is present.
    """
    df = df.copy()
    if "Outcome" not in df.columns:
        raise ValueError("DataFrame must contain 'Outcome' column")
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    return X, y

def single_input_to_df(input_dict: dict):
    """
    Given a dictionary of feature values, return a single-row DataFrame with the correct column order.
    """
    # Ensure required keys exist
    for k in COLUMNS:
        if k not in input_dict:
            raise KeyError(f"Missing feature '{k}' in input")
    df = pd.DataFrame([input_dict], columns=COLUMNS)
    # Clean any zeros that are invalid (in case user inputs 0)
    df = clean_data(df)
    return df
