import os
import sys
import json
import joblib
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def clean_dataframe(df, config):
    """Apply deterministic dataset cleaning shared by training and inference."""
    df = df.copy()
    target_column = config["data"]["target_column"]
    id_column = config["data"]["id_column"]

    if id_column in df.columns:
        df = df.drop(columns=[id_column])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if target_column in df.columns:
        df[target_column] = df[target_column].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0).astype(int)

    return df


def create_initial_test_ids(df, config, processed_dir):
    """Recreate the original test split and permanently save its customer IDs."""
    id_column = config["data"]["id_column"]
    target_column = config["data"]["target_column"]
    test_ids_path = os.path.join(processed_dir, "test_customer_ids.csv")

    if os.path.exists(test_ids_path):
        test_ids = pd.read_csv(test_ids_path)

        if id_column not in test_ids.columns:
            raise ValueError(f"{test_ids_path} does not contain {id_column}.")

        return test_ids[id_column].astype(str).tolist()

    print("[INFO] test_customer_ids.csv not found.")
    print("[INFO] Recreating the original train/test split to recover test customer IDs...")

    working_df = df.copy()
    working_df[id_column] = working_df[id_column].astype(str)
    working_df["TotalCharges"] = pd.to_numeric(working_df["TotalCharges"], errors="coerce")
    working_df[target_column] = working_df[target_column].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0).astype(int)

    X = working_df.drop(columns=[id_column, target_column])
    y = working_df[target_column]

    _, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=y,
    )

    test_ids = working_df.loc[X_test.index, id_column].astype(str).tolist()

    pd.DataFrame({id_column: test_ids}).to_csv(test_ids_path, index=False)

    print(f"[SUCCESS] Original test-set IDs saved to {test_ids_path}")
    print(f"[INFO] Fixed test set size: {len(test_ids)}")

    return test_ids


def split_raw_data(raw_df, config, processed_dir):
    id_column = config.get("data", {}).get("id_column", raw_df.columns[0])
    target_column = config.get("data", {}).get("target_column", "churn")
    
    # Ensure case-insensitive match for target and id columns if needed
    col_map = {c.lower(): c for c in raw_df.columns}
    if target_column.lower() in col_map:
        target_column = col_map[target_column.lower()]
    else:
        # Fallback to the last column if target not found
        target_column = raw_df.columns[-1]

    if id_column.lower() in col_map:
        id_column = col_map[id_column.lower()]

    test_ids_path = os.path.join(processed_dir, "test_customer_ids.csv")
    test_ids_df = pd.read_csv(test_ids_path)
    
    test_id_col = test_ids_df.columns[0]
    if test_id_col != id_column:
        test_ids_df.rename(columns={test_id_col: id_column}, inplace=True)

    if target_column not in test_ids_df.columns:
        test_ids_df = test_ids_df.merge(raw_df[[id_column, target_column]], on=id_column, how="inner")
        
    test_ids = set(test_ids_df[id_column].astype(str))
    train_df = raw_df[~raw_df[id_column].astype(str).isin(test_ids)]
    test_df = raw_df[raw_df[id_column].astype(str).isin(test_ids)]
    
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column].apply(lambda x: 1 if str(x).strip().lower() in ["yes", "1", "true"] else 0).astype(int)
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column].apply(lambda x: 1 if str(x).strip().lower() in ["yes", "1", "true"] else 0).astype(int)
    return X_train, X_test, y_train, y_test


def build_preprocessor(X_train):
    """Create the single canonical feature-preprocessing pipeline used by ML Pipeline."""
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    ), cat_cols, num_cols


def run_preprocessing():
    print("[INFO] Starting Preprocessing...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = config["data"]["raw_path"]
    processed_dir = config["data"]["processed_dir"]
    target_column = config["data"]["target_column"]

    if not os.path.exists(data_path):
        print(f"[ERROR] Raw dataset not found at: {data_path}")
        return False

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    raw_df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = split_raw_data(raw_df, config, processed_dir)

    preprocessor, cat_cols, num_cols = build_preprocessor(X_train)

    print("[INFO] Fitting preprocessing pipeline on training data...")
    X_train_final = preprocessor.fit_transform(X_train)
    X_test_final = preprocessor.transform(X_test)

    np.save(os.path.join(processed_dir, "X_train_final.npy"), X_train_final)
    np.save(os.path.join(processed_dir, "X_test_final.npy"), X_test_final)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train.to_numpy(dtype=np.int64))
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test.to_numpy(dtype=np.int64))

    X_train.to_csv(os.path.join(processed_dir, "X_train_raw.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test_raw.csv"), index=False)

    joblib.dump(preprocessor, config["paths"]["preprocessor"])

    metadata = {
        "dataset_name": "Telco Customer Churn",
        "target_column": target_column,
        "train_shape_raw": list(X_train.shape),
        "test_shape_raw": list(X_test.shape),
        "train_shape_transformed": list(X_train_final.shape),
        "test_shape_transformed": list(X_test_final.shape),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "split_test_size": config["split"]["test_size"],
        "split_random_state": config["split"]["random_state"],
        "fixed_test_set": True,
        "test_customer_ids_path": os.path.join(processed_dir, "test_customer_ids.csv"),
        "raw_train_path": os.path.join(processed_dir, "X_train_raw.csv"),
        "raw_test_path": os.path.join(processed_dir, "X_test_raw.csv"),
    }

    with open(os.path.join(processed_dir, "dataset_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[SUCCESS] Preprocessing complete. Preprocessor saved to {config['paths']['preprocessor']}")
    return True


if __name__ == "__main__":
    passed = run_preprocessing()
    sys.exit(0 if passed else 1)