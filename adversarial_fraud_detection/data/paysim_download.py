"""PaySim dataset loader and preprocessor."""
import os
import pandas as pd
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi


# Features to use for the model
FEATURE_COLS = ['amount', 'amount_log', 'hour', 'balance_diff', 'velocity_error', 'oldbalanceOrg']
TARGET_COL = 'isFraud'

# Path configuration
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_NAME = 'mtalaltariq/paysim-data'
CSV_FILENAME = 'paysim dataset.csv'


def download_paysim(force_download: bool = False) -> pd.DataFrame:
    """
    Download PaySim dataset from Kaggle.

    Args:
        force_download: If True, re-download even if file exists

    Returns:
        Raw DataFrame from PaySim dataset
    """
    csv_path = os.path.join(DATA_DIR, CSV_FILENAME)

    if not force_download and os.path.exists(csv_path):
        print(f"Loading existing dataset from {csv_path}")
        return pd.read_csv(csv_path)

    print("Downloading PaySim dataset from Kaggle...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET_NAME, path=DATA_DIR, unzip=True)

    print(f"Dataset downloaded to {DATA_DIR}")
    return pd.read_csv(csv_path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for fraud detection.

    Args:
        df: Raw PaySim DataFrame

    Returns:
        DataFrame with engineered features
    """
    df = df.copy()

    # Time-based feature: hour of day (step is hourly)
    df['hour'] = df['step'] % 24

    # Amount transformation for better distribution
    df['amount_log'] = np.log1p(df['amount'])

    # Balance difference (potential anomaly indicator)
    df['balance_diff'] = df['oldbalanceOrg'] - df['newbalanceOrig']

    # Velocity error: when balance equations don't add up (fraud indicator)
    df['velocity_error'] = (
        df['oldbalanceOrg'] - df['amount'] != df['newbalanceOrig']
    ).astype(int)

    return df


def load_training_data(sample_size: int = None) -> tuple:
    """
    Load and preprocess PaySim data for model training.

    Args:
        sample_size: If set, sample this many rows (useful for quick testing)

    Returns:
        Tuple of (X, y) where X is features DataFrame and y is target Series
    """
    df = download_paysim()
    df = preprocess(df)

    if sample_size is not None:
        # Stratified sampling to maintain fraud ratio
        fraud = df[df[TARGET_COL] == 1]
        legit = df[df[TARGET_COL] == 0]

        fraud_ratio = len(fraud) / len(df)
        fraud_sample = min(int(sample_size * fraud_ratio), len(fraud))
        legit_sample = sample_size - fraud_sample

        df = pd.concat([
            fraud.sample(n=fraud_sample, random_state=42),
            legit.sample(n=legit_sample, random_state=42)
        ]).sample(frac=1, random_state=42)  # Shuffle

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    return X, y


if __name__ == '__main__':
    # Quick test of data loading
    print("Testing PaySim data loader...")
    X, y = load_training_data(sample_size=10000)
    print(f"Loaded {len(X)} samples")
    print(f"Fraud rate: {y.mean():.4f}")
    print(f"Features: {list(X.columns)}")
    print("\nSample data:")
    print(X.head())
