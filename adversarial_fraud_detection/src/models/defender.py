"""XGBoost fraud detection model (Defender)."""
import os
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib
from typing import Optional, Union


# Default model save path
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, 'defender.pkl')


class DefenderModel:
    """
    XGBoost-based fraud detection model.

    This is the "defender" in our adversarial framework - the model
    that the attacker agent will attempt to evade.
    """

    def __init__(self, **xgb_params):
        """
        Initialize the defender model.

        Args:
            **xgb_params: Override default XGBoost parameters
        """
        default_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'use_label_encoder': False,
            'random_state': 42
        }
        default_params.update(xgb_params)
        self.model = xgb.XGBClassifier(**default_params)
        self._is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> 'DefenderModel':
        """
        Train the fraud detection model.

        Args:
            X: Feature DataFrame
            y: Target Series (0 = legit, 1 = fraud)
            test_size: Fraction for validation set

        Returns:
            self (for method chaining)
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Calculate and store metrics
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        self.val_auc = roc_auc_score(y_val, y_pred_proba)
        self._is_trained = True

        print(f"Training complete. Validation AUC: {self.val_auc:.4f}")
        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Return fraud probability for each transaction.

        Args:
            X: Feature matrix (DataFrame or array)

        Returns:
            Array of fraud probabilities [0-1]
        """
        if not self._is_trained and self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        if isinstance(X, pd.DataFrame):
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict_proba(X)[:, 1]

    def predict(self, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """
        Return binary fraud predictions.

        Args:
            X: Feature matrix
            threshold: Classification threshold

        Returns:
            Array of predictions (0 = legit, 1 = fraud)
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Evaluate model performance.

        Args:
            X: Feature DataFrame
            y: True labels

        Returns:
            Dictionary with AUC and classification metrics
        """
        y_proba = self.predict_proba(X)
        y_pred = self.predict(X)

        return {
            'auc': roc_auc_score(y, y_proba),
            'report': classification_report(y, y_pred, output_dict=True)
        }

    def save(self, path: str = None) -> str:
        """
        Save model to disk.

        Args:
            path: Save path (uses default if not specified)

        Returns:
            Path where model was saved
        """
        if path is None:
            path = DEFAULT_MODEL_PATH

        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
        return path

    def load(self, path: str = None) -> 'DefenderModel':
        """
        Load model from disk.

        Args:
            path: Load path (uses default if not specified)

        Returns:
            self (for method chaining)
        """
        if path is None:
            path = DEFAULT_MODEL_PATH

        self.model = joblib.load(path)
        self._is_trained = True
        print(f"Model loaded from {path}")
        return self

    @property
    def feature_importances(self) -> dict:
        """Get feature importance scores."""
        if not self._is_trained:
            raise RuntimeError("Model not trained yet.")

        return dict(zip(
            self.model.feature_names_in_,
            self.model.feature_importances_
        ))


def train_and_save_model(sample_size: int = None) -> DefenderModel:
    """
    Convenience function to train and save a new model.

    Args:
        sample_size: If set, use sampled data for faster training

    Returns:
        Trained DefenderModel
    """
    # Import here to avoid circular imports
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from data.paysim_download import load_training_data

    print("Loading training data...")
    X, y = load_training_data(sample_size=sample_size)
    print(f"Loaded {len(X)} samples with {y.sum()} fraud cases ({y.mean()*100:.2f}%)")

    print("\nTraining defender model...")
    defender = DefenderModel()
    defender.train(X, y)

    defender.save()
    return defender


if __name__ == '__main__':
    # Train and save model when run directly
    import argparse

    parser = argparse.ArgumentParser(description='Train fraud detection model')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Sample size for quick training (default: use all data)')
    args = parser.parse_args()

    model = train_and_save_model(sample_size=args.sample_size)
    print(f"\nFinal Validation AUC: {model.val_auc:.4f}")
    print("\nFeature Importances:")
    for feat, imp in sorted(model.feature_importances.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")
