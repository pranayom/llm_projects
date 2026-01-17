"""SHAP-based model explainability for fraud detection."""
import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List
import warnings

from src.models.defender import DefenderModel
from src.models.transaction import Transaction

# Suppress SHAP warnings
warnings.filterwarnings('ignore', category=UserWarning, module='shap')

# Output directories
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')


class FraudExplainer:
    """
    SHAP-based explainer for fraud detection model.

    Provides interpretable explanations for why transactions
    are flagged as fraudulent, supporting SR 11-7 compliance.
    """

    def __init__(self, defender: DefenderModel, background_data: Optional[pd.DataFrame] = None):
        """
        Initialize the explainer.

        Args:
            defender: Trained fraud detection model
            background_data: Sample data for SHAP baseline (uses synthetic if None)
        """
        self.defender = defender

        # Create background data if not provided
        if background_data is None:
            background_data = self._create_background_data()

        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(
            defender.model,
            data=background_data,
            feature_perturbation='interventional'
        )
        self.feature_names = list(background_data.columns)

        # Ensure output directories exist
        os.makedirs(PLOTS_DIR, exist_ok=True)

    def _create_background_data(self, n_samples: int = 100) -> pd.DataFrame:
        """Create synthetic background data for SHAP baseline."""
        np.random.seed(42)
        return pd.DataFrame({
            'amount': np.random.lognormal(8, 2, n_samples),
            'amount_log': np.random.normal(8, 2, n_samples),
            'hour': np.random.randint(0, 24, n_samples),
            'balance_diff': np.random.normal(5000, 10000, n_samples),
            'velocity_error': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'oldbalanceOrg': np.random.lognormal(10, 2, n_samples)
        })

    def explain_transaction(self, transaction: Transaction) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single transaction.

        Args:
            transaction: Transaction to explain

        Returns:
            Dictionary with SHAP values and feature contributions
        """
        features = transaction.to_features()
        X = pd.DataFrame([features])[self.feature_names]

        # Get SHAP values
        shap_values = self.explainer.shap_values(X)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Class 1 (fraud) SHAP values

        shap_values = shap_values.flatten()

        # Get fraud probability
        fraud_prob = float(self.defender.predict_proba(X)[0])

        # Build explanation
        contributions = {}
        for i, feat in enumerate(self.feature_names):
            contributions[feat] = {
                'value': float(X[feat].iloc[0]),
                'shap_value': float(shap_values[i]),
                'direction': 'increases' if shap_values[i] > 0 else 'decreases'
            }

        # Sort by absolute SHAP value
        sorted_features = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]['shap_value']),
            reverse=True
        )

        return {
            'fraud_probability': fraud_prob,
            'base_value': float(self.explainer.expected_value) if not isinstance(self.explainer.expected_value, list) else float(self.explainer.expected_value[1]),
            'contributions': dict(sorted_features),
            'top_factors': [f[0] for f in sorted_features[:3]]
        }

    def plot_waterfall(
        self,
        transaction: Transaction,
        save_path: Optional[str] = None,
        show: bool = False
    ) -> str:
        """
        Generate SHAP waterfall plot for a transaction.

        Args:
            transaction: Transaction to explain
            save_path: Path to save plot (auto-generated if None)
            show: Display plot interactively

        Returns:
            Path to saved plot
        """
        features = transaction.to_features()
        X = pd.DataFrame([features])[self.feature_names]

        # Get SHAP explanation object
        shap_values = self.explainer(X)

        # Create waterfall plot
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], show=False)
        plt.title('Transaction Fraud Risk Explanation')
        plt.tight_layout()

        # Save plot
        if save_path is None:
            save_path = os.path.join(PLOTS_DIR, f'waterfall_{id(transaction)}.png')

        plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

        return save_path

    def plot_force(
        self,
        transaction: Transaction,
        save_path: Optional[str] = None,
        show: bool = False
    ) -> str:
        """
        Generate SHAP force plot for a transaction.

        Args:
            transaction: Transaction to explain
            save_path: Path to save plot (auto-generated if None)
            show: Display plot interactively

        Returns:
            Path to saved plot
        """
        features = transaction.to_features()
        X = pd.DataFrame([features])[self.feature_names]

        # Get SHAP values
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Create force plot
        base_value = self.explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]

        force_plot = shap.force_plot(
            base_value,
            shap_values[0],
            X.iloc[0],
            matplotlib=True,
            show=False
        )

        # Save plot
        if save_path is None:
            save_path = os.path.join(PLOTS_DIR, f'force_{id(transaction)}.png')

        plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

        return save_path

    def explain_attack(
        self,
        original: Transaction,
        modified: Transaction
    ) -> Dict[str, Any]:
        """
        Explain how an attack changed the model's decision.

        Args:
            original: Original transaction
            modified: Modified (attacked) transaction

        Returns:
            Comparison of explanations
        """
        orig_explanation = self.explain_transaction(original)
        mod_explanation = self.explain_transaction(modified)

        # Calculate changes
        changes = {}
        for feat in self.feature_names:
            orig_val = orig_explanation['contributions'][feat]['shap_value']
            mod_val = mod_explanation['contributions'][feat]['shap_value']
            changes[feat] = {
                'original_shap': orig_val,
                'modified_shap': mod_val,
                'change': mod_val - orig_val
            }

        return {
            'original': orig_explanation,
            'modified': mod_explanation,
            'probability_change': mod_explanation['fraud_probability'] - orig_explanation['fraud_probability'],
            'feature_changes': changes,
            'most_exploited': max(changes.items(), key=lambda x: abs(x[1]['change']))[0]
        }

    def generate_summary_report(self, explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate aggregate summary of multiple explanations.

        Args:
            explanations: List of individual explanations

        Returns:
            Summary statistics and patterns
        """
        if not explanations:
            return {'error': 'No explanations provided'}

        # Aggregate SHAP values
        feature_importance = {feat: [] for feat in self.feature_names}
        fraud_probs = []

        for exp in explanations:
            fraud_probs.append(exp['fraud_probability'])
            for feat, contrib in exp['contributions'].items():
                feature_importance[feat].append(abs(contrib['shap_value']))

        # Calculate averages
        avg_importance = {
            feat: np.mean(vals) for feat, vals in feature_importance.items()
        }

        return {
            'n_transactions': len(explanations),
            'avg_fraud_probability': np.mean(fraud_probs),
            'avg_feature_importance': dict(sorted(
                avg_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )),
            'top_3_features': sorted(avg_importance.items(), key=lambda x: -x[1])[:3]
        }


if __name__ == '__main__':
    print("Testing FraudExplainer...")

    # Load model
    defender = DefenderModel()
    defender.load()

    # Create explainer
    explainer = FraudExplainer(defender)

    # Test transaction
    txn = Transaction(
        amount=50000.0,
        hour=3,
        oldbalanceOrg=100000.0,
        newbalanceOrig=50000.0,
        velocity_error=1
    )

    print(f"\nTest transaction: {txn.model_dump()}")

    # Get explanation
    explanation = explainer.explain_transaction(txn)
    print(f"\nFraud probability: {explanation['fraud_probability']:.4f}")
    print(f"Top factors: {explanation['top_factors']}")
    print("\nFeature contributions:")
    for feat, contrib in explanation['contributions'].items():
        print(f"  {feat}: {contrib['shap_value']:.4f} ({contrib['direction']} fraud risk)")

    # Generate plot
    plot_path = explainer.plot_waterfall(txn)
    print(f"\nWaterfall plot saved to: {plot_path}")
