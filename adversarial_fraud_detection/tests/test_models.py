"""Tests for Transaction and Defender models."""
import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.transaction import Transaction, TransactionPerturbation
from src.models.defender import DefenderModel


class TestTransaction:
    """Tests for Transaction data model."""

    def test_valid_transaction_creates(self):
        """Test that valid transactions are created successfully."""
        txn = Transaction(
            amount=1000.0,
            oldbalanceOrg=5000.0,
            newbalanceOrig=4000.0,
            hour=12,
            velocity_error=0
        )
        assert txn.amount == 1000.0
        assert txn.hour == 12

    def test_invalid_amount_raises(self):
        """Test that zero/negative amount raises error."""
        with pytest.raises(ValueError):
            Transaction(
                amount=0,
                oldbalanceOrg=5000.0,
                newbalanceOrig=5000.0,
                hour=12,
                velocity_error=0
            )

    def test_invalid_hour_raises(self):
        """Test that hour outside 0-23 raises error."""
        with pytest.raises(ValueError):
            Transaction(
                amount=100.0,
                oldbalanceOrg=5000.0,
                newbalanceOrig=4900.0,
                hour=25,  # Invalid
                velocity_error=0
            )

    def test_balance_constraint_validates(self):
        """Test balance must be within 10% of expected."""
        # This should fail - newbalanceOrig is way off
        with pytest.raises(ValueError, match="Balance constraint"):
            Transaction(
                amount=100.0,
                oldbalanceOrg=5000.0,
                newbalanceOrig=1000.0,  # Should be ~4900
                hour=12,
                velocity_error=0
            )

    def test_to_features_returns_correct_dict(self):
        """Test to_features returns all required fields."""
        txn = Transaction(
            amount=1000.0,
            oldbalanceOrg=5000.0,
            newbalanceOrig=4000.0,
            hour=12,
            velocity_error=0
        )
        features = txn.to_features()

        assert 'amount' in features
        assert 'amount_log' in features
        assert 'hour' in features
        assert 'balance_diff' in features
        assert 'velocity_error' in features
        assert 'oldbalanceOrg' in features

        assert features['amount'] == 1000.0
        assert np.isclose(features['amount_log'], np.log1p(1000.0))
        assert features['balance_diff'] == 1000.0  # 5000 - 4000


class TestTransactionPerturbation:
    """Tests for perturbation constraints."""

    def test_valid_perturbation_passes(self):
        """Test perturbation within bounds is valid."""
        original = Transaction(
            amount=1000.0,
            oldbalanceOrg=5000.0,
            newbalanceOrig=4000.0,
            hour=12,
            velocity_error=0
        )
        modified = Transaction(
            amount=1050.0,  # 5% change (within 10%)
            oldbalanceOrg=5000.0,
            newbalanceOrig=3950.0,
            hour=14,  # 2 hour shift (within 2)
            velocity_error=0
        )

        constraints = TransactionPerturbation()
        assert constraints.is_valid_perturbation(original, modified)

    def test_amount_violation_detected(self):
        """Test amount change >10% is detected."""
        original = Transaction(
            amount=1000.0,
            oldbalanceOrg=5000.0,
            newbalanceOrig=4000.0,
            hour=12,
            velocity_error=0
        )
        modified = Transaction(
            amount=850.0,  # 15% change (exceeds 10%)
            oldbalanceOrg=5000.0,
            newbalanceOrig=4150.0,
            hour=12,
            velocity_error=0
        )

        constraints = TransactionPerturbation()
        assert not constraints.is_valid_perturbation(original, modified)

    def test_hour_violation_detected(self):
        """Test hour shift >2 is detected."""
        original = Transaction(
            amount=1000.0,
            oldbalanceOrg=5000.0,
            newbalanceOrig=4000.0,
            hour=12,
            velocity_error=0
        )
        modified = Transaction(
            amount=1000.0,
            oldbalanceOrg=5000.0,
            newbalanceOrig=4000.0,
            hour=16,  # 4 hour shift (exceeds 2)
            velocity_error=0
        )

        constraints = TransactionPerturbation()
        assert not constraints.is_valid_perturbation(original, modified)


class TestDefenderModel:
    """Tests for the XGBoost fraud detection model."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 1000

        # Generate synthetic features
        X = pd.DataFrame({
            'amount': np.random.exponential(500, n_samples),
            'amount_log': np.log1p(np.random.exponential(500, n_samples)),
            'hour': np.random.randint(0, 24, n_samples),
            'balance_diff': np.random.normal(100, 50, n_samples),
            'velocity_error': np.random.binomial(1, 0.1, n_samples),
            'oldbalanceOrg': np.random.exponential(5000, n_samples)
        })

        # Generate target (correlated with features for realistic behavior)
        fraud_proba = 0.1 + 0.3 * X['velocity_error'] + 0.001 * X['amount'] / 1000
        y = pd.Series((np.random.random(n_samples) < fraud_proba).astype(int))

        return X, y

    def test_defender_trains(self, sample_data):
        """Test model trains without error."""
        X, y = sample_data
        defender = DefenderModel()
        defender.train(X, y)

        assert defender._is_trained
        assert hasattr(defender, 'val_auc')
        assert 0 <= defender.val_auc <= 1

    def test_predict_proba_returns_valid_scores(self, sample_data):
        """Test predictions are valid probabilities."""
        X, y = sample_data
        defender = DefenderModel().train(X, y)

        scores = defender.predict_proba(X[:10])

        assert len(scores) == 10
        assert all(0 <= s <= 1 for s in scores)

    def test_predict_returns_binary(self, sample_data):
        """Test predict returns 0/1 values."""
        X, y = sample_data
        defender = DefenderModel().train(X, y)

        predictions = defender.predict(X[:10])

        assert len(predictions) == 10
        assert set(predictions).issubset({0, 1})

    def test_save_and_load(self, sample_data, tmp_path):
        """Test model can be saved and loaded."""
        X, y = sample_data
        defender = DefenderModel().train(X, y)
        original_scores = defender.predict_proba(X[:5])

        # Save
        save_path = str(tmp_path / "test_model.pkl")
        defender.save(save_path)

        # Load into new instance
        loaded_defender = DefenderModel().load(save_path)
        loaded_scores = loaded_defender.predict_proba(X[:5])

        np.testing.assert_array_almost_equal(original_scores, loaded_scores)
