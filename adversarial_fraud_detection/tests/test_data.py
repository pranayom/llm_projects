"""Tests for data loading and preprocessing."""
import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.paysim_download import (
    download_paysim,
    preprocess,
    load_training_data,
    FEATURE_COLS,
    TARGET_COL
)


class TestPaySimDownload:
    """Tests for PaySim data loading."""

    @pytest.mark.slow
    def test_paysim_downloads_and_loads(self):
        """Test that PaySim dataset downloads and loads correctly."""
        df = download_paysim()
        assert len(df) > 6_000_000, "Expected >6M rows in PaySim dataset"
        assert 'isFraud' in df.columns, "Missing isFraud column"
        assert 'amount' in df.columns, "Missing amount column"
        assert 'step' in df.columns, "Missing step column"

    def test_preprocess_creates_features(self):
        """Test that preprocessing creates required features."""
        # Create minimal test data
        df = pd.DataFrame({
            'step': [1, 25, 50],
            'amount': [100.0, 500.0, 1000.0],
            'oldbalanceOrg': [1000.0, 2000.0, 3000.0],
            'newbalanceOrig': [900.0, 1500.0, 2000.0],
            'isFraud': [0, 0, 1]
        })

        processed = preprocess(df)

        # Check new features exist
        assert 'hour' in processed.columns
        assert 'amount_log' in processed.columns
        assert 'balance_diff' in processed.columns
        assert 'velocity_error' in processed.columns

        # Check hour calculation (step % 24)
        assert processed['hour'].iloc[0] == 1
        assert processed['hour'].iloc[1] == 1  # 25 % 24 = 1

        # Check amount_log
        assert np.isclose(processed['amount_log'].iloc[0], np.log1p(100.0))

        # Check balance_diff
        assert processed['balance_diff'].iloc[0] == 100.0  # 1000 - 900

    def test_load_training_data_returns_correct_format(self):
        """Test load_training_data returns X, y with correct columns."""
        # This test uses sampling for speed
        X, y = load_training_data(sample_size=1000)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y) == 1000

        # Check all feature columns present
        for col in FEATURE_COLS:
            assert col in X.columns, f"Missing feature column: {col}"

        # Check target is binary
        assert set(y.unique()).issubset({0, 1})

    def test_velocity_error_calculation(self):
        """Test velocity error detects balance inconsistencies."""
        # Transaction where balance adds up correctly
        correct_df = pd.DataFrame({
            'step': [1],
            'amount': [100.0],
            'oldbalanceOrg': [1000.0],
            'newbalanceOrig': [900.0],  # 1000 - 100 = 900
            'isFraud': [0]
        })

        # Transaction where balance doesn't add up
        incorrect_df = pd.DataFrame({
            'step': [1],
            'amount': [100.0],
            'oldbalanceOrg': [1000.0],
            'newbalanceOrig': [850.0],  # Should be 900
            'isFraud': [1]
        })

        correct_processed = preprocess(correct_df)
        incorrect_processed = preprocess(incorrect_df)

        assert correct_processed['velocity_error'].iloc[0] == 0
        assert incorrect_processed['velocity_error'].iloc[0] == 1
