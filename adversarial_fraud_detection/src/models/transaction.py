"""Transaction data model with validation."""
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, Any


class Transaction(BaseModel):
    """
    Represents a financial transaction with fraud detection features.

    Attributes:
        amount: Transaction amount (must be positive)
        oldbalanceOrg: Account balance before transaction
        newbalanceOrig: Account balance after transaction
        hour: Hour of day (0-23)
        velocity_error: Whether balance equation has discrepancy (0 or 1)
    """
    amount: float = Field(gt=0, description="Transaction amount")
    oldbalanceOrg: float = Field(ge=0, description="Balance before transaction")
    newbalanceOrig: float = Field(description="Balance after transaction")
    hour: int = Field(ge=0, le=23, description="Hour of day (0-23)")
    velocity_error: int = Field(ge=0, le=1, description="Balance discrepancy flag")

    @model_validator(mode='after')
    def validate_balance_logic(self) -> 'Transaction':
        """Ensure balance makes sense within tolerance."""
        expected = self.oldbalanceOrg - self.amount
        tolerance = self.amount * 0.1  # 10% tolerance

        if abs(self.newbalanceOrig - expected) > tolerance:
            raise ValueError(
                f"Balance constraint violated: newbalanceOrig={self.newbalanceOrig}, "
                f"expected ~{expected} (oldbalanceOrg - amount)"
            )
        return self

    def to_features(self) -> Dict[str, float]:
        """Convert transaction to feature vector for XGBoost."""
        return {
            'amount': self.amount,
            'amount_log': float(np.log1p(self.amount)),
            'hour': self.hour,
            'balance_diff': self.oldbalanceOrg - self.newbalanceOrig,
            'velocity_error': self.velocity_error,
            'oldbalanceOrg': self.oldbalanceOrg
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (alias for model_dump)."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transaction':
        """Create Transaction from dictionary."""
        return cls(**data)


class TransactionPerturbation(BaseModel):
    """
    Defines constraints for how a transaction can be modified.

    Used by the attacker agent to ensure perturbations stay within
    realistic bounds.
    """
    max_amount_change_pct: float = Field(default=0.10, description="Max % change to amount")
    max_hour_shift: int = Field(default=2, description="Max hours to shift")

    def is_valid_perturbation(self, original: Transaction, modified: Transaction) -> bool:
        """Check if modified transaction is within perturbation bounds."""
        # Check amount constraint
        amount_change = abs(modified.amount - original.amount) / original.amount
        if amount_change > self.max_amount_change_pct:
            return False

        # Check hour constraint
        hour_diff = abs(modified.hour - original.hour)
        # Handle wrap-around (e.g., 23 to 1 = 2 hours)
        hour_diff = min(hour_diff, 24 - hour_diff)
        if hour_diff > self.max_hour_shift:
            return False

        return True
