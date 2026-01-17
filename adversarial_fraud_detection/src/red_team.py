"""Red Team adversarial testing loop."""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from src.models.transaction import Transaction, TransactionPerturbation
from src.models.defender import DefenderModel
from src.agents.attacker import AttackerAgent


@dataclass
class AttackResult:
    """Result of a single attack attempt."""
    original_transaction: Dict[str, Any]
    modified_transaction: Optional[Dict[str, Any]]
    original_score: float
    modified_score: Optional[float]
    evaded: bool
    score_reduction: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'original': self.original_transaction,
            'modified': self.modified_transaction,
            'original_score': self.original_score,
            'modified_score': self.modified_score,
            'evaded': self.evaded,
            'score_reduction': self.score_reduction
        }


@dataclass
class RedTeamReport:
    """Summary of a red team campaign."""
    total_attacks: int
    successful_perturbations: int
    successful_evasions: int
    evasion_rate: float
    avg_score_reduction: float
    attack_results: List[AttackResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def summary(self) -> str:
        return f"""
Red Team Campaign Report
========================
Timestamp: {self.timestamp}
Total Attacks: {self.total_attacks}
Successful Perturbations: {self.successful_perturbations}
Successful Evasions: {self.successful_evasions}
Evasion Rate: {self.evasion_rate:.2%}
Avg Score Reduction: {self.avg_score_reduction:.4f}
"""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_attacks': self.total_attacks,
            'successful_perturbations': self.successful_perturbations,
            'successful_evasions': self.successful_evasions,
            'evasion_rate': self.evasion_rate,
            'avg_score_reduction': self.avg_score_reduction,
            'timestamp': self.timestamp,
            'results': [r.to_dict() for r in self.attack_results]
        }


class RedTeamLoop:
    """
    Orchestrates adversarial attacks against the fraud detection model.

    This class runs the attacker agent against the defender model to find
    evasion strategies and measure model robustness.
    """

    def __init__(
        self,
        defender: DefenderModel,
        attacker: AttackerAgent,
        evasion_threshold: float = 0.5
    ):
        """
        Initialize the red team loop.

        Args:
            defender: Trained fraud detection model
            attacker: LLM-based attacker agent
            evasion_threshold: Score below which attack is considered successful
        """
        self.defender = defender
        self.attacker = attacker
        self.evasion_threshold = evasion_threshold

    def _transaction_to_features(self, txn: Transaction) -> pd.DataFrame:
        """Convert a Transaction to feature DataFrame for the defender."""
        features = txn.to_features()
        return pd.DataFrame([features])

    def attack_single(self, transaction: Transaction) -> AttackResult:
        """
        Execute a single attack attempt.

        Args:
            transaction: Original fraudulent transaction

        Returns:
            AttackResult with details of the attack
        """
        # Get original fraud score
        original_features = self._transaction_to_features(transaction)
        original_score = float(self.defender.predict_proba(original_features)[0])

        # Generate perturbation
        modified = self.attacker.generate_perturbation(transaction, original_score)

        if modified is None:
            return AttackResult(
                original_transaction=transaction.to_dict(),
                modified_transaction=None,
                original_score=original_score,
                modified_score=None,
                evaded=False,
                score_reduction=None
            )

        # Score the modified transaction
        modified_features = self._transaction_to_features(modified)
        modified_score = float(self.defender.predict_proba(modified_features)[0])

        evaded = modified_score < self.evasion_threshold
        score_reduction = original_score - modified_score

        return AttackResult(
            original_transaction=transaction.to_dict(),
            modified_transaction=modified.to_dict(),
            original_score=original_score,
            modified_score=modified_score,
            evaded=evaded,
            score_reduction=score_reduction
        )

    def run_campaign(
        self,
        transactions: List[Transaction],
        verbose: bool = True
    ) -> RedTeamReport:
        """
        Run a full red team campaign on multiple transactions.

        Args:
            transactions: List of fraudulent transactions to attack
            verbose: Print progress updates

        Returns:
            RedTeamReport with campaign results
        """
        results = []
        successful_perturbations = 0
        successful_evasions = 0
        score_reductions = []

        for i, txn in enumerate(transactions):
            if verbose:
                print(f"Attacking transaction {i+1}/{len(transactions)}...", end=" ")

            result = self.attack_single(txn)
            results.append(result)

            if result.modified_transaction is not None:
                successful_perturbations += 1
                score_reductions.append(result.score_reduction)

                if result.evaded:
                    successful_evasions += 1
                    if verbose:
                        print(f"EVADED! {result.original_score:.3f} -> {result.modified_score:.3f}")
                else:
                    if verbose:
                        print(f"Detected. {result.original_score:.3f} -> {result.modified_score:.3f}")
            else:
                if verbose:
                    print("Failed to generate perturbation")

        evasion_rate = successful_evasions / len(transactions) if transactions else 0
        avg_reduction = np.mean(score_reductions) if score_reductions else 0

        return RedTeamReport(
            total_attacks=len(transactions),
            successful_perturbations=successful_perturbations,
            successful_evasions=successful_evasions,
            evasion_rate=evasion_rate,
            avg_score_reduction=avg_reduction,
            attack_results=results
        )


def create_test_transactions(n: int = 5, use_real_data: bool = True) -> List[Transaction]:
    """
    Create sample fraudulent transactions for testing.

    Args:
        n: Number of transactions to generate
        use_real_data: If True, sample from actual PaySim fraud cases.
                       If False, generate synthetic transactions.

    Returns:
        List of Transaction objects representing fraud cases
    """
    if use_real_data:
        return _load_real_fraud_samples(n)
    else:
        return _generate_synthetic_transactions(n)


def _load_real_fraud_samples(n: int) -> List[Transaction]:
    """Load actual fraud cases from PaySim dataset."""
    from data.paysim_download import download_paysim, preprocess

    print("Loading real fraud samples from PaySim dataset...")

    # Load and preprocess data
    df = download_paysim()
    df = preprocess(df)

    # Filter for fraud cases only
    fraud_df = df[df['isFraud'] == 1].copy()
    print(f"Found {len(fraud_df)} fraud cases in dataset")

    # Sample n transactions (or all if n > available)
    n_samples = min(n, len(fraud_df))
    sampled = fraud_df.sample(n=n_samples, random_state=42)

    transactions = []
    for _, row in sampled.iterrows():
        try:
            # Calculate velocity_error from the data
            expected_balance = row['oldbalanceOrg'] - row['amount']
            velocity_error = 1 if abs(row['newbalanceOrig'] - expected_balance) > 0.01 else 0

            txn = Transaction(
                amount=float(row['amount']),
                hour=int(row['hour']),
                oldbalanceOrg=float(row['oldbalanceOrg']),
                newbalanceOrig=float(row['newbalanceOrig']),
                velocity_error=velocity_error
            )
            transactions.append(txn)
        except Exception as e:
            # Skip invalid transactions (e.g., balance constraint violations)
            print(f"Skipping invalid transaction: {e}")
            continue

    print(f"Loaded {len(transactions)} valid fraud transactions")
    return transactions


def _generate_synthetic_transactions(n: int) -> List[Transaction]:
    """Generate synthetic fraudulent transactions (fallback)."""
    np.random.seed(42)
    transactions = []

    for _ in range(n):
        amount = np.random.uniform(5000, 50000)
        old_balance = np.random.uniform(10000, 100000)
        new_balance = old_balance - amount

        txn = Transaction(
            amount=amount,
            hour=np.random.randint(0, 24),
            oldbalanceOrg=old_balance,
            newbalanceOrig=max(0, new_balance),
            velocity_error=np.random.choice([0, 1], p=[0.3, 0.7])
        )
        transactions.append(txn)

    return transactions


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run red team adversarial attacks')
    parser.add_argument('--num-attacks', type=int, default=5,
                        help='Number of transactions to attack')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Evasion threshold (default: 0.5)')
    args = parser.parse_args()

    print("Loading defender model...")
    defender = DefenderModel()
    defender.load()

    print("Initializing attacker agent...")
    attacker = AttackerAgent()

    if not attacker.test_connection():
        print("\nERROR: Cannot connect to Ollama.")
        print("Please ensure Ollama is running: ollama serve")
        exit(1)

    print(f"\nRunning red team campaign with {args.num_attacks} attacks...")
    print("-" * 50)

    red_team = RedTeamLoop(defender, attacker, evasion_threshold=args.threshold)
    transactions = create_test_transactions(args.num_attacks)
    report = red_team.run_campaign(transactions)

    print(report.summary())
