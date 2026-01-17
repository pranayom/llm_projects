"""LLM-based attacker agent for adversarial fraud detection."""
import json
import re
from typing import Optional, Dict, Any, List
from openai import OpenAI

from src.models.transaction import Transaction, TransactionPerturbation


# Default Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.2:3b"


class AttackerAgent:
    """
    LLM-powered adversarial agent that generates transaction perturbations
    to evade fraud detection.

    Uses Ollama's local LLM to intelligently modify fraudulent transactions
    while staying within realistic business constraints.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        constraints: Optional[TransactionPerturbation] = None
    ):
        """
        Initialize the attacker agent.

        Args:
            base_url: Ollama API endpoint
            model: Model name to use
            constraints: Perturbation constraints (default: 10% amount, 2hr time)
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key="ollama"  # Ollama doesn't need a real key
        )
        self.model = model
        self.constraints = constraints or TransactionPerturbation()

    def _build_prompt(self, transaction: Transaction, fraud_score: float) -> str:
        """Build the prompt for the LLM."""
        return f"""You are a red-team agent testing a fraud detection system. Your goal is to modify a fraudulent transaction to evade detection while keeping it realistic.

CURRENT TRANSACTION:
- Amount: ${transaction.amount:,.2f}
- Hour of day: {transaction.hour}
- Old balance: ${transaction.oldbalanceOrg:,.2f}
- New balance: ${transaction.newbalanceOrig:,.2f}
- Velocity error: {transaction.velocity_error}
- Current fraud score: {fraud_score:.4f} (1.0 = definitely fraud)

CONSTRAINTS (must follow these exactly):
- Amount can only change by ±{self.constraints.max_amount_change_pct*100:.0f}% (range: ${transaction.amount * (1-self.constraints.max_amount_change_pct):,.2f} to ${transaction.amount * (1+self.constraints.max_amount_change_pct):,.2f})
- Hour can only shift by ±{self.constraints.max_hour_shift} hours (range: {max(0, transaction.hour - self.constraints.max_hour_shift)} to {min(23, transaction.hour + self.constraints.max_hour_shift)})
- New balance must approximately equal: old_balance - amount (within 10% tolerance)
- Velocity error must be 0 or 1

GOAL: Suggest modifications that might lower the fraud score while staying within constraints.

Respond ONLY with valid JSON in this exact format:
{{"amount": <new_amount>, "hour": <new_hour>, "oldbalanceOrg": <same_or_adjusted>, "newbalanceOrig": <calculated_new_balance>, "velocity_error": <0_or_1>, "reasoning": "<brief_explanation>"}}"""

    def _parse_response(self, response: str, original: Transaction) -> Optional[Dict[str, Any]]:
        """Parse LLM response to extract transaction modifications."""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if not json_match:
                return None

            data = json.loads(json_match.group())

            # Validate required fields
            required = ['amount', 'hour', 'oldbalanceOrg', 'velocity_error']
            if not all(k in data for k in required):
                return None

            # Auto-correct newbalanceOrig to satisfy balance constraint
            # LLMs often forget to recalculate this when amount changes
            data['newbalanceOrig'] = data['oldbalanceOrg'] - data['amount']

            return data

        except (json.JSONDecodeError, KeyError):
            return None

    def generate_perturbation(
        self,
        transaction: Transaction,
        fraud_score: float,
        max_retries: int = 3
    ) -> Optional[Transaction]:
        """
        Generate a perturbed transaction using the LLM.

        Args:
            transaction: Original fraudulent transaction
            fraud_score: Current fraud probability from defender
            max_retries: Number of LLM call attempts

        Returns:
            Modified Transaction if successful, None if failed
        """
        prompt = self._build_prompt(transaction, fraud_score)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7 + (attempt * 0.1),  # Increase randomness on retries
                    max_tokens=300
                )

                response_text = response.choices[0].message.content
                parsed = self._parse_response(response_text, transaction)

                if parsed is None:
                    continue

                # Create modified transaction
                try:
                    modified = Transaction(
                        amount=float(parsed['amount']),
                        hour=int(parsed['hour']),
                        oldbalanceOrg=float(parsed['oldbalanceOrg']),
                        newbalanceOrig=float(parsed['newbalanceOrig']),
                        velocity_error=int(parsed['velocity_error'])
                    )

                    # Validate perturbation is within constraints
                    if self.constraints.is_valid_perturbation(transaction, modified):
                        return modified

                except (ValueError, TypeError):
                    continue

            except Exception as e:
                print(f"LLM call failed (attempt {attempt + 1}): {e}")
                continue

        return None

    def batch_generate(
        self,
        transactions: List[Transaction],
        fraud_scores: List[float]
    ) -> List[Optional[Transaction]]:
        """
        Generate perturbations for multiple transactions.

        Args:
            transactions: List of original transactions
            fraud_scores: Corresponding fraud scores

        Returns:
            List of modified transactions (None for failures)
        """
        results = []
        for txn, score in zip(transactions, fraud_scores):
            modified = self.generate_perturbation(txn, score)
            results.append(modified)
        return results

    def test_connection(self) -> bool:
        """Test if Ollama is accessible."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Say 'OK' if you can read this."}],
                max_tokens=10
            )
            return len(response.choices) > 0
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False


if __name__ == '__main__':
    # Quick test
    print("Testing AttackerAgent...")
    agent = AttackerAgent()

    if agent.test_connection():
        print("✓ Connected to Ollama")

        # Create test transaction
        test_txn = Transaction(
            amount=10000.0,
            hour=14,
            oldbalanceOrg=50000.0,
            newbalanceOrig=40000.0,
            velocity_error=1
        )

        print(f"\nOriginal transaction: {test_txn.model_dump()}")

        modified = agent.generate_perturbation(test_txn, fraud_score=0.95)
        if modified:
            print(f"Modified transaction: {modified.model_dump()}")
        else:
            print("Failed to generate valid perturbation")
    else:
        print("✗ Could not connect to Ollama")
        print("  Make sure Ollama is running: ollama serve")
