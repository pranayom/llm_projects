# Learnings & Troubleshooting Guide

## Adversarial Fraud Detection System - Implementation Learnings

This document captures errors encountered during implementation, their solutions, and best practices for similar projects.

---

## Errors Encountered & Solutions

### 1. Kaggle Dataset Filename Mismatch

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data\PS_20174392719_1491204439457_log.csv'
```

**Root Cause:**
The Kaggle API downloaded the PaySim dataset with a different filename than expected. The code hardcoded `PS_20174392719_1491204439457_log.csv` but the actual downloaded file was `paysim dataset.csv`.

**Solution:**
Updated `data/paysim_download.py` line 15:
```python
# Before
CSV_FILENAME = 'PS_20174392719_1491204439457_log.csv'

# After
CSV_FILENAME = 'paysim dataset.csv'
```

**Best Practice:**
- After downloading datasets via API, always verify the actual filename before hardcoding
- Consider using glob patterns to find CSV files dynamically:
  ```python
  import glob
  csv_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
  ```

---

### 2. LLM JSON Response Balance Calculation Error

**Error:**
```
ValueError: Balance constraint violated: newbalanceOrig=40000.0, expected ~41000.0 (oldbalanceOrg - amount)
```

**Root Cause:**
When the LLM (llama3.2:3b) modified the transaction amount, it often forgot to recalculate `newbalanceOrig`. The Pydantic model's validator correctly rejected these invalid transactions.

**Example LLM Response:**
```json
{"amount": 9000, "hour": 14, "oldbalanceOrg": 50000, "newbalanceOrig": 40000, "velocity_error": 1}
```
The LLM changed amount from 10000 to 9000 but left newbalanceOrig at 40000 (should be 41000).

**Solution:**
Auto-correct the balance in `src/agents/attacker.py`:
```python
def _parse_response(self, response: str, original: Transaction) -> Optional[Dict[str, Any]]:
    # ... parse JSON ...

    # Auto-correct newbalanceOrig to satisfy balance constraint
    # LLMs often forget to recalculate this when amount changes
    data['newbalanceOrig'] = data['oldbalanceOrg'] - data['amount']

    return data
```

**Best Practice:**
- Don't trust LLMs to maintain mathematical consistency
- Add post-processing to fix deterministic calculations
- Keep validation in the model (Pydantic) but handle LLM limitations gracefully
- Consider using structured output / function calling for more reliable JSON

---

### 3. Windows Console Unicode Encoding Error

**Error:**
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' in position 14: character maps to <undefined>
```

**Root Cause:**
Windows Command Prompt uses `cp1252` encoding by default, which doesn't support the arrow character `→` used in the output.

**Solution:**
Replace Unicode arrows with ASCII alternatives in `src/red_team.py`:
```python
# Before
print(f"EVADED! {result.original_score:.3f} → {result.modified_score:.3f}")

# After
print(f"EVADED! {result.original_score:.3f} -> {result.modified_score:.3f}")
```

**Best Practice:**
- Avoid Unicode special characters in console output for cross-platform compatibility
- Use ASCII alternatives: `→` becomes `->`, `✓` becomes `[OK]`, `✗` becomes `[FAIL]`
- If Unicode is required, set encoding explicitly:
  ```python
  import sys
  sys.stdout.reconfigure(encoding='utf-8')
  ```

---

### 4. Ollama Service Confusion on Windows

**Error (User Confusion):**
User tried running `ollama serve` in command prompt, thinking it was required.

**Root Cause:**
On Windows, Ollama installs as a background service that starts automatically. Unlike Linux/Mac where you might need to run `ollama serve`, Windows users don't need to do this.

**Solution:**
Verify Ollama is running with:
```bash
ollama list
```
Or check `http://localhost:11434` in browser.

**Best Practice:**
- Document platform-specific differences in setup instructions
- Provide verification commands rather than startup commands
- Add connection testing in code:
  ```python
  def test_connection(self) -> bool:
      try:
          response = self.client.chat.completions.create(...)
          return len(response.choices) > 0
      except Exception:
          return False
  ```

---

### 5. SHAP Expected Value Format Variation

**Potential Error:**
```
TypeError: 'list' object cannot be interpreted as float
```

**Root Cause:**
SHAP's `TreeExplainer.expected_value` can return either a float (binary classification) or a list (multi-class). Code must handle both.

**Solution:**
```python
base_value = self.explainer.expected_value
if isinstance(base_value, list):
    base_value = base_value[1]  # Class 1 (fraud) for binary classification
```

**Best Practice:**
- Always check SHAP output types - they vary by model and configuration
- Use `isinstance()` checks for flexibility
- Document which class index corresponds to the positive class

---

## Architecture Recommendations

### 1. Separation of Concerns

```
RedTeamLoop (Orchestrator)
    ├── DefenderModel (XGBoost classifier)
    ├── AttackerAgent (LLM perturbation engine)
    └── FraudExplainer (SHAP explanations)
```

The `AttackerAgent` doesn't know about the `DefenderModel` directly. The `RedTeamLoop` coordinates between them, passing only necessary information (fraud scores). This makes components independently testable.

### 2. Constraint Validation Strategy

**Two-Layer Validation:**
1. **Pydantic Models** (`Transaction`) - Hard constraints, raise errors
2. **Perturbation Checker** (`TransactionPerturbation.is_valid_perturbation`) - Soft constraints, return boolean

This allows the LLM to generate candidates that are structurally valid (Pydantic) even if they violate business rules (perturbation limits).

### 3. LLM Integration Pattern

```python
class AttackerAgent:
    def generate_perturbation(self, txn, score, max_retries=3):
        for attempt in range(max_retries):
            response = self._call_llm(txn, score)
            parsed = self._parse_response(response)

            if parsed and self._validate(parsed):
                return Transaction(**parsed)

        return None  # Graceful failure
```

**Key patterns:**
- Retry with increasing temperature
- Parse and validate separately
- Always have a fallback (return None, not raise)
- Log failures for debugging

### 4. Report Generation Architecture

```
RedTeamReport (dataclass)
    └── ReportGenerator
        ├── Jinja2 Template (HTML)
        └── pdfkit (optional PDF)
```

**Benefits:**
- Data (RedTeamReport) is separate from presentation (templates)
- HTML works everywhere; PDF is optional dependency
- Easy to add new report formats (Excel, Markdown, etc.)

---

## Testing Recommendations

### 1. Test LLM Components with Mocks

```python
def test_attacker_with_mock_llm():
    mock_response = '{"amount": 9000, "hour": 14, ...}'
    with patch.object(AttackerAgent, '_call_llm', return_value=mock_response):
        result = agent.generate_perturbation(txn, 0.95)
        assert result is not None
```

### 2. Integration Tests Need Real Services

For LLM integration tests, check service availability first:
```python
@pytest.mark.skipif(not ollama_available(), reason="Ollama not running")
def test_real_llm_integration():
    ...
```

### 3. Use Deterministic Seeds

```python
np.random.seed(42)  # In test setup
```

---

## Performance Considerations

| Operation | Typical Time | Notes |
|-----------|--------------|-------|
| Model training (6.3M rows) | ~30 seconds | XGBoost is fast |
| Single LLM perturbation | 2-5 seconds | Depends on model size |
| SHAP explanation | 1-2 seconds | TreeExplainer is efficient |
| PDF generation | 3-5 seconds | Requires wkhtmltopdf |

**Bottleneck:** LLM calls are the slowest part. For large-scale testing:
- Batch transactions where possible
- Consider smaller models (llama3.2:1b)
- Cache successful perturbations

---

## Security Considerations

1. **This is a red-team tool** - Designed to find vulnerabilities, not exploit them
2. **Perturbation constraints** - Built-in limits prevent unrealistic attacks
3. **Local LLM** - Ollama runs locally, no data leaves the machine
4. **No production credentials** - Uses synthetic/sampled data only

---

## Future Improvements

1. **Structured Output** - Use Ollama's JSON mode for more reliable parsing
2. **Adversarial Training** - Feed successful evasions back into model training
3. **Multi-Model Ensemble** - Test against multiple defenders
4. **Automated CI/CD** - Run red team tests on model changes
5. **Cost Analysis** - Quantify financial impact of evasions

---

## Quick Reference: Common Commands

```bash
# Train model
python -m src.models.defender

# Run red team attacks
python -m src.red_team --num-attacks 10

# Generate SHAP explanations
python -m src.explainer

# Generate report
python -m reports.generator

# Run all tests
pytest -v
```

---

*Document created: January 17, 2026*
*Last updated: January 17, 2026*
