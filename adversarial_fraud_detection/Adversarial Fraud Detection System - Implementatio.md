<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Adversarial Fraud Detection System - Implementation Guide

## Project Overview

A self-red-teaming fraud detection platform that automatically probes an XGBoost fraud model for weaknesses and quantifies financial risk. This MVP demonstrates adversarial ML, P\&L optimization, and automated regulatory reporting for Sr. Leadership roles in fintech.

**Timeline**: 10-14 days
**Budget**: <\$10
**Tech Stack**: Python 3.10+, XGBoost, Ollama (local LLM), Streamlit, SHAP

***

## Architecture (Simplified)

```
Transaction → Defender Model (XGBoost) → Attacker Agent (Ollama) → 
Perturbation Loop → SHAP Explainer → SR 11-7 Report Generator
```

**Three Core Components**:

1. **Defender**: XGBoost classifier (fraud probability scorer)
2. **Attacker**: LLM-powered agent that modifies transactions to evade detection
3. **Strategist**: Cost-benefit calculator (Jupyter notebook)

***

## File Structure

```
fraud-red-team/
├── data/
│   └── paysim_download.py          # Kaggle dataset loader
├── src/
│   ├── models/
│   │   ├── defender.py             # XGBoost training & inference
│   │   └── transaction.py          # Pydantic data models
│   ├── agents/
│   │   └── attacker.py             # LLM perturbation engine
│   ├── red_team.py                 # Main adversarial loop
│   └── explainer.py                # SHAP integration
├── reports/
│   ├── template.html               # SR 11-7 HTML template
│   └── generator.py                # PDF generation
├── notebooks/
│   └── cost_analysis.ipynb         # Monte Carlo P&L simulator
├── app.py                          # Streamlit dashboard
├── requirements.txt
├── pytest.ini
└── README.md
```


***

## Phase 1: Data \& Defender Model (Days 1-2)

### Task 1.1: Data Loading

**File**: `data/paysim_download.py`

```python
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

def download_paysim():
    """Download PaySim dataset from Kaggle."""
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('ntnu-testimon/paysim1', path='./data', unzip=True)
    return pd.read_csv('./data/PS_20174392719_1491204439457_log.csv')

def preprocess(df):
    """Engineer features for fraud detection."""
    df['hour'] = df['step'] % 24
    df['amount_log'] = np.log1p(df['amount'])
    df['balance_diff'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['velocity_error'] = (df['oldbalanceOrg'] - df['amount'] != df['newbalanceOrig']).astype(int)
    return df

# Features to use
FEATURE_COLS = ['amount', 'amount_log', 'hour', 'balance_diff', 'velocity_error', 'oldbalanceOrg']
TARGET_COL = 'isFraud'
```

**Test Case**: `tests/test_data.py`

```python
def test_paysim_loads():
    df = download_paysim()
    assert len(df) > 6000000
    assert 'isFraud' in df.columns
```


***

### Task 1.2: Defender Model

**File**: `src/models/defender.py`

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib

class DefenderModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            objective='binary:logistic',
            eval_metric='auc'
        )
    
    def train(self, X, y):
        """Train fraud detection model."""
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=False
        )
        return self
    
    def predict_proba(self, X):
        """Return fraud probability [0-1]."""
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, path='models/defender.pkl'):
        joblib.dump(self.model, path)
```

**Test Case**:

```python
def test_defender_trains():
    X, y = load_sample_data()
    defender = DefenderModel().train(X, y)
    score = defender.predict_proba(X[:1])[^0]
    assert 0 <= score <= 1
```


***

## Phase 2: Attacker Agent (Days 3-5)

### Task 2.1: Transaction Data Model

**File**: `src/models/transaction.py`

```python
from pydantic import BaseModel, Field, validator

class Transaction(BaseModel):
    amount: float = Field(gt=0)
    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float
    hour: int = Field(ge=0, le=23)
    velocity_error: int = Field(ge=0, le=1)
    
    @validator('newbalanceOrig')
    def balance_logic(cls, v, values):
        """Ensure balance makes sense."""
        if 'oldbalanceOrg' in values and 'amount' in values:
            expected = values['oldbalanceOrg'] - values['amount']
            assert abs(v - expected) < values['amount'] * 0.1, "Balance constraint violated"
        return v
    
    def to_features(self):
        """Convert to XGBoost feature vector."""
        return {
            'amount': self.amount,
            'amount_log': np.log1p(self.amount),
            'hour': self.hour,
            'balance_diff': self.oldbalanceOrg - self.newbalanceOrig,
            'velocity_error': self.velocity_error,
            'oldbalanceOrg': self.oldbalanceOrg
        }
```


***

### Task 2.2: Attacker Agent

**File**: `src/agents/attacker.py`

```python
from openai import OpenAI
import json

class AttackerAgent:
    def __init__(self, model="llama3.2:3b"):
        """Initialize Ollama LLM client."""
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"  # Required but unused for local
        )
        self.model = model
    
    def generate_perturbation(self, transaction: Transaction, current_score: float) -> Transaction:
        """Generate adversarial modification to evade fraud detection."""
        
        prompt = f"""You are a fraud evasion expert. The current transaction has fraud score {current_score:.2f}.
        
Transaction:
{transaction.json(indent=2)}

Constraints:
- Amount can change by max 10%
- Hour can shift by max 2 hours
- Balance equations must remain valid
- newbalanceOrig = oldbalanceOrg - amount (within 10% tolerance)

Tactics to try:
1. Structuring: Split into smaller amounts
2. Timing: Shift to off-peak hours (3-6am)
3. Balance manipulation: Adjust newbalanceOrig within tolerance

Output a modified transaction as JSON that might score <0.5:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        modified = json.loads(response.choices[^0].message.content)
        return Transaction(**modified)
```

**Test Case**:

```python
def test_attacker_respects_constraints():
    original = Transaction(amount=1000, oldbalanceOrg=5000, newbalanceOrig=4000, hour=12, velocity_error=0)
    attacker = AttackerAgent()
    modified = attacker.generate_perturbation(original, 0.95)
    
    assert 900 <= modified.amount <= 1100  # 10% constraint
    assert 10 <= modified.hour <= 14  # 2-hour constraint
```


***

### Task 2.3: Red Team Loop

**File**: `src/red_team.py`

```python
from src.models.defender import DefenderModel
from src.agents.attacker import AttackerAgent
import pandas as pd

def red_team_transaction(
    transaction: Transaction,
    defender: DefenderModel,
    attacker: AttackerAgent,
    max_steps: int = 5
) -> dict:
    """Run adversarial attack loop."""
    
    history = []
    current = transaction
    
    for step in range(max_steps):
        # Score current transaction
        features = pd.DataFrame([current.to_features()])
        score = defender.predict_proba(features)[^0]
        
        history.append({
            'step': step,
            'transaction': current.dict(),
            'score': float(score)
        })
        
        # Success = evasion
        if score < 0.5:
            return {
                'success': True,
                'steps': step + 1,
                'final_score': float(score),
                'history': history
            }
        
        # Generate new attack
        current = attacker.generate_perturbation(current, score)
    
    return {
        'success': False,
        'steps': max_steps,
        'final_score': float(score),
        'history': history
    }
```


***

## Phase 3: Explainability \& Reporting (Days 6-8)

### Task 3.1: SHAP Integration

**File**: `src/explainer.py`

```python
import shap
import matplotlib.pyplot as plt

class FraudExplainer:
    def __init__(self, defender_model):
        self.explainer = shap.TreeExplainer(defender_model.model)
    
    def explain_attack(self, original: Transaction, modified: Transaction):
        """Generate SHAP waterfall showing feature changes."""
        
        orig_features = pd.DataFrame([original.to_features()])
        mod_features = pd.DataFrame([modified.to_features()])
        
        orig_shap = self.explainer(orig_features)
        mod_shap = self.explainer(mod_features)
        
        # Create side-by-side waterfall plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        shap.plots.waterfall(orig_shap[^0], show=False, ax=ax1)
        ax1.set_title(f"Original (Score: {orig_shap.base_values[^0]:.2f})")
        
        shap.plots.waterfall(mod_shap[^0], show=False, ax=ax2)
        ax2.set_title(f"Modified (Score: {mod_shap.base_values[^0]:.2f})")
        
        return fig
```


***

### Task 3.2: SR 11-7 Report Generator

**File**: `reports/template.html`

```html
<!DOCTYPE html>
<html>
<head>
    <title>SR 11-7 Model Stress Test Report</title>
    <style>
        body { font-family: Arial; margin: 40px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid black; padding: 8px; text-align: left; }
        .risk-high { background-color: #ffcccc; }
    </style>
</head>
<body>
    <h1>Model Risk Management: Adversarial Stress Test</h1>
    <p><strong>Model:</strong> XGBoost Fraud Detector v1.0</p>
    <p><strong>Test Date:</strong> {{ date }}</p>
    <p><strong>Regulation:</strong> Federal Reserve SR 11-7</p>
    
    <h2>Executive Summary</h2>
    <p>{{ total_attacks }} adversarial attacks attempted. {{ successful_attacks }} succeeded in evading detection ({{ success_rate }}% evasion rate).</p>
    
    <h2>Identified Vulnerabilities</h2>
    <table>
        <tr>
            <th>Attack ID</th>
            <th>Original Score</th>
            <th>Final Score</th>
            <th>Primary Exploit</th>
            <th>Risk Tier</th>
        </tr>
        {% for attack in attacks %}
        <tr class="{{ 'risk-high' if attack.final_score < 0.3 else '' }}">
            <td>{{ attack.id }}</td>
            <td>{{ "%.2f"|format(attack.original_score) }}</td>
            <td>{{ "%.2f"|format(attack.final_score) }}</td>
            <td>{{ attack.exploit_method }}</td>
            <td>{{ attack.risk_tier }}</td>
        </tr>
        {% endfor %}
    </table>
    
    <h2>Recommendations</h2>
    <ul>
        {% for rec in recommendations %}
        <li>{{ rec }}</li>
        {% endfor %}
    </ul>
</body>
</html>
```

**File**: `reports/generator.py`

```python
from jinja2 import Template
import pdfkit
from datetime import datetime

def generate_sr11_7_report(results: list, output_path='sr11_7_report.pdf'):
    """Generate SR 11-7 compliance PDF."""
    
    with open('reports/template.html') as f:
        template = Template(f.read())
    
    successful = [r for r in results if r['success']]
    
    html = template.render(
        date=datetime.now().strftime('%Y-%m-%d'),
        total_attacks=len(results),
        successful_attacks=len(successful),
        success_rate=round(len(successful)/len(results)*100, 1),
        attacks=[
            {
                'id': i,
                'original_score': r['history'][^0]['score'],
                'final_score': r['final_score'],
                'exploit_method': 'Structuring' if r['success'] else 'N/A',
                'risk_tier': 'HIGH' if r['final_score'] < 0.3 else 'MEDIUM'
            }
            for i, r in enumerate(results)
        ],
        recommendations=[
            'Implement velocity checks with 15-min windows',
            'Add cross-channel correlation features',
            'Retrain model with synthetic adversarial examples'
        ]
    )
    
    pdfkit.from_string(html, output_path)
```


***

## Phase 4: Dashboard \& Analysis (Days 9-10)

### Task 4.1: Streamlit App

**File**: `app.py`

```python
import streamlit as st
import pandas as pd
from src.red_team import red_team_transaction
from src.models.defender import DefenderModel
from src.agents.attacker import AttackerAgent
from src.explainer import FraudExplainer
import joblib

st.set_page_config(page_title="Fraud Red Team", layout="wide")

# Load model
@st.cache_resource
def load_models():
    defender = DefenderModel()
    defender.model = joblib.load('models/defender.pkl')
    attacker = AttackerAgent()
    explainer = FraudExplainer(defender)
    return defender, attacker, explainer

defender, attacker, explainer = load_models()

st.title("🛡️ Adversarial Fraud Detection System")

# Input section
st.header("Test Transaction")
col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Amount ($)", value=1500.0, min_value=0.0)
    oldbalance = st.number_input("Old Balance ($)", value=5000.0, min_value=0.0)
    hour = st.slider("Hour of Day", 0, 23, 14)

with col2:
    newbalance = st.number_input("New Balance ($)", value=oldbalance-amount)
    velocity_error = st.selectbox("Velocity Error", [0, 1])
    max_steps = st.slider("Max Attack Steps", 1, 10, 5)

if st.button("🔴 Run Red Team Attack"):
    transaction = Transaction(
        amount=amount,
        oldbalanceOrg=oldbalance,
        newbalanceOrig=newbalance,
        hour=hour,
        velocity_error=velocity_error
    )
    
    with st.spinner("Running adversarial attacks..."):
        result = red_team_transaction(transaction, defender, attacker, max_steps)
    
    # Results
    st.header("Attack Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Status", "✅ EVADED" if result['success'] else "❌ BLOCKED")
    col2.metric("Steps Taken", result['steps'])
    col3.metric("Final Score", f"{result['final_score']:.3f}")
    
    # History chart
    st.subheader("Attack Progression")
    history_df = pd.DataFrame(result['history'])
    st.line_chart(history_df.set_index('step')['score'])
    
    # SHAP explanation
    if result['success']:
        st.subheader("Feature Exploitation Analysis")
        original_txn = Transaction(**result['history'][^0]['transaction'])
        final_txn = Transaction(**result['history'][-1]['transaction'])
        fig = explainer.explain_attack(original_txn, final_txn)
        st.pyplot(fig)
```


***

### Task 4.2: Cost Analysis Notebook

**File**: `notebooks/cost_analysis.ipynb`

```python
# Cell 1: Setup
import numpy as np
import matplotlib.pyplot as plt

# Financial parameters
FALSE_POSITIVE_COST = 50  # Customer service call
FRAUD_COST = 200  # Average fraud loss
DAILY_TRANSACTIONS = 100000
FRAUD_RATE = 0.002  # 0.2% base rate

# Cell 2: Monte Carlo Simulation
def simulate_threshold_impact(threshold, n_simulations=10000):
    results = []
    
    for _ in range(n_simulations):
        # Generate synthetic scores
        fraud_scores = np.random.beta(8, 2, int(DAILY_TRANSACTIONS * FRAUD_RATE))
        legit_scores = np.random.beta(2, 8, int(DAILY_TRANSACTIONS * (1-FRAUD_RATE)))
        
        # Calculate costs
        blocked_fraud = np.sum(fraud_scores > threshold)
        missed_fraud = np.sum(fraud_scores <= threshold)
        false_positives = np.sum(legit_scores > threshold)
        
        total_cost = (missed_fraud * FRAUD_COST) + (false_positives * FALSE_POSITIVE_COST)
        results.append(total_cost)
    
    return np.mean(results), np.percentile(results, [5, 95])

# Cell 3: Threshold Sweep
thresholds = np.linspace(0.3, 0.9, 20)
costs = [simulate_threshold_impact(t)[^0] for t in thresholds]

plt.figure(figsize=(10, 6))
plt.plot(thresholds, costs, marker='o')
plt.xlabel('Fraud Detection Threshold')
plt.ylabel('Expected Daily Cost ($)')
plt.title('P&L Optimization: Fraud vs. Friction Trade-off')
plt.grid(True)
plt.show()

# Cell 4: Recommendation
optimal_threshold = thresholds[np.argmin(costs)]
print(f"Optimal Threshold: {optimal_threshold:.2f}")
print(f"Expected Daily Savings: ${min(costs) - costs[^10]:.2f}")
```


***

## Testing Strategy

**File**: `pytest.ini`

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
```

**File**: `tests/test_integration.py`

```python
def test_end_to_end_red_team():
    """Full pipeline test."""
    # Load model
    defender = DefenderModel()
    defender.model = joblib.load('models/defender.pkl')
    attacker = AttackerAgent()
    
    # Create high-fraud transaction
    txn = Transaction(
        amount=5000,
        oldbalanceOrg=6000,
        newbalanceOrig=1000,
        hour=3,  # Suspicious late hour
        velocity_error=1
    )
    
    # Run attack
    result = red_team_transaction(txn, defender, attacker, max_steps=3)
    
    # Verify structure
    assert 'success' in result
    assert 'history' in result
    assert len(result['history']) >= 1
```


***

## Requirements File

**File**: `requirements.txt`

```
xgboost==2.0.3
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
shap==0.44.0
pydantic==2.5.3
openai==1.6.1
streamlit==1.29.0
jinja2==3.1.2
pdfkit==1.0.0
joblib==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
pytest==7.4.3
kaggle==1.5.16
```


***

## Setup Instructions

```bash
# 1. Install Ollama (MacOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b

# 2. Setup Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Download data
python data/paysim_download.py

# 4. Train defender
python -m src.models.defender

# 5. Run tests
pytest

# 6. Launch dashboard
streamlit run app.py
```


***

## Success Criteria

- ✅ XGBoost model achieves AUC > 0.95 on PaySim
- ✅ Attacker finds ≥1 evasion in 5 steps for high-fraud transactions
- ✅ SHAP waterfall correctly identifies exploited features
- ✅ SR 11-7 PDF generates with <5 lines of code
- ✅ Streamlit loads in <3 seconds
- ✅ All tests pass with `pytest`
- ✅ Total cost <\$10

***

## LinkedIn Demo Script

**Video Length**: 2-3 minutes

1. **Hook** (0:00-0:15): "I built an AI that tries to rob my own fraud detection system."
2. **Problem** (0:15-0:45): Show a fraudulent transaction getting caught (score=0.92)
3. **Solution** (0:45-1:30): Run red team attack in Streamlit, show score dropping to 0.34
4. **Insight** (1:30-2:00): Show SHAP waterfall revealing the exploit was "structuring"
5. **Business Value** (2:00-2:30): Pull up cost analysis notebook showing "\$50k/year savings from fixing this vulnerability"
6. **CTA** (2:30-3:00): "At Fidelity, I managed models protecting \$X billion. This is how I'd automate that process. Link in comments."

***

## Extensions (Post-MVP)

- Add network graph visualization (account-to-account flows)
- Integrate with live transaction stream via Kafka
- Multi-model ensemble (XGBoost + LightGBM + Neural Net)
- Deploy as FastAPI backend + React frontend
- Add reinforcement learning for adaptive attacker[^1][^2]

***

**This structure is optimized for Claude Code**: modular files, clear contracts, testable functions, and progressive complexity.[^3][^4]

<div align="center">⁂</div>

[^1]: https://superagi.com/advanced-deep-learning-architectures-for-ai-powered-fraud-detection-trends-and-best-practices/

[^2]: https://www.techaheadcorp.com/blog/mlops-for-real-time-fraud-detection-for-financial-services/

[^3]: https://www.anthropic.com/engineering/claude-code-best-practices

[^4]: https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents

