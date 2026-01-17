# Adversarial Fraud Detection System

A self-red-teaming fraud detection platform that automatically probes an XGBoost fraud model for vulnerabilities using LLM-powered adversarial attacks.

## Overview

This system demonstrates:
- **Adversarial ML Testing**: LLM-based attacks that modify fraudulent transactions to evade detection
- **Model Explainability**: SHAP-based explanations showing which features are exploited
- **Regulatory Compliance**: SR 11-7 compliant model risk assessment reports
- **Cost-Benefit Analysis**: Monte Carlo simulation for quantifying business impact

## Architecture

```
Transaction (Real Fraud) --> Defender Model (XGBoost) --> Attacker Agent (Ollama LLM)
                                    |                            |
                                    v                            v
                            Fraud Score (0.78)         Perturbation (-10% amount, +2hr)
                                    |                            |
                                    +------- Re-score -----------+
                                                |
                                                v
                                    Modified Score (0.005) --> EVASION!
                                                |
                                                v
                                    SHAP Explanation + SR 11-7 Report
```

## Features

| Component | Description |
|-----------|-------------|
| **Defender** | XGBoost classifier trained on PaySim (6.3M transactions, AUC 0.9989) |
| **Attacker** | Local LLM (Ollama + llama3.2:3b) generates adversarial perturbations |
| **Explainer** | SHAP TreeExplainer identifies exploited features |
| **Reporter** | Jinja2 + pdfkit generates SR 11-7 compliant PDFs |
| **Dashboard** | Streamlit UI for interactive red team testing |

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running
- Kaggle account (for PaySim dataset)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/adversarial-fraud-detection.git
cd adversarial-fraud-detection

# Create virtual environment
python -m venv .fraud-venv
source .fraud-venv/bin/activate  # Windows: .fraud-venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull LLM model
ollama pull llama3.2:3b
```

### Setup Kaggle API

1. Go to https://www.kaggle.com/settings
2. Click "Create New Token" under API section
3. Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)
4. Accept PaySim dataset terms: https://www.kaggle.com/datasets/mtalaltariq/paysim-data

### Run

```bash
# Download data and train model
python data/paysim_download.py
python -m src.models.defender

# Run tests
pytest -v

# Launch dashboard
streamlit run app.py
```

## Usage

### Command Line

```bash
# Run red team campaign (5 attacks)
python -m src.red_team --num-attacks 5

# Generate SHAP explanations
python -m src.explainer

# Generate SR 11-7 report
python -m reports.generator
```

### Streamlit Dashboard

The dashboard provides 5 pages:
1. **Overview** - Model metrics and feature importance
2. **Single Transaction** - Test individual transactions
3. **Red Team Campaign** - Run batch adversarial attacks
4. **Explainability** - SHAP waterfall plots
5. **Report Generator** - Generate PDF/HTML reports

## Project Structure

```
adversarial-fraud-detection/
├── data/
│   └── paysim_download.py      # Dataset loader
├── src/
│   ├── models/
│   │   ├── defender.py         # XGBoost fraud detector
│   │   └── transaction.py      # Pydantic data models
│   ├── agents/
│   │   └── attacker.py         # LLM perturbation engine
│   ├── red_team.py             # Adversarial testing loop
│   └── explainer.py            # SHAP integration
├── reports/
│   ├── template.html           # SR 11-7 report template
│   └── generator.py            # PDF generation
├── notebooks/
│   └── cost_analysis.ipynb     # Monte Carlo P&L simulation
├── tests/
│   ├── test_data.py
│   └── test_models.py
├── app.py                      # Streamlit dashboard
└── requirements.txt
```

## How It Works

### 1. Defender Model
XGBoost classifier trained on PaySim synthetic fraud dataset:
- 6.3M transactions, 8,213 fraud cases (0.13%)
- Features: amount, hour, balance_diff, velocity_error
- Validation AUC: 0.9989

### 2. Attacker Agent
LLM-powered adversarial agent that:
- Receives a flagged fraud transaction (score > 0.5)
- Generates perturbations within constraints (±10% amount, ±2 hour)
- Aims to reduce fraud score below detection threshold

### 3. Perturbation Constraints
Realistic business constraints ensure attacks are plausible:
```python
max_amount_change_pct = 0.10  # ±10% of transaction amount
max_hour_shift = 2            # ±2 hours
```

### 4. Explainability
SHAP TreeExplainer reveals which features the attacker exploited:
- `velocity_error` (75% importance) - most commonly targeted
- `balance_diff` (12% importance)
- `amount` (4% importance)

## Results

Typical red team campaign results:
```
Total Attacks: 5
Successful Evasions: 3
Evasion Rate: 60%
Avg Score Reduction: 0.77
```

## Dependencies

- xgboost >= 2.0.0
- pandas >= 2.0.0
- shap >= 0.44.0
- openai >= 1.6.0 (for Ollama API)
- streamlit >= 1.29.0
- pdfkit >= 1.0.0
- pydantic >= 2.5.0

## Cost

**Total cloud cost: $0** - Everything runs locally with Ollama.

## License

MIT License

## Acknowledgments

- [PaySim](https://www.kaggle.com/datasets/mtalaltariq/paysim-data) synthetic fraud dataset
- [Ollama](https://ollama.com/) for local LLM inference
- [SHAP](https://github.com/slundberg/shap) for model explainability
