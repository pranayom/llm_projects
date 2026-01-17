# LinkedIn Demo Video Script

## Adversarial Fraud Detection System

**Video Length**: 2-3 minutes
**Format**: Screen recording with voiceover

---

## Updated Script (Based on Actual Implementation)

### 1. HOOK (0:00-0:15)

**[Show terminal with Ollama running]**

> "I built an AI that attacks my own fraud detection system - using a local LLM to find vulnerabilities before real fraudsters do."

---

### 2. THE PROBLEM (0:15-0:45)

**[Show Streamlit dashboard - Single Transaction page]**

> "Here's a real fraudulent transaction from the PaySim dataset - $50,000 transfer at 3am with a balance discrepancy."

**[Click Analyze - show fraud score 0.78]**

> "My XGBoost model correctly flags this as 78% likely fraud. But here's the question every risk manager should ask: Can a sophisticated attacker modify this transaction to slip past detection?"

---

### 3. THE ATTACK (0:45-1:30)

**[Switch to Red Team Campaign page]**

> "This is where the LLM attacker comes in. It's running locally on Ollama - no data leaves my machine."

**[Set attacks to 5, click Launch Campaign]**

> "The attacker samples REAL fraud cases from our dataset - 8,000+ actual fraud transactions. Then it uses an LLM to intelligently modify each one within realistic constraints - only 10% amount change, 2-hour time shift."

**[Show results loading - highlight a successful evasion]**

> "Look at this: Original score 0.78... modified score 0.005. The attack succeeded. The model was fooled."

**[Show the bar chart comparison]**

> "Out of 5 real fraud transactions, 3 evaded detection. That's a 60% evasion rate - a serious vulnerability."

---

### 4. THE EXPLANATION (1:30-2:00)

**[Switch to Explainability page]**

> "But we don't just find vulnerabilities - we explain them. This SHAP waterfall shows exactly which features the attacker exploited."

**[Generate SHAP plot]**

> "See how velocity_error dominates at 75% importance? The attacker learned to manipulate this single feature to flip the model's decision. That's actionable intelligence for model improvement."

---

### 5. THE BUSINESS VALUE (2:00-2:30)

**[Switch to Report Generator - generate PDF]**

> "Everything gets packaged into an SR 11-7 compliant report - the same format regulators expect from major banks."

**[Show generated PDF/HTML report]**

> "Evasion rate, feature vulnerabilities, specific recommendations - all documented for your model risk management team."

**[Quick flash of cost_analysis notebook]**

> "And the Monte Carlo analysis quantifies the business impact: reducing evasion rate from 20% to 5% saves an estimated $200,000 annually in fraud losses."

---

### 6. CALL TO ACTION (2:30-3:00)

**[Back to Overview page showing architecture diagram]**

> "This entire system runs locally - XGBoost for detection, Ollama for attacks, SHAP for explanations. Total cloud cost: zero dollars."

> "At Fidelity, I managed models protecting billions in assets. This is how I'd automate adversarial testing for any ML system in production."

> "The code is linked in the comments. If you're building fraud detection, credit scoring, or any high-stakes ML - this approach finds vulnerabilities before they cost you money."

**[End with GitHub link overlay]**

---

## Key Talking Points to Emphasize

1. **Real Data**: "We attack REAL fraud transactions, not synthetic data"
2. **Local LLM**: "Everything runs locally - no data exposure"
3. **Explainability**: "SHAP shows exactly HOW attacks succeed"
4. **Compliance**: "SR 11-7 ready reports for regulators"
5. **Cost**: "Zero cloud cost - fully local stack"

---

## Technical Details to Mention (If Asked)

| Component | Technology |
|-----------|------------|
| Defender Model | XGBoost (AUC: 0.9989) |
| Attacker Agent | Ollama + llama3.2:3b |
| Explainability | SHAP TreeExplainer |
| Dashboard | Streamlit |
| Data | PaySim (6.3M transactions, 8,213 fraud cases) |
| Reports | Jinja2 + pdfkit |

---

## Screen Recording Checklist

- [ ] Ollama running in background
- [ ] Streamlit app loaded (`streamlit run app.py`)
- [ ] Model trained and loaded
- [ ] Clean browser window (no bookmarks bar)
- [ ] Terminal hidden or minimized
- [ ] 1080p or higher resolution
- [ ] Quiet recording environment

---

## Backup Talking Points (If Demo Fails)

If LLM is slow:
> "The LLM is reasoning about the transaction - trying different perturbation strategies within our constraints."

If attack fails:
> "Not every attack succeeds - that's actually good! It means our model has some robustness. But finding even one evasion is valuable intelligence."

If SHAP plot takes time:
> "SHAP is computing feature attributions across the model's decision trees - this is the same explainability used by major banks for regulatory compliance."

---

*Script updated: January 17, 2026*
*Based on actual implementation results*
