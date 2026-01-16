# Adversarial Fraud Detection System - Implementation Roles & Best Practices

---
## CHECKPOINT - January 15, 2026
---

### Current State: Phase 1 Code Complete

**What's Done:**
- Virtual environment created: `.fraud-venv/`
- All dependencies installed (Python 3.13 compatible)
- Phase 1 code files created:
  - `data/paysim_download.py` - Data loader
  - `src/models/transaction.py` - Pydantic models
  - `src/models/defender.py` - XGBoost classifier
  - `tests/test_data.py` - Data tests
  - `tests/test_models.py` - Model tests (12/12 passing)

**What's Pending:**
- Kaggle dataset download (needs terms acceptance)
- Model training (HUMAN task)
- Phase 2-4 code files

---

### Resume Instructions

When resuming, tell Claude Code:
> "Resume from checkpoint - I've completed the HUMAN tasks. Continue with CC tasks."

---

### Next Steps (Task Ownership)

| Step | Owner | Task | Command/Action |
|------|-------|------|----------------|
| 1 | **[HUMAN]** | Accept PaySim dataset terms | Visit https://www.kaggle.com/datasets/ntnu-testimon/paysim1 and click Download |
| 2 | **[HUMAN]** | Verify kaggle.json exists | Check `C:\Users\prana\.kaggle\kaggle.json` |
| 3 | **[CC]** | Download dataset | Run `python data/paysim_download.py` |
| 4 | **[CC]** | Verify data tests pass | Run `pytest tests/test_data.py` |
| 5 | **[HUMAN]** | Train defender model | Run `python -m src.models.defender --sample-size 100000` |
| 6 | **[CC]** | Verify model file exists | Check `models/defender.pkl` |
| 7 | **[CC]** | Create Phase 2 (attacker) | Write `src/agents/attacker.py`, `src/red_team.py` |
| 8 | **[HUMAN]** | Start Ollama service | Run `ollama serve` (if not auto-started) |
| 9 | **[CC]** | Test LLM connectivity | Run attacker tests |
| 10 | **[CC]** | Create Phase 3 (SHAP, reports) | Write `src/explainer.py`, `reports/` |
| 11 | **[HUMAN]** | Install wkhtmltopdf | Download from wkhtmltopdf.org |
| 12 | **[CC]** | Create Phase 4 (dashboard) | Write `app.py`, notebook |
| 13 | **[HUMAN]** | Test Streamlit UI | Run `streamlit run app.py` |
| 14 | **[HUMAN]** | Record demo video | Follow LinkedIn script |

---

## Overview
This document categorizes each implementation step by who should execute it (Claude Code vs Human) and provides best practices for effective collaboration.

---

## Your Current Status
- **Ollama**: Installed, llama3.2:3b pulled
- **Kaggle**: Account exists, needs terms acceptance for PaySim dataset
- **Experience**: Experienced with ML/DS - minimal explanations needed

---

## Legend
- **[CC]** = Claude Code can do this autonomously
- **[HUMAN]** = Human must do this (setup, accounts, manual verification)
- **[COLLAB]** = Collaborative - Claude Code does the work, Human verifies/approves

---

## IMPORTANT: Ollama is NOT a Python Package

Ollama is a **standalone application** that runs as a system service on your machine. Python connects to it via HTTP API. This means:

1. **Install Ollama system-wide** (not in any venv)
2. **Run Ollama as a background service** (it listens on localhost:11434)
3. **Python uses the `openai` package** to communicate with Ollama's OpenAI-compatible API

```
[Your Python Code] --HTTP--> [Ollama Service:11434] ---> [llama3.2:3b model]
     (in venv)                   (system-level)
```

---

## Pre-Implementation Setup (Human Tasks)

### Step 1: Install Ollama [HUMAN]
```bash
# Download from: https://ollama.com/download/windows
# Run the installer
# After installation, Ollama runs as a background service automatically
```

### Step 2: Pull the LLM Model [HUMAN]
```bash
# Open a new terminal (not in any venv)
ollama pull llama3.2:3b

# Verify it's running
ollama list
```

### Step 3: Setup Kaggle API [HUMAN]
1. Go to https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New Token" - downloads `kaggle.json`
4. Place it in: `C:\Users\<YourUsername>\.kaggle\kaggle.json`

---

## Phase 1: Data & Defender Model

### Environment Setup
| Task | Owner | Notes |
|------|-------|-------|
| Install Ollama (system-level) | **[HUMAN]** | Download from ollama.com, runs as Windows service |
| Run `ollama pull llama3.2:3b` | **[HUMAN]** | In any terminal (not venv-related) |
| Create Python virtual environment | **[CC]** | `python -m venv venv` |
| Activate virtual environment | **[HUMAN]** | `venv\Scripts\activate` (must be done in your terminal) |
| Install requirements.txt | **[CC]** | After human activates venv |
| Setup Kaggle API credentials | **[HUMAN]** | Download kaggle.json to ~/.kaggle/ |

### Task 1.1: Data Loading
| Task | Owner | Notes |
|------|-------|-------|
| Create `data/paysim_download.py` | **[CC]** | Write download and preprocessing code |
| Create `tests/test_data.py` | **[CC]** | Write test case |
| Download PaySim dataset | **[COLLAB]** | CC runs script, Human ensures Kaggle auth works |
| Verify data loads correctly | **[CC]** | Run tests |

### Task 1.2: Defender Model
| Task | Owner | Notes |
|------|-------|-------|
| Create `src/models/defender.py` | **[CC]** | Write XGBoost model class |
| Create test for defender | **[CC]** | Write test case |
| Train and save model | **[HUMAN]** | Run training script manually |
| Verify AUC > 0.95 | **[COLLAB]** | Human trains, CC verifies metrics |

---

## Phase 2: Attacker Agent (Days 3-5)

### Task 2.1: Transaction Data Model
| Task | Owner | Notes |
|------|-------|-------|
| Create `src/models/transaction.py` | **[CC]** | Pydantic models with validation |
| Write validation tests | **[CC]** | Test constraints |

### Task 2.2: Attacker Agent
| Task | Owner | Notes |
|------|-------|-------|
| Verify Ollama is running | **[HUMAN]** | Must start Ollama server |
| Create `src/agents/attacker.py` | **[CC]** | LLM perturbation engine |
| Write constraint tests | **[CC]** | Test 10% amount, 2-hour constraints |
| Test LLM connectivity | **[COLLAB]** | CC runs test, Human ensures Ollama running |

### Task 2.3: Red Team Loop
| Task | Owner | Notes |
|------|-------|-------|
| Create `src/red_team.py` | **[CC]** | Main adversarial loop |
| Write integration tests | **[CC]** | End-to-end test |
| Run and verify attacks work | **[COLLAB]** | CC runs, Human observes results |

---

## Phase 3: Explainability & Reporting (Days 6-8)

### Task 3.1: SHAP Integration
| Task | Owner | Notes |
|------|-------|-------|
| Create `src/explainer.py` | **[CC]** | SHAP waterfall plots |
| Test visualization output | **[COLLAB]** | CC generates, Human reviews plots |

### Task 3.2: SR 11-7 Report Generator
| Task | Owner | Notes |
|------|-------|-------|
| Install wkhtmltopdf | **[HUMAN]** | Required for pdfkit, download from wkhtmltopdf.org |
| Create `reports/template.html` | **[CC]** | Jinja2 HTML template |
| Create `reports/generator.py` | **[CC]** | PDF generation code |
| Test PDF output | **[COLLAB]** | CC generates, Human opens/reviews PDF |

---

## Phase 4: Dashboard & Analysis (Days 9-10)

### Task 4.1: Streamlit App
| Task | Owner | Notes |
|------|-------|-------|
| Create `app.py` | **[CC]** | Full Streamlit dashboard |
| Launch Streamlit | **[CC]** | `streamlit run app.py` |
| Test UI interactions | **[HUMAN]** | Click through UI, verify functionality |
| Record demo video | **[HUMAN]** | LinkedIn demo script |

### Task 4.2: Cost Analysis Notebook
| Task | Owner | Notes |
|------|-------|-------|
| Create `notebooks/cost_analysis.ipynb` | **[CC]** | Monte Carlo simulation |
| Run notebook cells | **[CC]** | Execute analysis |
| Interpret business insights | **[HUMAN]** | Review P&L optimization results |

---

## Testing & Verification

| Task | Owner | Notes |
|------|-------|-------|
| Create `pytest.ini` | **[CC]** | Test configuration |
| Create `tests/test_integration.py` | **[CC]** | Full pipeline test |
| Run `pytest` | **[CC]** | Execute all tests |
| Verify all success criteria | **[COLLAB]** | CC runs checks, Human confirms |

---

## Summary: Task Distribution

| Category | Count | Examples |
|----------|-------|----------|
| **Claude Code [CC]** | ~70% | Writing all Python code, tests, configs |
| **Human [HUMAN]** | ~15% | Ollama setup, Kaggle auth, wkhtmltopdf, UI testing, model training |
| **Collaborative [COLLAB]** | ~15% | Running tests, reviewing outputs, verifying quality |

---

# Best Practices for Using Claude Code Effectively

## 1. Project Setup Best Practices

### Before Starting
- **Create a clear file structure first**: Tell Claude Code the desired directory layout upfront
- **Have prerequisites ready**: Install Ollama, setup Kaggle credentials, activate venv BEFORE asking CC to write code
- **Use `.claude/` directory**: Store project-specific prompts or context

### Working Directory
```bash
# Always work from the project root
cd C:\projects\adhoc_projects\llm_usecases\adversarial_fraud_detection
```

## 2. Effective Prompting Strategies

### Be Specific and Incremental
```
# GOOD: Specific task
"Create src/models/defender.py with the XGBoost DefenderModel class as specified in the implementation guide"

# BAD: Vague request
"Build the fraud detection system"
```

### Reference Existing Files
```
# GOOD: Reference context
"Update src/agents/attacker.py to handle JSON parsing errors, similar to how we handle them in src/red_team.py"

# BAD: No context
"Fix the JSON error"
```

### Break Large Tasks into Phases
```
Phase 1: "Create the data loading module and test it works"
Phase 2: "Create the defender model and verify AUC"
Phase 3: "Create the attacker agent and test LLM connectivity"
```

## 3. Code Review Workflow

### After Claude Code Writes Code
1. **Read the output**: Ask CC to show you what it wrote
2. **Run tests immediately**: "Run pytest on the new code"
3. **Ask for explanations**: "Explain how the perturbation loop works"
4. **Request modifications**: "Add type hints to the Transaction class"

### Use Todo Lists
Claude Code tracks tasks automatically. You can say:
- "What's left to do?"
- "Mark the defender model as complete"
- "Add a task to implement SHAP visualization"

## 4. Debugging with Claude Code

### When Tests Fail
```
"The test_attacker_respects_constraints test is failing. Read the error output and fix the issue."
```

### When Models Don't Converge
```
"The XGBoost model AUC is only 0.82. Analyze the training data distribution and suggest feature engineering improvements."
```

### When LLM Outputs are Invalid
```
"The attacker agent is returning invalid JSON. Add error handling and retry logic."
```

## 5. File Organization Tips

### Let Claude Code Create Structure
```
"Create the complete file structure for the fraud-red-team project as specified in the implementation guide"
```

### Use Consistent Imports
```
"Ensure all imports follow the pattern: from src.models.defender import DefenderModel"
```

## 6. Testing Best Practices

### Write Tests First (TDD)
```
"Write the test for the DefenderModel.train() method first, then implement the method to pass the test"
```

### Run Tests Frequently
```
"After every file change, run pytest to ensure nothing broke"
```

## 7. When to Involve Human vs Claude Code

### Human Should Handle:
- Installing system-level dependencies (Ollama, wkhtmltopdf)
- Managing API credentials (Kaggle, any cloud services)
- Visual verification of plots and dashboards
- Business decision-making (thresholds, costs)
- Recording demo videos
- Training ML models (when preferred)

### Claude Code Excels At:
- Writing boilerplate code quickly
- Creating test files
- Refactoring existing code
- Debugging error messages
- Generating documentation from code
- Running batch operations

## 8. Session Management

### Long Sessions
- Save progress frequently: "Commit current changes with message 'Phase 1 complete'"
- Take breaks: Claude Code maintains context but human attention matters

### Resuming Work
```
"Read the current state of src/red_team.py and remind me what's left to implement"
```

## 9. Common Pitfalls to Avoid

1. **Don't skip prerequisites**: Ensure Ollama is running BEFORE testing attacker
2. **Don't ignore test failures**: Fix them immediately
3. **Don't over-engineer**: Start simple, iterate
4. **Don't forget validation**: Human should verify generated outputs

## 10. Recommended Workflow for This Project

```
Step 1: [HUMAN] Install Ollama, setup Kaggle, create venv
Step 2: [CC] Create all file structures and requirements.txt
Step 3: [HUMAN] Activate venv, pip install
Step 4: [CC] Implement Phase 1 (data + defender)
Step 5: [HUMAN] Train model manually
Step 6: [CC] Implement Phase 2 (attacker agent)
Step 7: [HUMAN] Start Ollama server
Step 8: [COLLAB] Test LLM connectivity
Step 9: [CC] Implement Phase 3 (SHAP + reports)
Step 10: [HUMAN] Install wkhtmltopdf
Step 11: [CC] Implement Phase 4 (dashboard)
Step 12: [HUMAN] Test UI, record demo
```

---

## Files to be Created (by Claude Code)

1. `data/paysim_download.py` - DONE
2. `src/__init__.py` - DONE
3. `src/models/__init__.py` - DONE
4. `src/models/defender.py` - DONE
5. `src/models/transaction.py` - DONE
6. `src/agents/__init__.py` - DONE
7. `src/agents/attacker.py` - PENDING
8. `src/red_team.py` - PENDING
9. `src/explainer.py` - PENDING
10. `reports/template.html` - PENDING
11. `reports/generator.py` - PENDING
12. `notebooks/cost_analysis.ipynb` - PENDING
13. `app.py` - PENDING
14. `requirements.txt` - DONE
15. `pytest.ini` - DONE
16. `tests/__init__.py` - DONE
17. `tests/test_data.py` - DONE
18. `tests/test_integration.py` - PENDING

---

## Verification Checklist

Before considering each phase complete:

- [x] All Phase 1 Python files created and syntactically valid
- [x] Phase 1 tests pass (`pytest` - 12/12)
- [ ] Dataset downloaded from Kaggle
- [ ] Model trained with AUC > 0.95
- [ ] Attacker finds at least 1 evasion
- [ ] SHAP visualizations render correctly
- [ ] PDF report generates successfully
- [ ] Streamlit app loads in < 3 seconds
- [ ] Total cloud costs < $10 (Ollama is free/local)
