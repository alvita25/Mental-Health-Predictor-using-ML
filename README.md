README — Mental-Health-Predictor-using-ML
========================================

Project overview
----------------
The Mental Health Chatbot is an intelligent conversational system designed to provide emotional support and assist users in understanding their mental well-being through natural language interaction. It leverages Machine Learning (ML) models to interpret user input, detect underlying emotional states (like stress, anxiety, or sadness), and respond empathetically in real-time.

The chatbot integrates five separate ML classifiers —
- XGBoost
- LightGBM
- Random Forest
- Naive Bayes
- Logistic Regression
Each trained to predict the emotional category of user messages based on text data and linguistic features.
Among them, XGBoost serves as the primary model due to its superior accuracy, generalization capability, and robustness to complex, non-linear relationships in text-based features.

Prerequisites
-------------
- Python 3.8+ (3.10 recommended)
- git
- pip
- Jupyter Notebook or JupyterLab

If you use conda, you may create a conda environment instead of virtualenv.

Local setup (step-by-step)
-------------------------
1. Clone the repository
   - git clone https://github.com/alvita25/Mental-Health-Predictor-using-ML.git
   - cd Mental-Health-Predictor-using-ML

2. Create and activate a virtual environment (venv example)
   - python -m venv .venv
   - On macOS / Linux: source .venv/bin/activate
   - On Windows (PowerShell): .venv\Scripts\Activate.ps1

3. Install dependencies
   - If a requirements.txt exists:
     - pip install -r requirements.txt

Running the notebooks (interactive)
-----------------------------------
Open and run the notebooks in Jupyter Notebook or JupyterLab. All four notebooks need to be executed to ensure models are created/available as expected.

1. Start Jupyter:
   - jupyter notebook
   - or jupyter lab

2. Open and run notebooks in order (replace placeholders with actual filenames):
   - mentalHealth_Lightgbm.ipynb
   - MentalHealth_LogisticRegression.ipynb
   - MentalHealth_NaiveBayes.ipynb
   - MentalHealth_XGB.ipynb

3. For each notebook, use Kernel → Restart & Run All (or run cells sequentially) so all preprocessing and model-save steps execute.

Notes about models
------------------
- I lost the random forest model notebook. You can use the other model code as reference to create a random forest model. Sorry for the inconvenience.

---

## Run
```bash
python app.py
```

---

Screenshots
------------------
<img width="1899" height="913" alt="image" src="https://github.com/user-attachments/assets/d8115f0d-310c-468b-81f1-72ea79664240" align="center" />
<img width="1892" height="912" alt="image" src="https://github.com/user-attachments/assets/a41bbcd0-588d-4b3f-9ea5-94cc51821419" slign="center"/>


Contributing
------------
- Feel free to open issues or submit pull requests.
- Before contributing, run the notebooks locally and ensure they execute end-to-end.
