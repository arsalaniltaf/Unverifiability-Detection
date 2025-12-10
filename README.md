Unverifiability Detector (Weighted TF-IDF + Numeric Features + Optuna)
Author: Muhammad Arsalan

Dataset Information (from ARTA / Requirement Testability Dataset)
This project can also utilize the public dataset released with the research paper “Requirement Testability Measurement Based on Requirement Smells”. The dataset—published on Zenodo—contains a curated collection of real-world software requirements annotated with various requirement smells, including ambiguity, unverifiability, and other quality issues. The dataset was produced as part of the Automatic Requirement Testability Analyzer (ARTA) initiative, a web application designed to support requirement engineering quality assurance and assist analysts in evaluating and managing the testability of their requirements. This publicly available dataset supports reproducible research and enables benchmarking for tools focused on requirement quality analysis.

Zakeri-Nasrabadi, M., & Parsa, S. (2024). Natural Language Requirements Testability Measurement Based on Requirement Smells. 

Zenodo. https://doi.org/10.5281/zenodo.4266727

This project provides a complete machine learning pipeline for detecting
UNVERIFIABILITY in natural-language software requirements. It uses weighted
word- and character-level TF-IDF features, handcrafted linguistic features,
a RandomForest classifier, Optuna hyperparameter optimization, probability
calibration, and optimized threshold selection.

------------------------------------------------------------
1. FEATURES
------------------------------------------------------------
- Interactive dataset loading (user enters CSV file path)
- Word-level TF-IDF (1–3 grams)
- Character-level TF-IDF (3–6 n-grams)
- Handcrafted linguistic/numeric features:
    • sentence length
    • number of words
    • modals
    • vague terms
    • comparatives & superlatives
    • long sentence indicator
    • average word length
- Per-feature scaling and weighting
- RandomForest classifier with balanced class weights
- Hyperparameter tuning using Optuna
- Probability calibration and F1-based threshold optimization
- Saves final models and evaluation results

------------------------------------------------------------
2. PROJECT STRUCTURE
------------------------------------------------------------
project_folder/
│
├── unverifiability_detector.py       (main script)
├── dataset.csv (or other CSV file)   (script will ask for its path)
└── unverifiability_results/          (created automatically)

------------------------------------------------------------
3. INSTALLATION
------------------------------------------------------------
Required packages:
    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    joblib
    optuna

Install using:
    pip install pandas numpy matplotlib seaborn scikit-learn joblib optuna

Python version: 3.8 or above recommended.

------------------------------------------------------------
4. HOW THE PIPELINE WORKS
------------------------------------------------------------
1. Data Loading:
   - Script prompts user for CSV path.
   - Set text column as (Requiremnt) and label column as (Unverifiability).
   - Normalizes column names and cleans data.

2. Feature Engineering:
   - Word TF-IDF: unigrams, bigrams, trigrams
   - Character TF-IDF: char-ngrams 3–6
   - Numeric/Linguistic features extracted from text
   - Each feature block weighted using a custom transformer

3. Optuna Optimization:
   - Searches best RandomForest parameters:
       n_estimators
       max_depth
       min_samples_leaf
   - Objective: maximize weighted F1-score

4. Final Training & Calibration:
   - Trains final model using best parameters
   - Uses CalibratedClassifierCV to calibrate probabilities
   - Finds optimal decision threshold based on precision–recall F1 curve

5. Saving Results:
   - Saves pipeline model: model_optuna.joblib
   - Saves calibrated model: calibrated_optuna.joblib
   - Saves optimized evaluation metrics: unverifiability_results.json

------------------------------------------------------------
5. USAGE
------------------------------------------------------------

MACOS:
1. Open Terminal.
2. Navigate to project folder:
       cd /path/to/project_folder
3. Run the script:
       python3 unverifiability_detector.py
4. Follow prompts:
       - Enter dataset CSV path
       - Enter save directory (or press Enter for default)

WINDOWS:
1. Open Command Prompt or PowerShell.
2. Navigate to project folder:
       cd C:\path\to\project_folder
3. Run the script:
       python unverifiability_detector.py
4. Follow prompts as above.

------------------------------------------------------------
6. OUTPUT FILES
------------------------------------------------------------
Saved to selected directory (default: ./unverifiability_results):

   model_optuna.joblib          - trained RandomForest pipeline
   calibrated_optuna.joblib     - calibrated classifier
   unverifiability_results.json - optimized evaluation metrics and threshold

Console output includes:
   - Classification report
   - Confusion matrix
   - Optuna optimization logs

------------------------------------------------------------
7. NOTES
------------------------------------------------------------
- Label column must be numeric (0/1).
- Rename dataset columns if auto-detection fails.
- Reduce Optuna trials to speed up training.
- Feature weights can be adjusted inside build_pipeline().
- Large datasets may require more RAM for TF-IDF matrices.

------------------------------------------------------------
8. CONTACT
------------------------------------------------------------
For questions, improvements, or issues, contact the author or open a GitHub issue.

