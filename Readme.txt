================================================================================
Diabetes Prediction using Bayesian Networks
================================================================================

This project implements a Bayesian Network model to predict diabetes using the 
Pima Indians Diabetes Dataset. It combines structure learning algorithms with 
expert medical knowledge to uncover causal relationships between health indicators 
and diabetes outcomes.

--------------------------------------------------------------------------------
1. Requirements
--------------------------------------------------------------------------------

Python Version:
- Python 3.7 or higher

Required Python Packages:
- pandas==1.3.3
- numpy==1.21.2
- matplotlib==3.4.3
- seaborn==0.11.2
- scikit-learn==0.24.2
- pgmpy==0.1.14
- networkx==2.6.3
- missingno==0.5.0
- warnings (built-in)

--------------------------------------------------------------------------------
2. Installation Guide
--------------------------------------------------------------------------------

Step 1: Upgrade pip
    pip install --upgrade pip

Step 2: Install required packages
    pip install pandas==1.3.3 numpy==1.21.2
    pip install matplotlib==3.4.3 seaborn==0.11.2
    pip install scikit-learn==0.24.2
    pip install pgmpy==0.1.14
    pip install networkx==2.6.3
    pip install missingno==0.5.0

--------------------------------------------------------------------------------
3. File Structure
--------------------------------------------------------------------------------

Required Files:
- diabetes.csv                  : Pima Indians Diabetes Dataset
- diabetes_bayesian_network.py : Main Python script

Generated Files:
- bn_model.pkl                  : Trained Bayesian Network model
- pima_predictions.csv          : Test predictions
- pima_metrics.csv              : Performance metrics

Generated Visualizations:
- 01_target_distribution.png
- 02_feature_distributions.png
- 03_boxplots_by_outcome.png
- 04_correlation_matrix.png
- 04a_target_correlation.png
- 05_feature_importance.png
- 06_train_test_split.png
- 07_bayesian_network_structure.png
- 08_threshold_optimization.png
- 09_confusion_matrix.png
- 10_roc_curve.png
- 11_model_comparison.png

--------------------------------------------------------------------------------
4. Features
--------------------------------------------------------------------------------

- Data Quality Analysis: Missing value detection, duplicate checking, zero-value handling
- Feature Discretization: Medical threshold-based binning for Bayesian Networks
- Structure Learning: Hill Climb Search with BIC scoring and expert knowledge
- Model Evaluation: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
- Threshold Optimization: Optimal classification threshold selection
- Model Comparison: Logistic Regression, Random Forest, SVM
- Visualization: Network diagrams, correlation matrices, ROC curves, performance plots

--------------------------------------------------------------------------------
5. How to Run
--------------------------------------------------------------------------------

1. Clone or download this repository
2. Navigate to the project directory
3. Ensure 'diabetes.csv' is present
4. Run the script:
    python diabetes_bayesian_network.py

--------------------------------------------------------------------------------

================================================================================
