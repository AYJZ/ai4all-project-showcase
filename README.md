# Sounding Out Parkinson's Disease

Built an end-to-end pipeline in the AI4ALL Ignite accelerator program that turns one sustained “ah” sound into a high-accuracy Parkinson’s disease screen, applying feature selection, class balancing, and machine learning modeling skills.


## Problem Statement <!--- do not change this line -->

Parkinson’s disease is often diagnosed only after motor symptoms appear, missing a crucial early-treatment window.  
Because subtle voice changes emerge in ~90 % of patients years earlier, we asked: **Can we detect PD from short voice recordings with transparent, low-cost machine learning?**


## Key Results <!--- do not change this line -->

1. Aggregated 756 vowel recordings into **252 unique speakers** (188 PD / 64 control) to avoid data leakage.  
2. Reduced **755 acoustic features → top 20** via variance + correlation filters and Pearson ranking.  
3. Balanced classes with **SMOTE-ENN**, selected ML model based on testing, and trained an XGBoost model achieving:  
   - **Accuracy = 0.9664**  
   - **ROC-AUC = 0.9853**  


## Methodologies <!--- do not change this line -->

* Speaker-aware aggregation (mean per ID) to eliminate cross-sample leakage.  
* Two-step feature pruning: low-variance & high-correlation removal, then top-20 absolute Pearson r with class.  
* Tested multiple machine learning models including Logistic Regression, KNN, MLP, Random Forest, and XGBoost.  
* **SMOTE-ENN** applied **only on training folds** for balanced, clean samples.  
* Hyper-parameter-tuned **XGBoost** model.  
* Interpreted feature importance and generated SHAP explanations to keep the model clinician-friendly.


## Data Sources <!--- do not change this line -->

**UCI Parkinson’s Disease Classification Dataset** (Sakar et al., 2018)  
  <https://archive.ics.uci.edu/dataset/470/parkinson+s+disease+classification>

## Technologies Used <!--- do not change this line -->

- Python 3.10  
- pandas, NumPy, scikit-learn  
- imbalanced-learn (SMOTE-ENN)  
- XGBoost 1.7  
- SHAP for model explainability  
- Google Colab notebooks

## Authors <!--- do not change this line -->

*This project was completed in collaboration with:*
- *Sophia Tang ([sstang@bu.edu](mailto:sstang@bu.edu))*
- *Fernando Peralta ([fperalta0248@gmail.com](mailto:fperalta0248@gmail.com))*

*Special Thanks to:*
- Eliza Salamon (eliza@ai-4-all[dot]org)
- Orvil Escalante (sc_orvil_e@ai-4-all[dot]org)
- Tatiane Wu Li (wu@uni[dot]minerva[dot]edu)
