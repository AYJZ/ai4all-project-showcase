# Sounding Out Parkinson's Disease

Built an end-to-end pipeline in the AI4ALL Ignite accelerator program that turns one sustained “ah” sound into a high-accuracy Parkinson’s disease screen, applying feature selection, class balancing, and machine learning modeling skills.

<br>

## Problem Statement <!--- do not change this line -->

Parkinson’s disease is often diagnosed **after motor symptoms appear**, losing valuable early-treatment opportunities.  
However, subtle **voice changes occur in up to 90% of PD patients years earlier**.  
We asked:  

*Can short, low-cost voice recordings be used for accurate, transparent, and early Parkinson’s screening?*  

<br>

## Data Sources <!--- do not change this line -->

**UCI Parkinson’s Disease Classification Dataset** (Sakar et al., 2018)  
  <https://archive.ics.uci.edu/dataset/470/parkinson+s+disease+classification>

<br>

## Methodologies <!--- do not change this line -->

### 1. **Speaker-Aware Aggregation**  
- Combined multiple recordings per speaker into one row.  
- Strategies: `mean`, `mean+std`, `median+IQR`.  
- Prevents **speaker leakage** across train/test splits.  

### 2. **Feature Engineering**  
- Started with **~755 acoustic features**.  
- Applied multiple selectors:  
  - Variance + correlation pruning  
  - Statistical ranking (Pearson, ANOVA F-score, Mutual Information)  
  - Model-based (Logistic Regression L1/L2, Random Forest, XGBoost)  
- Final feature set: **top 20 speaker-level predictors**.  

### 3. **Group-Aware CV & Model Search**  
- Train/test split and CV folds done **by speaker ID** (no leakage).  
- Pipelines tested with:  
  - **Models**: Logistic Regression, Random Forest, XGBoost  
  - **Samplers**: None, SMOTE, SMOTE-ENN  
  - **Selectors**: 6 feature-selection recipes  
- **GridSearchCV with GroupKFold (5-fold)** tuned hyperparameters.  
- Optimized primarily for **PD Recall (Sensitivity)** to minimize false negatives.  

### 4. **Threshold Tuning**  
- Instead of default 0.50, tuned decision thresholds.  
- Used a **weighted utility function (α=0.7 Recall, β=0.3 Specificity)** with a **Precision floor** to balance catching PD patients vs. over-flagging controls.  

### 5. **Explainability**  
- Interpreted key acoustic features with **SHAP values**.  
- Keeps the model clinician-friendly.  

<br>

## Key Results <!--- do not change this line -->

- **Best model**: XGBoost with tree-based feature selection.  
- **Cross-Validation (GroupKFold)** prioritized PD recall:  
  - Recall (PD) ≈ 0.90+
  - ROC-AUC ≈ 0.75–0.80 (expected variability due to small dataset)
- **Test Set** (with tuned threshold, α=0.7 recall weight):
  - Accuracy = 0.83
  - Recall (PD) = 0.87 (high sensitivity)
  - Precision (PD) = 0.89
- **Trade-off**: Missing a PD patient (false negative) is worse than flagging a healthy person (false positive).  

<br>

## Technologies Used <!--- do not change this line -->

- Python 3.10  
- pandas, NumPy, scikit-learn  
- imbalanced-learn (SMOTE-ENN)  
- XGBoost 1.7  
- SHAP for model explainability  
- Google Colab notebooks

<br>

## Authors <!--- do not change this line -->

*This project was completed in collaboration with:*
- *Sophia Tang ([sstang@bu.edu](mailto:sstang@bu.edu))*
- *Fernando Peralta ([fperalta0248@gmail.com](mailto:fperalta0248@gmail.com))*

*Special Thanks to:*
- Eliza Salamon (eliza@ai-4-all[dot]org)
- Orvil Escalante (sc_orvil_e@ai-4-all[dot]org)
- Tatiane Wu Li (wu@uni[dot]minerva[dot]edu)
