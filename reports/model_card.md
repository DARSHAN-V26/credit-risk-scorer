# Model Card — Credit Risk Scorer

## Model Overview
- **Model type:** LightGBM binary classifier
- **Version:** V2 (with behavioral features)
- **Task:** Predict probability of loan default for thin-file borrowers
- **Owner:** Darshan V, IIT Hyderabad

## Intended Use
- **Primary use:** Assist lenders in evaluating loan applications from 
  borrowers with limited traditional credit history
- **Intended users:** Financial institutions, fintech lenders
- **Out-of-scope:** Should not be used as the sole basis for loan 
  rejection without human review

## Training Data
- **Source:** Home Credit Default Risk dataset (Kaggle)
- **Size:** 307,511 loan applications
- **Features:** 111 features across 6 tables (application data + 
  bureau history + installment payments + credit card behavior + 
  previous applications)
- **Class balance:** 8% default rate (severely imbalanced)
- **Imbalance strategy:** class_weight='balanced' in LightGBM

## Model Performance
| Metric | Value |
|--------|-------|
| ROC-AUC | 0.7755 |
| Average Precision | 0.2724 |
| Best F1 Score | 0.2866 (threshold 0.50) |

### Threshold Analysis (recommended threshold: 0.40)
| Metric | Value |
|--------|-------|
| Defaulters caught (Recall) | 79.9% |
| Good customers wrongly rejected | 22,935 |
| FP/TP ratio | 5.8 |

## Key Predictors (SHAP Analysis)
1. EXT_SOURCE_2 — external credit score (strongest signal)
2. EXT_SOURCE_3 — external credit score
3. EXT_SOURCE_1 — external credit score
4. instal_late_payment_ratio — fraction of late installment payments
5. bureau_debt_to_credit_ratio — debt burden from bureau history

## Limitations
- **False positive rate is high:** At any practical threshold, the 
  model wrongly rejects a significant number of good customers due 
  to feature space overlap between defaulters and repayers
- **Static snapshot:** Application data reflects a single point in 
  time — recent financial changes are not captured
- **Geographic scope:** Trained on data from a specific lending 
  market — may not generalise to other regions
- **EXT_SOURCE opacity:** The three most predictive features are 
  external scores whose exact methodology is undisclosed

## Fairness Considerations
- CODE_GENDER appeared as the 4th most impactful SHAP feature
- Model should be audited for disparate approval rates across 
  demographic groups before production deployment
- Threshold should be calibrated separately per demographic segment 
  if legally required

## Improvements Over Baseline
| | V1 (static) | V2 (behavioral) |
|---|---|---|
| Features | 77 | 111 |
| ROC-AUC | 0.7597 | 0.7755 |
| Average Precision | 0.2488 | 0.2724 |
| FP at threshold 0.40 | 24,687 | 22,935 |

## Future Improvements
- Hyperparameter tuning with Optuna (estimated +1-2% ROC-AUC)
- Fairness-aware modeling to reduce gender disparity
- Incorporation of transaction-level time series features
- Probability calibration for better risk tier assignment