# Baseline Cross-Validation

This report summarizes 5-fold stratified cross-validation on the training split.

```csv
model,n_splits,accuracy_mean,accuracy_std,precision_mean,recall_mean,f1_score_mean,attack_f1_score_mean
decision_tree,5,0.9623,0.0004,0.9442,0.9942,0.9686,0.9529
random_forest,5,0.9618,0.0005,0.9421,0.9958,0.9682,0.9521
linear_svm,5,0.912,0.0003,0.8826,0.9796,0.9286,0.8852
```