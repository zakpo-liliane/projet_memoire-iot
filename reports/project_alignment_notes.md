# Project Alignment Notes

This note summarizes the code changes made to align the IDS project more closely with the thesis plan.

## What was aligned

- Replaced hard-coded project paths with repository-relative paths in the data preparation and baseline scripts.
- Added a hybrid `cnn_lstm_attention` architecture to the deep learning pipeline.
- Added optional class-imbalance handling through `class_weight` and an optional `focal_loss` for supervised deep learning models.
- Added a 5-fold stratified cross-validation workflow for the classical baselines.
- Added a statistical analysis pipeline for McNemar, ANOVA, and Wilcoxon tests.
- Added a basic explainability report for the final decision tree model using feature and permutation importance.
- Added pruning selection for the final decision tree and TFLite quantization export for deep models.
- Updated the evaluation pipeline so it can automatically include the attention model when its artifacts are available.

## Files updated

- `src/merge_dataset.py`
- `src/clean_dataset.py`
- `src/prepare_features.py`
- `src/split_dataset.py`
- `src/run_eda.py`
- `src/train_baselines.py`
- `src/train_deep_models.py`
- `src/tune_deep_models.py`
- `src/evaluate_models_test.py`
- `src/compare_models_final.py`
- `src/generate_result_figures.py`
- `src/cross_validate_baselines.py`
- `src/statistical_analysis.py`
- `src/explain_best_model.py`
- `src/export_edge_artifacts.py`

## What should be regenerated

The current CSV/PNG reports already present in `reports/` reflect the previous execution of the pipeline.  
To fully refresh the project artifacts after the code alignment, rerun the main scripts in this order:

1. `python src/merge_dataset.py`
2. `python src/clean_dataset.py`
3. `python src/prepare_features.py`
4. `python src/split_dataset.py`
5. `python src/run_eda.py`
6. `python src/train_baselines.py`
7. `python src/cross_validate_baselines.py`
8. `python src/train_deep_models.py --balance-method smote --loss-function binary_crossentropy`
9. `python src/tune_deep_models.py --balance-method smote --loss-function binary_crossentropy`
10. `python src/evaluate_models_test.py`
11. `python src/statistical_analysis.py`
12. `python src/explain_best_model.py`
13. `python src/compare_models_final.py`
14. `python src/save_best_model.py`
15. `python src/explain_best_model.py`
16. `python src/export_edge_artifacts.py`
17. `python src/predict_new_network_data.py --input <your_new_data.csv>`

## Remaining theoretical gaps in the written plan

Some items in the original plan are not yet implemented as code artifacts and should be handled carefully in the thesis text or added later if required:

- explicit SHAP/LIME explainability
- adversarial robustness experiments
- pruning and quantization for edge deployment
