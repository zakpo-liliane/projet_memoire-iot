# Preuves d'execution - Section 2.3

Ce document relie les elements de la section 2.3 du memoire aux scripts et aux artefacts produits dans le projet.

## 2.3.1 Nettoyage, normalisation et selection de caracteristiques

### Nettoyage

Commande:

```powershell
python src/clean_dataset.py
```

Preuves:

- Script: `src/clean_dataset.py`
- Sortie: `outputs/cic_iiot_2025_cleaned.pkl.gz`
- Traitements effectues: suppression des doublons, controle des valeurs manquantes, conversion des colonnes textuelles en categories, conversion des timestamps.

### Normalisation et encodage

Commandes:

```powershell
python src/prepare_features.py
python src/split_dataset.py
```

Preuves:

- Script d'encodage: `src/prepare_features.py`
- Script de split/normalisation: `src/split_dataset.py`
- Dataset prepare: `outputs/cic_iiot_2025_prepared.pkl.gz`
- Mappings de labels: `outputs/label_mappings.csv`
- Donnees normalisees: `outputs/splits/X_train_scaled.pkl.gz`, `outputs/splits/X_val_scaled.pkl.gz`, `outputs/splits/X_test_scaled.pkl.gz`
- Resume des splits: `outputs/splits/split_summary.csv`

Resume actuel des splits:

```csv
split,rows,attack_count,benign_count
train,479969,199499,280470
validation,102851,42750,60101
test,102851,42750,60101
train_smote,560940,280470,280470
```

## 2.3.1.1 Random Forest + correlation de Pearson

### Correlation de Pearson

Commande:

```powershell
python src/run_eda.py
```

Preuves:

- Script: `src/run_eda.py`
- Figure de correlation: `reports/eda/top_feature_correlation_heatmap.png`
- Resume EDA: `reports/eda/eda_summary.md`
- Resume numerique: `reports/eda/numeric_summary.csv`

Remarque methodologique: la correlation est calculee avec `DataFrame.corr()`, qui utilise Pearson par defaut dans pandas.

### Random Forest

Commande:

```powershell
python src/train_baselines.py
```

Preuves:

- Script: `src/train_baselines.py`
- Modele sauvegarde: `models/baselines/random_forest.joblib`
- Resultats validation: `reports/baselines/baseline_metrics_validation.csv`

Resultat actuel du Random Forest:

```csv
model,accuracy,precision,recall,f1_score,train_seconds
random_forest,0.9625769316778641,0.9429709893848237,0.9962063859170397,0.9688579635098508,1055.2955348491669
```

Complement de selection/importance:

- Script: `src/explain_best_model.py`
- Sortie: `reports/explainability/best_model_feature_importance.csv`
- Figure: `reports/explainability/best_model_feature_importance.png`

Attention: dans les artefacts actuels, l'explicabilite est faite sur le meilleur modele final, qui est un arbre de decision, pas directement sur le Random Forest. Le Random Forest a bien ete entraine comme baseline.

## 2.3.2 Gestion du desequilibre de classes

## 2.3.2.1 SMOTE, focal loss et class weights

### SMOTE

Commande:

```powershell
python src/split_dataset.py
```

Preuves:

- Script: `src/split_dataset.py`
- Donnees d'entrainement equilibrees: `outputs/splits/X_train_scaled_smote.pkl.gz`
- Labels equilibres: `outputs/splits/y_train_smote.csv`
- Resume: `outputs/splits/split_summary.csv`

Preuve numerique:

```csv
split,rows,attack_count,benign_count
train,479969,199499,280470
train_smote,560940,280470,280470
```

Cela montre que SMOTE a equilibre les deux classes a 280470 observations chacune dans l'ensemble d'entrainement.

### Focal loss et class weights

Les deux methodes sont implementees comme options dans les scripts deep learning.

Commandes de verification:

```powershell
python src/train_deep_models.py --help
python src/tune_deep_models.py --help
```

Commandes pour generer des preuves d'execution focal loss / class weights:

```powershell
python src/train_deep_models.py --balance-method class_weight --loss-function focal_loss
python src/tune_deep_models.py --balance-method class_weight --loss-function focal_loss
```

Preuves de code:

- `src/train_deep_models.py` contient `--balance-method`, `--loss-function`, `compute_class_weights()` et `binary_focal_loss()`.
- `src/tune_deep_models.py` reprend les memes options pour le tuning.

Limite actuelle:

- Le rapport existant `reports/deep_learning/deep_learning_experiment_setup.csv` indique une execution avec `Balance method: smote` et `Loss function: binary_crossentropy`.
- Donc les artefacts actuels prouvent SMOTE, mais pas encore une execution finale de `focal_loss` ou `class_weight`. Pour les prouver, il faut relancer les commandes ci-dessus et conserver les fichiers produits dans `reports/deep_learning/`.

