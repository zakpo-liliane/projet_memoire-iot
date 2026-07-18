# Plan Compliance Report

Ce rapport complete les elements demandes dans le plan du memoire.

## Analyses ajoutees

- Analyse par famille d'attaque: `attack_family_performance_test.csv`.
- Analyse par type d'attaque detaille: `attack_type_performance_test.csv`.
- Analyse par scenario: `attack_scenario_performance_test.csv`.
- Temps d'inference du meilleur modele: `best_model_inference_timing_summary.csv`.
- Tableau de synthese critique de 15 references: `state_of_art_15_references.csv`.

## Inference

- Modele mesure: `decision_tree`.
- Echantillons test: `102851`.
- Temps moyen total: `0.063099` seconde(s).
- Temps moyen par echantillon: `0.000613` ms.
- Debit moyen: `1657164.18` echantillons/seconde.

## Familles d'attaques principales

```csv
label2,support,attack_support,attack_detected,attack_missed_as_benign,attack_detection_rate,accuracy,precision_attack,recall_attack,f1_attack,macro_f1
recon,15798,15798,13554,2244,0.858,0.858,1.0,0.858,0.9235,0.4618
dos,8643,8643,8406,237,0.9726,0.9726,1.0,0.9726,0.9861,0.493
ddos,8593,8593,8025,568,0.9339,0.9339,1.0,0.9339,0.9658,0.4829
mitm,3822,3822,3691,131,0.9657,0.9657,1.0,0.9657,0.9826,0.4913
malware,3605,3605,3487,118,0.9673,0.9673,1.0,0.9673,0.9834,0.4917
web,1347,1347,1206,141,0.8953,0.8953,1.0,0.8953,0.9448,0.4724
bruteforce,942,942,821,121,0.8715,0.8715,1.0,0.8715,0.9314,0.4657
benign,60101,0,0,0,0.0,0.9952,0.0,0.0,0.0,0.4988
```

## Statut multiclasse

Les labels multiclasse `label2`, `label3` et `label4` sont conserves et analyses par groupe. Le modele final reste volontairement binaire (`attack` contre `benign`) car l'objectif experimental principal est la detection d'intrusion. Un entrainement multiclasse complet peut etre ajoute comme extension, mais il ne doit pas etre presente comme deja realise.

## Statut SHAP/LIME

Les bibliotheques SHAP et LIME ne sont pas installees dans cet environnement. L'explicabilite realisee repose donc sur l'importance des variables et la permutation importance. SHAP/LIME sont conserves comme perspective ou limite methodologique.