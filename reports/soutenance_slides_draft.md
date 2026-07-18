# Plan de soutenance

## Diapositive 1 - Titre

- Approches intelligentes et adaptatives pour la detection d'intrusion dans les reseaux IIoT industriels
- Modélisation, apprentissage profond et validation expérimentale sur CIC-IIoT-2025
- Nom de l'etudiant
- Encadreur
- Etablissement / annee academique

## Diapositive 2 - Contexte

- La connectivite industrielle augmente la surface d'attaque
- Les IDS traditionnels detectent mal les attaques nouvelles ou modifiees
- Besoin de modeles adaptatifs pour les environnements IIoT

## Diapositive 3 - Problématique

- Comment detecter efficacement les attaques dans un environnement IIoT ?
- Les modeles classiques ou profonds sont-ils les plus performants ?
- Quel est le meilleur compromis entre detection, robustesse et simplicite ?

## Diapositive 4 - Objectifs

- Construire un pipeline complet de detection d'intrusion
- Comparer des modeles classiques et deep learning
- Identifier le meilleur modele pour le dataset CIC-IIoT-2025
- Tester le modele final sur de nouvelles donnees reseau

## Diapositive 5 - Dataset et preparation

- Dataset CIC-IIoT-2025 / DataSense
- Fusion des CSV attaques et benins
- Nettoyage, encodage, normalisation
- Separation train / validation / test
- Equilibrage du train avec SMOTE

## Diapositive 6 - Analyse exploratoire

- Distribution des classes: fort desequilibre
- Variables majoritairement numeriques
- Identification des features les plus informatives
- Corrélations et histogrammes pour comprendre le trafic

## Diapositive 7 - Modeles testes

- Modeles classiques:
  - Decision Tree
  - Random Forest
  - Linear SVM
- Modeles deep learning:
  - CNN
  - LSTM
  - Autoencoder
  - CNN + LSTM
  - CNN + LSTM + Attention

## Diapositive 8 - Protocole experimental

- Entrainement sur le train
- Validation sur l'ensemble de validation
- Tuning des hyperparametres
- Evaluation finale sur le test
- Mesures utilisees:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Attack F1-score
  - ROC-AUC

## Diapositive 9 - Resultats validation

- Les modeles CNN et CNN+LSTM montrent de bonnes performances en validation
- Les modeles classiques restent competitifs
- L'autoencoder depasse la simple accuracy mais reste fragile sur la detection fine

## Diapositive 10 - Resultats test

- Decision Tree: meilleure performance globale
- Attack F1-score: 0.9532
- Accuracy: 0.9626
- ROC-AUC: 0.9815
- Random Forest juste derriere
- Les modeles profonds sont moins performants sur le test

## Diapositive 11 - Comparaison finale

- Les modeles classiques dominent les modeles deep learning
- Le meilleur modele final est Decision Tree
- Raison du choix:
  - meilleur Attack F1-score
  - bonne accuracy
  - simplicite et interpretabilite

## Diapositive 12 - Explicabilité et validation supplementaire

- Importance des variables du meilleur modele
- Variables les plus influentes:
  - tailles de fenetre
  - deltas temporels
  - compteurs reseau
- Tests statistiques:
  - McNemar
  - ANOVA
  - Wilcoxon

## Diapositive 13 - Deploiement et test sur nouvelles donnees

- Sauvegarde du meilleur modele
- Test sur de nouvelles donnees reseau
- Production de:
  - predictions
  - matrice de confusion
  - metriques
- Export edge quantifie disponible

## Diapositive 14 - Limites

- Pas de SHAP / LIME dans l'etat actuel
- Pas d'etude adversariale
- Les performances deep learning restent inferieures sur ce dataset tabulaire

## Diapositive 15 - Conclusion

- Pipeline IDS complet realise
- Le modele retenu est Decision Tree
- Le projet est techniquement coherent et experimentalement valide
- Perspectives:
  - explicabilite plus avancee
  - robustesse adversariale
  - passage au temps reel

## Diapositive 16 - Remerciements

- Remerciements a l'encadreur
- Remerciements a l'etablissement
- Remerciements a la famille et aux proches

