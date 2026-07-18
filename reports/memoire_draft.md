# Approches intelligentes et adaptatives pour la detection d'intrusion dans les reseaux IIoT industriels

## Pages liminaires

- Page de garde
- Dedicace
- Remerciements
- Sigles et abreviations
- Liste des figures
- Liste des tableaux
- Resume
- Abstract

## Resume

Ce memoire etudie la detection d'intrusion dans les reseaux industriels IIoT a travers une comparaison entre des modeles classiques de machine learning et des modeles de deep learning sur le dataset CIC-IIoT-2025. Un pipeline complet a ete mis en place: fusion des CSV, nettoyage, encodage, normalisation, equilibrage, EDA, separation train/validation/test, entrainement, tuning, evaluation et comparaison finale. Les resultats montrent que les modeles classiques, en particulier l'arbre de decision, surpassent les architectures profondes sur le jeu de test pour la detection des attaques.

## Abstract

This thesis investigates intrusion detection in industrial IIoT networks through a comparative study of classical machine learning and deep learning models on the CIC-IIoT-2025 dataset. A full pipeline was implemented: CSV merging, cleaning, encoding, normalization, balancing, exploratory data analysis, train/validation/test splitting, model training, tuning, evaluation, and final comparison. Experimental results show that classical models, especially the decision tree, outperform deep architectures on the test set for attack detection.

## Introduction generale

La transformation numerique des environnements industriels a favorise l'essor de l'Industrial Internet of Things (IIoT), qui connecte capteurs, actionneurs, automates et plateformes de supervision. Cette interconnexion apporte des gains de performance mais expose aussi les infrastructures a de nouvelles menaces de cybersecurite. Dans ce contexte, les systemes de detection d'intrusion (IDS) deviennent essentiels pour preserver la disponibilite, l'integrite et la confidentialite des reseaux industriels.

L'objectif de ce travail est de concevoir, entrainer et comparer plusieurs approches de detection d'intrusion sur le dataset CIC-IIoT-2025. La methode retenue repose sur une chaine de traitement complete allant du preprocessing des donnees jusqu'a la comparaison finale des performances. Les modeles etudies comprennent des approches classiques (Decision Tree, Random Forest, Linear SVM) et des architectures de deep learning (CNN, LSTM, Autoencoder, CNN + LSTM, CNN + LSTM + Attention).

## Chapitre I : Generalites et etat de l'art

### 1.1 Industrial Internet of Things (IIoT)

L'IIoT etend les principes de l'Internet des objets au monde industriel. Il met en relation les capteurs, les actionneurs, les automates, les passerelles et les systemes de supervision afin de permettre un pilotage en temps reel des processus industriels. Les protocoles industriels comme Modbus, OPC UA et DNP3 jouent un role central dans ces architectures.

### 1.2 Systemes de detection d'intrusion (IDS)

Les IDS traditionnels reposent souvent sur des signatures ou des regles statiques. Ils sont efficaces pour des attaques connues mais limitent face aux attaques nouvelles, polymorphes ou zero-day. Les approches basees sur l'IA apportent une capacite d'adaptation superieure en apprenant directement a partir des donnees reseau.

### 1.3 Machine learning et deep learning pour la cybersecurite

Le machine learning classique offre des solutions rapides, robustes et souvent interpretablees, tandis que le deep learning permet de capturer des relations plus complexes entre les caracteristiques. Dans ce projet, les architectures CNN, LSTM, hybrides et autoencodeur ont ete experimentes pour mesurer leur capacite de detection sur des donnees tabulaires industrielles.

### 1.4 Synthese critique

La litterature montre une forte heterogeneite des approches et des datasets. Les references recentes sur l'IIoT, les IDS et les jeux de donnees comme CIC-IoT-2023 et CIC-IIoT-2025 soulignent l'importance de benchmarks realistes. Les lacunes frequentes concernent l'explicabilite, la robustesse, les tests statistiques et le passage a l'echelle.

## Chapitre II : Materiel et methodes

### 2.1 Environnement experimental

Le projet a ete developpe en Python avec `pandas`, `scikit-learn`, `imbalanced-learn`, `tensorflow`, `matplotlib`, `seaborn` et `joblib`. La structure du depot se compose de `src`, `outputs`, `models` et `reports`, afin de garantir la reproductibilite du pipeline.

### 2.2 Dataset CIC-IIoT-2025

Le dataset contient des flux benins et malveillants issus d'un environnement IIoT industriel. L'analyse exploratoire montre un volume important de donnees, une majorite de variables numeriques et un desequilibre des classes, justifiant l'usage de `SMOTE` et de mesures de performance sensibles a la classe attaque.

### 2.3 Pretraitement

Le pipeline comprend la fusion des CSV, le nettoyage des doublons et des valeurs manquantes, la conversion des types, l'encodage des labels, la creation d'une variable de duree de fenetre, la normalisation avec `StandardScaler` et l'equilibrage du train avec `SMOTE`. Les donnees sont ensuite separees en train, validation et test par stratification.

### 2.4 Modeles

Les modeles classiques sont entrainees sur les donnees tabulaires preprocesses. Les modeles deep learning sont construits sur des representations sequentielles des features. Le modele hybride CNN + LSTM + Attention a ete ajoute pour rapprocher davantage le projet du plan de memoire. L'autoencodeur sert a la detection non supervisee par erreur de reconstruction.

## Chapitre III : Resultats, analyse et discussion

### 3.1 Protocole et metriques

Les performances sont evaluees avec Accuracy, Precision, Recall, F1-score, Attack F1-score, Macro F1-score et ROC-AUC. La matrice de confusion et les courbes ROC complètent l'analyse. Une validation croisee 5-fold a ete ajoutee pour les modeles classiques afin de renforcer la partie methodologique.

### 3.2 Resultats

Sur le jeu de test, l'arbre de decision obtient les meilleures performances globales. La foret aleatoire suit de pres, tandis que les modeles deep learning restent en dessous sur ce jeu de donnees tabulaires. Les resultats confirment qu'un pipeline de preparation solide peut permettre a des modeles classiques de surpasser des architectures plus complexes.

### 3.3 Discussion

Les resultats montrent que le deep learning n'apporte pas systematiquement un gain de performance sur des donnees de type tabulaire industriel. Le choix de la metrique d'attaque est essentiel pour eviter qu'une accuracy elevee masque une mauvaise detection des intrusions. Des tests statistiques ont ete ajoutes pour renforcer l'analyse des differences entre modeles.

## Conclusion generale

Le travail realise montre qu'une chaine de traitement rigoureuse, allant de la preparation des donnees a la comparaison finale, permet de construire un IDS efficace sur le dataset CIC-IIoT-2025. Dans ce contexte, le modele final retenu est un arbre de decision, qui offre le meilleur compromis entre detection des attaques, simplicite et performance globale. Les perspectives incluent l'explicabilite plus avancee, la robustesse adversariale et des optimisations de deploiement plus poussées.

## Annexes

- Annexe A : code source complet
- Annexe B : matrices de confusion et courbes ROC
- Annexe C : details techniques du dataset CIC-IIoT-2025

