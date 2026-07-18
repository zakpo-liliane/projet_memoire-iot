# Approches intelligentes et adaptatives pour la detection d'intrusion dans les reseaux IIoT industriels

## Pages liminaires

### Page de garde

**Etablissement:** [A completer]  
**Filiere:** [A completer]  
**Auteur:** [A completer]  
**Encadreur:** [A completer]  
**Annee academique:** [A completer]

### Dedicace

Je dedie ce travail a ma famille, pour son soutien moral, sa patience et ses encouragements constants tout au long de ce parcours.

### Remerciements

Je tiens a exprimer ma profonde gratitude a toutes les personnes qui ont contribue a la realisation de ce memoire. J'adresse mes remerciements a mon encadreur pour ses conseils, sa disponibilite et ses orientations methodologiques. Je remercie egalement ma famille, mes proches et toutes les personnes qui m'ont soutenu durant ce travail.

### Sigles et abreviations

- IIoT : Industrial Internet of Things
- IDS : Intrusion Detection System
- ML : Machine Learning
- DL : Deep Learning
- CNN : Convolutional Neural Network
- LSTM : Long Short-Term Memory
- SVM : Support Vector Machine
- ROC : Receiver Operating Characteristic
- AUC : Area Under the Curve
- SMOTE : Synthetic Minority Over-sampling Technique
- EDA : Exploratory Data Analysis

### Resume

Ce memoire etudie la detection d'intrusion dans les reseaux industriels IIoT a travers une comparaison entre des modeles classiques de machine learning et des modeles de deep learning sur le dataset CIC-IIoT-2025. Un pipeline complet a ete mis en place, allant de la fusion des fichiers CSV jusqu'a la comparaison finale des performances. Les donnees ont ete nettoyees, pretraitees, encodees, normalisees et reparties en ensembles d'entrainement, de validation et de test. Les classes ont ete reequilibrees a l'aide de SMOTE et l'analyse exploratoire a permis d'identifier les variables les plus informatives. Les resultats montrent que les modeles classiques, en particulier l'arbre de decision, surpassent les architectures profondes sur le jeu de test pour la detection des attaques.

### Abstract

This thesis investigates intrusion detection in industrial IIoT networks through a comparative study of classical machine learning models and deep learning models on the CIC-IIoT-2025 dataset. A complete pipeline was implemented, from CSV merging to final performance comparison. The data were cleaned, preprocessed, encoded, normalized, and split into training, validation, and test sets. Class imbalance was handled using SMOTE, and exploratory data analysis helped identify the most informative variables. Experimental results show that classical models, especially the decision tree, outperform deep architectures on the test set for attack detection.

### Liste des figures

[A generer automatiquement dans Word]

### Liste des tableaux

[A generer automatiquement dans Word]

## Introduction generale

La transformation numerique des environnements industriels a favorise l'essor de l'Industrial Internet of Things, ou IIoT, qui connecte capteurs, actionneurs, automates et plateformes de supervision dans des architectures interconnectees. Cette evolution offre des avantages importants en termes de productivite, de supervision en temps reel et de maintenance intelligente. Toutefois, elle augmente aussi la surface d'attaque des infrastructures industrielles et expose les systemes a des menaces de cybersecurite de plus en plus sophistiquees.

Dans ce contexte, la detection d'intrusion devient un enjeu majeur pour garantir la disponibilite, l'integrite et la confidentialite des reseaux industriels. Les approches traditionnelles fondees sur des signatures ou des regles statiques montrent rapidement leurs limites face a des attaques inconnues, evolutives ou polymorphes. Pour repondre a cette problematique, les approches basees sur le machine learning et le deep learning suscitent un interet croissant, car elles permettent d'apprendre directement a partir des donnees reseau et d'identifier des comportements suspects avec plus de flexibilite.

Ce memoire a pour objectif d'etudier, de concevoir et d'evaluer plusieurs approches intelligentes de detection d'intrusion appliquees au contexte IIoT, en utilisant le dataset CIC-IIoT-2025. L'etude suit une demarche experimentale complete: preparation des donnees, analyse exploratoire, entrainement de plusieurs modeles, optimisation des hyperparametres, evaluation sur jeu de validation et de test, comparaison finale et sauvegarde du meilleur modele.

Le travail vise a repondre a quatre questions principales: comment preparer efficacement les donnees CIC-IIoT-2025 pour la classification binaire attaque/benign; quels modeles sont les plus adaptes a la detection d'intrusions dans ce contexte; quel compromis existe entre performance, complexite et interpretabilite; et quel modele final peut etre retenu pour un usage pratique dans un systeme IDS industriel.

## Chapitre I : Generalites et etat de l'art

### 1.1 Industrial Internet of Things

L'IIoT represente l'adaptation des principes de l'Internet des objets au monde industriel. Il s'agit d'un ensemble de dispositifs physiques connectes capables de mesurer, transmettre et exploiter des donnees en temps reel afin d'optimiser les processus industriels. Contrairement a l'IoT grand public, l'IIoT evolue dans des environnements ou la fiabilite, la latence et la securite sont des exigences critiques.

L'architecture IIoT repose generalement sur plusieurs couches: une couche de perception composee de capteurs et d'actionneurs; une couche de communication chargee du transport des donnees; une couche de traitement et de supervision; et une couche applicative dediee a l'analyse, a la visualisation et a la prise de decision. Les protocoles industriels tels que Modbus, OPC UA ou DNP3 jouent un role essentiel dans ces architectures, mais leurs mecanismes de securite sont parfois insuffisants face aux menaces modernes.

### 1.2 Systemes de detection d'intrusion

Un systeme de detection d'intrusion est conu pour identifier des activites suspectes ou malveillantes au sein d'un reseau ou d'un systeme informatique. Dans le contexte industriel, il doit detecter des comportements anormaux sur les communications entre equipements, les flux reseau ou les journaux systeme. Les IDS traditionnels sont souvent bases sur des signatures d'attaque connues ou sur des regles expertes. Ils sont rapides et interpretables, mais leur capacite de detection diminue fortement face aux attaques nouvelles ou modifiees.

Les attaques modernes peuvent etre furtives, distribuees, multi-etapes ou adaptees pour ressembler a un trafic legitime. Dans ce cas, les IDS classiques montrent leurs limites et necessitent des compléments fondes sur des methodes apprenantes capables de generaliser a partir des donnees.

### 1.3 Machine learning et deep learning pour la cybersecurite

Le machine learning classique utilise des algorithmes qui apprennent a partir de donnees preprocesses et de caracteristiques selectionnees. Il presente l'avantage d'etre relativement interpretable, rapide a entrainer et souvent efficace sur des donnees tabulaires. Le deep learning, quant a lui, repose sur des architectures neuronales profondes capables d'apprendre automatiquement des representations complexes. Il est potentiellement plus puissant, mais demande souvent davantage de donnees, de ressources de calcul et de reglage.

Dans ce projet, plusieurs architectures ont ete testees: CNN, LSTM, autoencoder, CNN + LSTM, ainsi qu'une version hybride avec mecanisme d'attention. L'objectif etait de mesurer si des modeles profonds peuvent surpasser les approches classiques sur des donnees de cybersecurite industrielle.

### 1.4 Synthese critique

Les travaux recents sur les IDS pour IIoT montrent une forte heterogeneite des approches et des datasets. Les benchmarks comme CIC-IoT-2023 et CIC-IIoT-2025 deviennent des references importantes car ils permettent d'evaluer les modeles dans des scenarios plus realistes. Cependant, plusieurs limites apparaissent frequemment dans la litterature: manque d'explicabilite, absence de tests statistiques, peu d'etudes sur la robustesse, et difficultes de deploiement sur des environnements contraints.

Ce memoire s'inscrit dans cette problematique en proposant une comparaison systematique entre plusieurs approches, avec une evaluation centree sur la detection de la classe attaque.

## Chapitre II : Materiel et methodes

### 2.1 Environnement experimental

Le projet a ete developpe en Python, a partir d'une architecture de dossiers claire: `src` pour les scripts, `outputs` pour les donnees intermediaires, `models` pour les modeles sauvegardes et `reports` pour les resultats et les syntheses. Les bibliotheques principales mobilisees sont `pandas`, `scikit-learn`, `imbalanced-learn`, `tensorflow`, `matplotlib`, `seaborn` et `joblib`.

### 2.2 Dataset CIC-IIoT-2025

Le dataset CIC-IIoT-2025, issu de l'environnement DataSense, contient des donnees de trafic benin et malveillant issues d'un environnement industriel. Le corpus est compose de plusieurs fichiers CSV correspondant a differentes fenetres temporelles. L'analyse exploratoire montre un volume important d'observations, une forte proportion de variables numeriques et un desequilibre entre les classes benignes et attaque.

Le probleme a ete formule comme une classification binaire: `attack` contre `benign`.

### 2.3 Pretraitement

Les donnees ont d'abord ete fusionnees, puis nettoyees en supprimant les doublons, en verifiant les valeurs manquantes et en convertissant les colonnes textuelles et temporelles dans des types adaptes. Les labels ont ete encodes numeriquement a l'aide de `LabelEncoder`. Une variable derivee, `window_duration_seconds`, a ete construite a partir des horodatages.

Les colonnes textuelles a forte cardinalite et les timestamps absolus ont ete retires de l'ensemble destine a l'apprentissage. La normalisation a ete effectuee via `StandardScaler`. La separation train/validation/test a ete realisee de facon stratifiee afin de conserver les proportions de classes. Enfin, `SMOTE` a ete applique sur l'ensemble d'entrainement pour compenser le desequilibre des classes.

### 2.4 Modeles

Les modeles classiques retenus sont `Decision Tree`, `Random Forest` et `Linear SVM`. Les modeles de deep learning comprennent `CNN`, `LSTM`, `Autoencoder`, `CNN + LSTM` et `CNN + LSTM + Attention`. Le modele autoencoder est entraine sur le trafic benin afin d'identifier les anomalies par erreur de reconstruction. Le modele final retenu est l'arbre de decision, qui offre le meilleur compromis entre detection des attaques et simplicite.

## Chapitre III : Resultats, analyse et discussion

### 3.1 Protocole experimental

Les performances ont ete evaluees a l'aide de plusieurs metriques: Accuracy, Precision, Recall, F1-score, Attack F1-score, Macro F1-score et ROC-AUC. La matrice de confusion et les courbes ROC ont complete l'analyse. Une validation croisee 5-fold a ete ajoutee pour les modeles classiques afin de renforcer la robustesse methodologique. Des tests statistiques ont egalement ete realises pour comparer les modeles.

### 3.2 Resultats principaux

Sur le jeu de test, l'arbre de decision obtient les meilleures performances globales, avec une accuracy de 0.9626 et un Attack F1-score de 0.9532. La foret aleatoire suit de tres pres. Les modeles de deep learning, bien qu'interessants, restent inferieurs sur ce jeu de donnees tabulaire.

### 3.3 Analyse

Les resultats montrent que, dans ce contexte precis, les modeles classiques surpassent les architectures profondes. Cela confirme qu'un pipeline de preparation solide et une caracterisation pertinente des variables peuvent rendre des approches simples tres competitives. La metrique centrale pour ce probleme est l'Attack F1-score, car elle mesure la capacite a detecter les attaques sans se limiter a l'accuracy globale.

### 3.4 Explicabilite et robustesse

Une analyse d'explicabilite a ete menee sur le modele final via l'importance des variables et la permutation importance. Les variables lies aux tailles de fenetre, aux deltas temporels et a certains compteurs reseau ressortent comme les plus discriminantes. Un export TFLite quantifie a egalement ete produit pour un scenario edge.

## Conclusion generale

Ce memoire a montre qu'un pipeline rigoureux de preparation des donnees, combine a une comparaison systematique entre modeles classiques et architectures profondes, permet de construire un IDS efficace pour les reseaux IIoT industriels. Dans le cadre du dataset CIC-IIoT-2025, l'arbre de decision s'est revele etre le meilleur modele global pour la detection des attaques.

Le travail met en evidence que le deep learning n'apporte pas automatiquement un gain de performance sur des donnees tabulaires industrielles. Il souligne egalement l'importance de l'equilibrage des classes, du choix des metriques et de la coherence du protocole experimental. Les perspectives portent sur l'explicabilite plus avancee, l'etude de la robustesse adversariale, et le deploiement en environnement temps reel.

## Bibliographie

1. Firouzi, A., Dadkhah, S., Maret, S. A., & Ghorbani, A. A. (2025). DataSense: A Real-Time Sensor-Based Benchmark Dataset for Attack Analysis in IIoT with Multi-Objective Feature Selection. *Electronics, 14*(20), 4095.
2. Canadian Institute for Cybersecurity. (2025). CIC-IIoT-2025 Dataset.
3. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique.
4. Breiman, L. (2001). Random Forests.
5. Cortes, C., & Vapnik, V. (1995). Support-vector networks.
6. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.
7. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition.
8. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.

## Annexes

- Annexe A : code source complet
- Annexe B : matrices de confusion et courbes ROC
- Annexe C : details techniques du dataset CIC-IIoT-2025

