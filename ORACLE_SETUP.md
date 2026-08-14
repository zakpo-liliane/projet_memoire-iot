# Connexion Oracle pour l'application IDS IIoT

## 1. Creer les tables dans SQL Developer

1. Ouvrir SQL Developer.
2. Choisir la connexion `zakpoDB`.
3. Ouvrir le fichier `sql/oracle_schema.sql`.
4. Cliquer sur `Exécuter le script` ou appuyer sur `F5`.

Ne tape pas `sql/oracle_schema.sql` dans la feuille SQL. Ce texte n'est pas une commande SQL.
Si SQL Developer affiche encore `@@oracle_full_setup.sql`, ferme l'onglet et rouvre `sql/oracle_schema.sql`.

Oracle XE 21c utilise une base conteneur. Le script commence par:

```sql
alter session set container = XEPDB1;
```

Cette ligne corrige l'erreur `ORA-65096`, qui apparait quand on essaie de creer un utilisateur local dans `CDB$ROOT`.

Le script cree les tables:

- `IDS_UPLOADS`
- `IDS_PREDICTIONS`
- `IDS_ALERTS`
- `IDS_MODEL_RESULTS`

Il cree aussi l'utilisateur applicatif:

```text
IDS_APP / Ids_App_2026
```

## 1 bis. Si IDS_APP existe deja

Le script est re-executable. Si `IDS_APP` existe deja, il remet le mot de passe attendu et deverrouille le compte:

```text
IDS_APP / Ids_App_2026
```

Si tu es deja connecte directement avec `IDS_APP`, utilise plutot:

```text
sql/oracle_app_schema_only.sql
```

## 2. Configurer la connexion de l'application

Copier `oracle_config_example.env` vers `oracle_config.env`, puis remplacer les valeurs:

```env
IDS_ORACLE_ENABLED=1
IDS_ORACLE_USER=IDS_APP
IDS_ORACLE_PASSWORD=Ids_App_2026
IDS_ORACLE_DSN=192.168.56.1:1521/xepdb1
```

Selon ton installation Oracle, le DSN peut aussi etre:

```env
IDS_ORACLE_DSN=192.168.56.1:1521/XE
```

Utilise les memes informations que dans la connexion SQL Developer `zakpoDB`.

## 3. Installer le pilote Python Oracle

```powershell
python -m pip install -r requirements-oracle.txt
```

## 3 bis. Installation automatique depuis Python

Si SQL Developer pose probleme, utiliser:

```powershell
python setup_oracle.py
```

Le script demande le mot de passe Oracle de `SYS`, puis cree `IDS_APP` et toutes les tables.

## 4. Lancer l'application

```powershell
python run_webapp.py --port 8060
```

Puis ouvrir:

```text
http://127.0.0.1:8060
```

## 5. Utilisation dans l'application

- La page `Historique Oracle` affiche l'etat de connexion.
- Le bouton `Synchroniser les resultats modeles` enregistre les resultats des 8 modeles dans Oracle.
- Chaque prediction CSV ou demo est sauvegardee dans `IDS_UPLOADS`.
- Les lignes predites sont sauvegardees dans `IDS_PREDICTIONS`.
- Les alertes sont sauvegardees dans `IDS_ALERTS`.
