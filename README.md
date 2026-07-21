# Détection de Fraude Bancaire — API & tableau de bord

Système de scoring de transactions bancaires en temps réel, entraîné sur des
transactions sénégalaises. Une API REST expose deux modèles XGBoost, et un
tableau de bord web permet de simuler des transactions et de visualiser le
risque.

Projet réalisé dans le cadre du cours d'IA Distribuée (Licence 3 Informatique).

## Ce que fait le projet

Chaque transaction envoyée à l'API est scorée en tenant compte de **l'historique
comportemental du client**, pas seulement de ses caractéristiques brutes. L'API
maintient, pour chaque client, un résumé glissant de ses transactions passées
(montant moyen, écart-type, dernière localisation, dernière date) et en dérive
13 variables comportementales :

- montant moyen et écart-type historiques du client,
- écart de la transaction au comportement habituel (en écarts-types),
- temps écoulé depuis la transaction précédente,
- changement de localisation depuis la dernière transaction,
- heure, jour de la semaine, week-end, transaction nocturne.

Ces variables sont calculées de façon **strictement causale** : une transaction
n'est jamais décrite à l'aide d'informations postérieures à elle-même.

## Résultats

Évaluation sur les 20 % de transactions les plus récentes, jamais vues à
l'entraînement (découpe **chronologique** : entraînement sur le passé, test sur
le futur).

| Modèle | Classes | Exactitude | Rappel sur les fraudes |
|---|---|---|---|
| `scenario2` (recommandé) | Normal / Fraude | **99,0 %** | 82 % (71 fraudes sur 87), aucune fausse alerte |
| `scenario1` | Normal / Suspect / Fraude | **90,3 %** | 58 % (32 fraudes sur 55) |

Le rappel du scénario 1 est la limite connue du système : 17 fraudes sur 55
passent pour des transactions normales. C'est la conséquence du fort déséquilibre
des classes (3,7 % de fraudes seulement).

## Contenu

```
fraude_deploy/
├── app.py                  # API FastAPI (scoring temps réel)
├── model_lib.py            # classe FraudModel, partagée entraînement / API
├── train_from_raw.py       # pipeline complet : nettoyage → features → entraînement → évaluation
│                           #   (ses fonctions sont réutilisées par l'API pour l'analyse par lot)
├── demo.html               # tableau de bord web
├── model_artifacts/        # modèles entraînés (.joblib) + métadonnées (.json)
├── requirements.txt
└── Dockerfile
```

Les fichiers de données (`Bank_transaction_scenario1.csv`,
`Bank_transaction_scenario2.csv`) ne sont **pas** versionnés : ils contiennent
des identifiants clients et des numéros de compte. Les modèles déjà entraînés
sont fournis, l'API fonctionne donc sans eux.

## Lancer l'API

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn app:app --port 8000
```

- Documentation interactive : http://localhost:8000/docs
- Tableau de bord : ouvrir `demo.html` dans un navigateur

Exemple d'appel :

```bash
curl -X POST http://localhost:8000/predict/scenario2 \
  -H "Content-Type: application/json" \
  -d '{
    "id_client": "C001",
    "type_transaction": "Paiement en ligne",
    "status_operation": "Validé",
    "localisation": "Dakar",
    "date": "2025-03-01T14:30:00",
    "montant": 50000
  }'
```

Réponse :

```json
{
  "scenario": "scenario2",
  "classe_predite": "Fraude",
  "probabilites": {"Normal": 0.0065, "Fraude": 0.9935},
  "alerte": true,
  "niveau_risque": "ÉLEVÉ",
  "features_utilisees": { "...": "les 16 variables transmises au modèle" }
}
```

Comme le modèle est à état, envoyer **plusieurs transactions pour le même
`id_client`** fait évoluer le risque : les premières servent à apprendre son
comportement habituel, et une transaction atypique se détache ensuite nettement.

## Analyser un fichier CSV entier

L'onglet **Import CSV** du tableau de bord accepte un fichier de transactions et
score toutes les lignes en une passe. En ligne de commande :

```bash
curl -X POST http://localhost:8000/predict/batch/scenario2 \
  -F "file=@mes_transactions.csv"
```

Colonnes attendues (séparateur `;`, `,` ou tabulation, détecté automatiquement) :
`ID Clients`, `Date`, `Type de transaction`, `Status operation`, `Localisation`,
`Montant`. Une colonne `Target` facultative déclenche en plus une évaluation des
prédictions contre la vérité terrain (exactitude, matrice de confusion,
précision et rappel par classe).

Les variables comportementales sont reconstruites **au sein du fichier**, client
par client et dans l'ordre chronologique, en réutilisant la fonction
`construire_features` du script d'entraînement — entraînement et inférence ne
peuvent donc pas diverger.

> Attention à l'interprétation : analyser le fichier ayant servi à entraîner le
> modèle donne des scores proches de 100 %, puisqu'il a déjà vu ces lignes. Seule
> l'évaluation sur des transactions inconnues (onglet **Modèle**) est
> significative.

## Déployer avec Docker

```bash
docker build -t fraude-api .
docker run -d --name fraude-api -p 8000:8000 fraude-api
```

## Réentraîner les modèles

Déposez `Bank_transaction_scenario1.csv` et `Bank_transaction_scenario2.csv`
(séparateur `;`, encodage UTF-8) à la racine, puis :

```bash
python train_from_raw.py
```

Le script nettoie les données (fusion des villes mal orthographiées, correction
d'un nom corrompu), reconstruit les variables comportementales, évalue les
modèles sur une découpe chronologique, puis réentraîne sur l'ensemble des
données et régénère `model_artifacts/`.

## Limites connues

- **Persistance de l'état client** : l'historique est conservé en mémoire (`dict`
  Python). Il est perdu au redémarrage et incohérent si l'API est répliquée sur
  plusieurs instances. En production, remplacer `CLIENT_STORE` par Redis ou une
  table PostgreSQL (les champs `n`, `mean`, `m2`, `derniere_date`,
  `derniere_localisation` suffisent).
- **Aucune authentification** : l'API est ouverte. Ajouter une clé API ou un
  token JWT avant toute exposition publique.
- **Rappel du scénario 1** : 58 % seulement. Pondérer les classes à
  l'entraînement améliorerait la détection, au prix de fausses alertes.
- **Seuils de décision** : les paliers de risque (0,3 et 0,7 sur la probabilité
  de fraude) sont des valeurs de départ, à calibrer selon le compromis
  précision / rappel souhaité.
