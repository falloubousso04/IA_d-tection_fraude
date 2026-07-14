"""
Pipeline complet de detection de fraude, reproductible depuis les donnees brutes.

    python train_from_raw.py

Etapes :
  1. Chargement des CSV bruts (Bank_transaction_scenario1/2.csv)
  2. Nettoyage (normalisation des villes, tri chronologique)
  3. Construction des variables comportementales, de facon CAUSALE :
     chaque transaction n'est decrite qu'a partir des transactions PRECEDENTES
     du meme client. Aucune information du futur ne fuit dans les features.
  4. Separation train/test CHRONOLOGIQUE (on entraine sur le passe, on teste
     sur le futur) : c'est le seul protocole honnete pour un modele deploye.
  5. Entrainement XGBoost + evaluation (matrice de confusion, precision/rappel)
  6. Reentrainement sur 100 % des donnees et sauvegarde pour l'API.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from model_lib import FraudModel

BASE = Path(__file__).parent
ART_DIR = BASE / "model_artifacts"
ART_DIR.mkdir(exist_ok=True)

NUM_FEATURES = [
    "Montant", "montant_log", "heure", "jour_semaine", "est_weekend", "est_nuit",
    "client_montant_moy_hist", "client_montant_std_hist", "client_nb_trans_hist",
    "ecart_montant_client", "temps_depuis_derniere_trans_min", "trans_rapide_5min",
    "changement_localisation",
]
CAT_FEATURES = ["Type de transaction", "Status operation", "Localisation"]

# Fautes de frappe et encodage abime reperes dans les donnees brutes.
CORRECTIONS_VILLES = {
    "Saint-Louis": "Saint Louis",   # 7 lignes ecrites avec un tiret
    "Kafrine": "Kaffrine",          # 4 lignes mal orthographiees
    "RunÃ©": "Inconnu",             # 5 lignes : nom corrompu, ville non identifiable
}

TEST_RATIO = 0.2  # les 20 % de transactions les plus recentes servent de test


def charger_et_nettoyer(path):
    df = pd.read_csv(path, sep=";", encoding="utf-8", parse_dates=["Date"])
    n0 = len(df)

    for col in CAT_FEATURES + ["Target"]:
        df[col] = df[col].astype(str).str.strip()
    df["Localisation"] = df["Localisation"].replace(CORRECTIONS_VILLES)

    df = df.dropna(subset=["ID Clients", "Date", "Montant", "Target"])
    df = df[df["Montant"] > 0]
    df = df.sort_values(["ID Clients", "Date"], kind="mergesort").reset_index(drop=True)

    print(f"  {n0} lignes brutes -> {len(df)} lignes retenues")
    print(f"  villes apres nettoyage : {df['Localisation'].nunique()}")
    return df


def construire_features(df, montant_median_defaut):
    """Variables comportementales calculees uniquement sur le passe du client.

    Le .shift(1) est la piece maitresse : il decale l'historique d'un cran, donc
    la transaction courante n'entre jamais dans sa propre moyenne / son ecart-type.
    """
    g = df.groupby("ID Clients")["Montant"]

    df["client_nb_trans_hist"] = g.cumcount()
    df["client_montant_moy_hist"] = g.transform(lambda s: s.expanding().mean().shift(1))
    df["client_montant_std_hist"] = g.transform(lambda s: s.expanding().std(ddof=0).shift(1))

    # Premiere transaction d'un client : aucun historique -> valeurs par defaut
    df["client_montant_moy_hist"] = df["client_montant_moy_hist"].fillna(montant_median_defaut)
    df["client_montant_std_hist"] = df["client_montant_std_hist"].fillna(0.0)

    std = df["client_montant_std_hist"]
    ecart = (df["Montant"] - df["client_montant_moy_hist"]) / std.where(std > 0)
    df["ecart_montant_client"] = ecart.fillna(0.0).clip(-10, 10)

    delta = df.groupby("ID Clients")["Date"].diff().dt.total_seconds() / 60
    df["temps_depuis_derniere_trans_min"] = delta.fillna(999999.0)
    df["trans_rapide_5min"] = (
        df["temps_depuis_derniere_trans_min"].between(0, 5, inclusive="left").astype(int)
    )

    loc_prec = df.groupby("ID Clients")["Localisation"].shift(1)
    df["changement_localisation"] = (
        loc_prec.notna() & (df["Localisation"] != loc_prec)
    ).astype(int)

    df["heure"] = df["Date"].dt.hour
    df["jour_semaine"] = df["Date"].dt.weekday
    df["est_weekend"] = df["jour_semaine"].isin([5, 6]).astype(int)
    df["est_nuit"] = ((df["heure"] >= 22) | (df["heure"] <= 5)).astype(int)
    df["montant_log"] = np.log1p(df["Montant"])
    return df


def construire_modele(classes):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])
    model = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss" if len(classes) > 2 else "logloss",
        random_state=42, n_jobs=-1,
    )
    return pre, model


def evaluer(nom, df, classes):
    """Entrainement sur le passe, evaluation sur le futur."""
    df = df.sort_values("Date", kind="mergesort").reset_index(drop=True)
    coupe = int(len(df) * (1 - TEST_RATIO))
    date_coupe = df.loc[coupe, "Date"]

    train, test = df.iloc[:coupe], df.iloc[coupe:]
    print(f"\n  Train : {len(train)} transactions (jusqu'au {date_coupe:%d/%m/%Y})")
    print(f"  Test  : {len(test)} transactions (apres cette date)")
    print(f"  Fraudes dans le test : {(test['Target'] == 'Fraude').sum()}")

    pre, model = construire_modele(classes)
    Xtr = pre.fit_transform(train[NUM_FEATURES + CAT_FEATURES])
    ytr = pd.Categorical(train["Target"], categories=classes).codes
    model.fit(Xtr, ytr)

    Xte = pre.transform(test[NUM_FEATURES + CAT_FEATURES])
    yte = pd.Categorical(test["Target"], categories=classes).codes
    pred = model.predict(Xte)

    print(f"\n  --- Matrice de confusion ({nom}) ---")
    cm = confusion_matrix(yte, pred, labels=range(len(classes)))
    entete = "  reel \\ predit  " + "".join(f"{c:>10}" for c in classes)
    print(entete)
    for i, c in enumerate(classes):
        print(f"  {c:<14}" + "".join(f"{v:>10}" for v in cm[i]))

    print(f"\n  --- Precision / rappel ({nom}) ---")
    print(classification_report(yte, pred, labels=range(len(classes)),
                                target_names=classes, digits=3, zero_division=0))


def entrainer_final(nom, df, classes):
    """Reentrainement sur 100 % des donnees, puis sauvegarde pour l'API."""
    pre, model = construire_modele(classes)
    X = df[NUM_FEATURES + CAT_FEATURES]
    y = pd.Categorical(df["Target"], categories=classes).codes
    model.fit(pre.fit_transform(X), y)

    wrapper = FraudModel(pre, model, classes, NUM_FEATURES, CAT_FEATURES, nom)
    import joblib
    joblib.dump(wrapper, ART_DIR / f"model_{nom}.joblib")

    meta = {
        "scenario": nom,
        "classes": classes,
        "num_features": NUM_FEATURES,
        "cat_features": CAT_FEATURES,
        "categories": {c: sorted(df[c].unique().tolist()) for c in CAT_FEATURES},
        "client_montant_median_default": float(df["Montant"].median()),
        "n_transactions": len(df),
        "n_clients": int(df["ID Clients"].nunique()),
    }
    (ART_DIR / f"meta_{nom}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  Modele final entraine sur {len(df)} transactions et sauvegarde.")


def traiter(fichier, nom, classes):
    print("=" * 72)
    print(f"{nom.upper()} — {fichier}")
    print("=" * 72)

    df = charger_et_nettoyer(BASE / fichier)

    # La mediane de reference est calculee sur le passe uniquement, pour ne pas
    # faire fuiter d'information du futur dans les features du jeu de test.
    df = df.sort_values("Date", kind="mergesort").reset_index(drop=True)
    coupe = int(len(df) * (1 - TEST_RATIO))
    mediane = float(df.iloc[:coupe]["Montant"].median())

    df = df.sort_values(["ID Clients", "Date"], kind="mergesort").reset_index(drop=True)
    df = construire_features(df, mediane)

    evaluer(nom, df, classes)
    entrainer_final(nom, df, classes)


if __name__ == "__main__":
    traiter("Bank_transaction_scenario1.csv", "scenario1", ["Normal", "Suspect", "Fraude"])
    print()
    traiter("Bank_transaction_scenario2.csv", "scenario2", ["Normal", "Fraude"])
    print("\nModeles prets pour l'API dans", ART_DIR)
