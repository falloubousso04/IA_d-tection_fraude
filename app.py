"""
API de détection de fraude bancaire — scoring en temps réel.

Architecture :
- Un "client_store" en mémoire (remplaçable par Redis/PostgreSQL en production) garde,
  pour chaque client, un résumé glissant (moyenne, écart-type, nb transactions, dernière
  transaction, dernière localisation) afin de calculer les mêmes features comportementales
  qu'à l'entraînement, de façon causale (jamais de fuite du futur).
- Deux modèles sont exposés : /predict/scenario1 (3 classes) et /predict/scenario2 (binaire).
- Un endpoint générique /predict route vers le modèle demandé.

Lancement local :
    uvicorn app:app --host 0.0.0.0 --port 8000

Documentation interactive générée automatiquement : http://localhost:8000/docs
"""
import sys
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import io

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))
from model_lib import FraudModel  # noqa: F401  (nécessaire pour désérialiser joblib)

# Le traitement par lot reutilise EXACTEMENT le code de construction des features
# de l'entrainement : toute divergence entre entrainement et inference est ainsi
# impossible par construction.
from train_from_raw import (
    CAT_FEATURES,
    CORRECTIONS_VILLES,
    NUM_FEATURES,
    construire_features,
)

ART_DIR = Path(__file__).parent / "model_artifacts"

app = FastAPI(
    title="API de Détection de Fraude Bancaire",
    description="Scoring en temps réel de transactions bancaires (TechKoom / BEM Africa).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre en production aux domaines autorisés
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Chargement des modèles et métadonnées au démarrage
# ---------------------------------------------------------------------------
MODELS = {}
META = {}
for scenario in ["scenario1", "scenario2"]:
    MODELS[scenario] = joblib.load(ART_DIR / f"model_{scenario}.joblib")
    META[scenario] = json.loads((ART_DIR / f"meta_{scenario}.json").read_text(encoding="utf-8"))

# Etat client en mémoire : { (scenario, id_client): {n, moyenne, m2 (var Welford), derniere_date, derniere_loc} }
CLIENT_STORE = {}


# ---------------------------------------------------------------------------
# Schémas d'entrée / sortie
# ---------------------------------------------------------------------------
class Transaction(BaseModel):
    id_client: str = Field(..., description="Identifiant client")
    type_transaction: Literal["ATM", "Paiement en ligne", "Paiement électronique"]
    status_operation: Literal["Validé", "Echoué", "En attente"]
    localisation: str = Field(..., description="Ville de la transaction")
    date: datetime = Field(..., description="Horodatage ISO 8601, ex: 2026-07-14T22:30:00")
    montant: float = Field(..., gt=0, description="Montant en FCFA")


class PredictionResponse(BaseModel):
    scenario: str
    classe_predite: str
    probabilites: dict
    alerte: bool
    niveau_risque: str
    features_utilisees: dict


def get_client_state(scenario, id_client):
    key = (scenario, id_client)
    return CLIENT_STORE.get(key)


def update_client_state(scenario, id_client, montant, date, localisation):
    key = (scenario, id_client)
    state = CLIENT_STORE.get(key)
    if state is None:
        CLIENT_STORE[key] = {
            "n": 1, "mean": montant, "m2": 0.0,
            "derniere_date": date, "derniere_loc": localisation,
        }
    else:
        n0 = state["n"]
        n1 = n0 + 1
        delta = montant - state["mean"]
        mean1 = state["mean"] + delta / n1
        m2_1 = state["m2"] + delta * (montant - mean1)
        state.update({
            "n": n1, "mean": mean1, "m2": m2_1,
            "derniere_date": date, "derniere_loc": localisation,
        })


def build_features(scenario, tx: Transaction):
    """Reconstruit exactement les mêmes features que celles utilisées à l'entraînement,
    en n'utilisant QUE l'historique connu avant cette transaction (calcul causal)."""
    meta = META[scenario]
    state = get_client_state(scenario, tx.id_client)

    heure = tx.date.hour
    jour_semaine = tx.date.weekday()
    est_weekend = int(jour_semaine in (5, 6))
    est_nuit = int(heure >= 22 or heure <= 5)
    montant_log = math.log1p(max(tx.montant, 0))

    if state is None:
        client_montant_moy_hist = meta["client_montant_median_default"]
        client_montant_std_hist = 0.0
        client_nb_trans_hist = 0
        temps_depuis_derniere_trans_min = 999999.0
        changement_localisation = 0
    else:
        client_montant_moy_hist = state["mean"]
        variance = state["m2"] / state["n"] if state["n"] > 0 else 0.0
        client_montant_std_hist = math.sqrt(max(variance, 0.0))
        client_nb_trans_hist = state["n"]
        # Une transaction anterieure a la derniere connue (arrivee hors ordre)
        # donnerait un temps negatif, valeur absente des donnees d'entrainement.
        # On plafonne a 0 : le modele n'extrapole jamais hors de son domaine.
        ecoule = (tx.date - state["derniere_date"]).total_seconds() / 60
        temps_depuis_derniere_trans_min = max(ecoule, 0.0)
        changement_localisation = int(tx.localisation != state["derniere_loc"])

    ecart_montant_client = (
        (tx.montant - client_montant_moy_hist) / client_montant_std_hist
        if client_montant_std_hist > 0 else 0.0
    )
    ecart_montant_client = max(min(ecart_montant_client, 10), -10)
    trans_rapide_5min = int(0 <= temps_depuis_derniere_trans_min < 5)

    localisation = tx.localisation if tx.localisation in meta["categories"]["Localisation"] else "Inconnu"

    features = {
        "Montant": tx.montant,
        "montant_log": montant_log,
        "heure": heure,
        "jour_semaine": jour_semaine,
        "est_weekend": est_weekend,
        "est_nuit": est_nuit,
        "client_montant_moy_hist": client_montant_moy_hist,
        "client_montant_std_hist": client_montant_std_hist,
        "client_nb_trans_hist": client_nb_trans_hist,
        "ecart_montant_client": ecart_montant_client,
        "temps_depuis_derniere_trans_min": temps_depuis_derniere_trans_min,
        "trans_rapide_5min": trans_rapide_5min,
        "changement_localisation": changement_localisation,
        "Type de transaction": tx.type_transaction,
        "Status operation": tx.status_operation,
        "Localisation": localisation,
    }
    return features


def score(scenario: str, tx: Transaction) -> PredictionResponse:
    if scenario not in MODELS:
        raise HTTPException(status_code=404, detail=f"Scénario inconnu : {scenario}")

    features = build_features(scenario, tx)
    X_df = pd.DataFrame([features])

    model: FraudModel = MODELS[scenario]
    pred_classes, proba = model.predict(X_df)
    pred_classe = pred_classes[0]
    proba_dict = {cls: round(float(p), 4) for cls, p in zip(model.classes, proba[0])}

    # mise à jour de l'état client APRES le scoring (la transaction courante devient historique)
    update_client_state(scenario, tx.id_client, tx.montant, tx.date, tx.localisation)

    p_fraude = proba_dict.get("Fraude", 0.0)
    if p_fraude >= 0.7:
        niveau = "ÉLEVÉ"
    elif p_fraude >= 0.3 or pred_classe == "Suspect":
        niveau = "MOYEN"
    else:
        niveau = "FAIBLE"
    alerte = pred_classe in ("Fraude", "Suspect")

    return PredictionResponse(
        scenario=scenario,
        classe_predite=pred_classe,
        probabilites=proba_dict,
        alerte=alerte,
        niveau_risque=niveau,
        features_utilisees={k: (round(v, 4) if isinstance(v, float) else v) for k, v in features.items()},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "modeles_charges": list(MODELS.keys())}


@app.get("/meta/{scenario}")
def get_meta(scenario: str):
    if scenario not in META:
        raise HTTPException(status_code=404, detail="Scénario inconnu")
    return META[scenario]


@app.get("/rapport/{scenario}")
def get_rapport(scenario: str):
    """Statistiques du jeu de donnees et performances mesurees du modele.

    Fige au moment de l'entrainement (train_from_raw.py), ce qui permet au
    tableau de bord de les afficher sans acces aux donnees brutes.
    """
    fichier = ART_DIR / f"rapport_{scenario}.json"
    if not fichier.exists():
        raise HTTPException(status_code=404, detail="Rapport indisponible pour ce scénario")
    return json.loads(fichier.read_text(encoding="utf-8"))


@app.post("/predict/scenario1", response_model=PredictionResponse)
def predict_scenario1(tx: Transaction):
    """Modèle 3 classes : Normal / Suspect / Fraude."""
    return score("scenario1", tx)


@app.post("/predict/scenario2", response_model=PredictionResponse)
def predict_scenario2(tx: Transaction):
    """Modèle binaire : Normal / Fraude (recommandé en priorité — le plus robuste)."""
    return score("scenario2", tx)


@app.post("/predict/{scenario}", response_model=PredictionResponse)
def predict_generic(scenario: str, tx: Transaction):
    return score(scenario, tx)


# ---------------------------------------------------------------------------
# Analyse par lot : un fichier CSV entier scoré en une passe
# ---------------------------------------------------------------------------
COLONNES_REQUISES = ["ID Clients", "Date", "Type de transaction",
                     "Status operation", "Localisation", "Montant"]
MAX_LIGNES = 20000


def _lire_csv(contenu: bytes) -> pd.DataFrame:
    """Lit le CSV en tolerant les separateurs et encodages courants.

    A defaut d'une lecture comportant toutes les colonnes attendues, renvoie la
    meilleure tentative : le controle des colonnes produira alors un message
    precis ("colonnes manquantes") plutot qu'un vague "fichier illisible".
    """
    meilleur = None
    for encodage in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            texte = contenu.decode(encodage)
        except UnicodeDecodeError:
            continue
        for sep in (";", ",", "\t"):
            try:
                df = pd.read_csv(io.StringIO(texte), sep=sep)
            except Exception:  # separateur invalide pour ce fichier
                continue
            if len(df.columns) >= len(COLONNES_REQUISES):
                return df
            if meilleur is None or len(df.columns) > len(meilleur.columns):
                meilleur = df
    if meilleur is not None:
        return meilleur
    raise HTTPException(
        status_code=400,
        detail="Fichier illisible : ce n'est pas un CSV valide "
               "(séparateur attendu « ; », « , » ou tabulation).",
    )


@app.post("/predict/batch/{scenario}")
async def predict_batch(scenario: str, file: UploadFile = File(...)):
    """Score un fichier CSV complet.

    Les variables comportementales sont reconstruites au sein du fichier, client
    par client et dans l'ordre chronologique : chaque transaction n'est decrite
    qu'a partir de celles qui la precedent. Si une colonne `Target` est presente,
    les predictions sont confrontees a la verite terrain.
    """
    if scenario not in MODELS:
        raise HTTPException(status_code=404, detail=f"Scénario inconnu : {scenario}")

    df = _lire_csv(await file.read())
    df.columns = [str(c).strip() for c in df.columns]

    manquantes = [c for c in COLONNES_REQUISES if c not in df.columns]
    if manquantes:
        raise HTTPException(
            status_code=400,
            detail=f"Colonnes manquantes : {', '.join(manquantes)}. "
                   f"Colonnes attendues : {', '.join(COLONNES_REQUISES)}.",
        )
    if len(df) > MAX_LIGNES:
        raise HTTPException(status_code=413,
                            detail=f"Fichier trop volumineux ({len(df)} lignes, maximum {MAX_LIGNES}).")

    n_brut = len(df)
    df = df[COLONNES_REQUISES + ([("Target")] if "Target" in df.columns else [])].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Montant"] = pd.to_numeric(df["Montant"], errors="coerce")
    for c in CAT_FEATURES:
        df[c] = df[c].astype(str).str.strip()
    df["Localisation"] = df["Localisation"].replace(CORRECTIONS_VILLES)

    df = df.dropna(subset=["ID Clients", "Date", "Montant"])
    df = df[df["Montant"] > 0]
    if df.empty:
        raise HTTPException(status_code=400, detail="Aucune ligne exploitable après nettoyage.")

    meta = META[scenario]
    connues = set(meta["categories"]["Localisation"])
    df["Localisation"] = df["Localisation"].where(df["Localisation"].isin(connues), "Inconnu")

    # Ordre chronologique par client : indispensable au calcul causal des features.
    df = df.sort_values(["ID Clients", "Date"], kind="mergesort").reset_index(drop=True)
    df = construire_features(df, meta["client_montant_median_default"])

    model: FraudModel = MODELS[scenario]
    classes, proba = model.predict(df[NUM_FEATURES + CAT_FEATURES])
    idx_fraude = model.classes.index("Fraude") if "Fraude" in model.classes else -1
    p_fraude = proba[:, idx_fraude] if idx_fraude >= 0 else np.zeros(len(df))

    niveaux = np.where(p_fraude >= 0.7, "ÉLEVÉ",
                       np.where((p_fraude >= 0.3) | (classes == "Suspect"), "MOYEN", "FAIBLE"))
    alerte = np.isin(classes, ["Fraude", "Suspect"])

    # Format compact : un tableau par ligne plutot qu'un objet, pour limiter le
    # volume transfere sur les gros fichiers.
    lignes = [
        [str(c), d.strftime("%Y-%m-%d %H:%M"), float(m), str(t), str(v),
         str(cl), round(float(p), 4), str(nv)]
        for c, d, m, t, v, cl, p, nv in zip(
            df["ID Clients"], df["Date"], df["Montant"], df["Type de transaction"],
            df["Localisation"], classes, p_fraude, niveaux)
    ]

    resultat = {
        "scenario": scenario,
        "fichier": file.filename,
        "n_lignes_fichier": int(n_brut),
        "n_analysees": int(len(df)),
        "n_clients": int(df["ID Clients"].nunique()),
        "periode": [df["Date"].min().strftime("%d/%m/%Y"), df["Date"].max().strftime("%d/%m/%Y")],
        "montant_total": float(df["Montant"].sum()),
        "repartition": {c: int((classes == c).sum()) for c in model.classes},
        "n_alertes": int(alerte.sum()),
        "montant_a_risque": float(df.loc[p_fraude >= 0.3, "Montant"].sum()),
        "colonnes": ["client", "date", "montant", "type", "ville", "classe", "p_fraude", "niveau"],
        "lignes": lignes,
    }

    # Verite terrain disponible : on evalue reellement le modele sur ce fichier.
    if "Target" in df.columns:
        reel = df["Target"].astype(str).str.strip()
        valides = reel.isin(model.classes)
        if valides.any():
            r, pr = reel[valides], pd.Series(classes, index=df.index)[valides]
            matrice = [[int(((r == a) & (pr == b)).sum()) for b in model.classes]
                       for a in model.classes]
            par_classe = {}
            for c in model.classes:
                vp = int(((r == c) & (pr == c)).sum())
                fp = int(((r != c) & (pr == c)).sum())
                fn = int(((r == c) & (pr != c)).sum())
                par_classe[c] = {
                    "precision": vp / (vp + fp) if vp + fp else 0.0,
                    "recall": vp / (vp + fn) if vp + fn else 0.0,
                    "support": int((r == c).sum()),
                }
            resultat["evaluation"] = {
                "classes": model.classes,
                "n_evaluees": int(valides.sum()),
                "accuracy": float((r == pr).mean()),
                "confusion": matrice,
                "par_classe": par_classe,
            }

    return resultat


@app.get("/", include_in_schema=False)
def root():
    """Sert le tableau de bord a la racine.

    L'interface et l'API partagent ainsi la meme origine : une seule URL a
    diffuser, et aucune configuration manuelle cote navigateur.
    """
    page = Path(__file__).parent / "demo.html"
    if page.exists():
        return FileResponse(page)
    return {
        "message": "API de détection de fraude bancaire — voir /docs pour la documentation interactive",
        "endpoints": ["/health", "/predict/scenario1", "/predict/scenario2",
                      "/meta/{scenario}", "/rapport/{scenario}"],
    }
