FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# train_from_raw.py est requis a l'execution : l'analyse par lot reutilise ses
# fonctions de construction des features.
COPY app.py model_lib.py train_from_raw.py demo.html ./
COPY model_artifacts ./model_artifacts

EXPOSE 8000

# Les hebergeurs (Render, Railway, Fly.io) imposent le port via la variable PORT.
# Forme "shell" indispensable pour que ${PORT} soit substitue au demarrage ;
# la forme JSON ne fait aucune expansion de variable.
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
