# Utiliser une image de base Python
FROM python:3.8-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers nécessaires dans le conteneur
COPY requirements.txt .
COPY src/ ./src/
COPY tests/ ./tests/
COPY data/ ./data/

# Installer les dépendances Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Exposer le port sur lequel l'application va écouter (si vous avez une API)
EXPOSE 5000

# Commande par défaut pour exécuter l'application
CMD ["python3", "src/main.py", "--train-data", "data/train.csv", "--test", "data/test.csv", "--train"]
