import os
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

def run_pipeline():
    print("Démarrage du pipeline Batch...")
    
    # 1. Chargement des données (Volume monté dans Docker)
    data_path = "/app/data/processed/tickets_clean.csv"
    if not os.path.exists(data_path):
        print(f"Erreur : {data_path} introuvable.")
        return

    df = pd.read_csv(data_path)
    
    # 2. Chargement du modèle d'embedding
    print("Génération des embeddings...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(df["clean_text"].astype(str).tolist())
    embeddings_norm = normalize(embeddings)

    # 3. Chargement du Classifieur et LabelEncoder
    print("Chargement des modèles ML...")
    clf = joblib.load("src/ml/ticket_classifier_rf.pkl")
    le = joblib.load("src/ml/label_encoder.pkl")

    # 4. Inférence (Prédiction)
    predictions = clf.predict(embeddings_norm)
    df["predicted_type"] = le.inverse_transform(predictions)

    # 5. Sauvegarde des résultats
    output_path = "data/processed/predictions_results.csv"
    df.to_csv(output_path, index=False)
    print(f"Terminé ! Résultats sauvegardés dans {output_path}")

if __name__ == "__main__":
    run_pipeline()