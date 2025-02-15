import argparse
from model_pipeline import prepare_data, train_model, save_model, load_model, evaluate_model, predict
from sklearn.metrics import accuracy_score
import numpy as np

def prepare_only(train_path, test_path):
    X_train, X_test, y_train, y_test, X_cluster, y_cluster = prepare_data(train_path, test_path)
    print("\n✅ Data Preparation Completed!")
    print(f"📊 X_train shape: {X_train.shape}")
    print(f"📊 X_test shape: {X_test.shape}")

def main(train_path, test_path, prepare_only_flag=False, predict_flag=False):
    if prepare_only_flag:
        prepare_only(train_path, test_path)
    elif predict_flag:
        print("\n🎯 Running Prediction Mode...")

        # Charger le modèle
        loaded_model = load_model()

        # Simuler des features pour la prédiction (remplace par des vraies données si nécessaire)
        sample_features = np.random.rand(1, 8)  # Remplace 10 par la bonne dimension

        # Faire la prédiction
        prediction = predict(sample_features)
        print(f"\n🎯 Prediction Result: {prediction}")
    else:
        X_train, X_test, y_train, y_test, X_cluster, y_cluster = prepare_data(train_path, test_path)
        print("\n✅ Data Preparation Completed!")

        print("\n🚀 Training Model...")
        
        # Entraîner le modèle
        model = train_model(X_train, y_train)
        
        # Sauvegarde du modèle
        save_model(model)

        # Chargement du modèle
        loaded_model = load_model()

        # Prédire sur les données de test
        y_pred = loaded_model.predict(X_test)

        # Évaluer le modèle
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n✅ Model Training Completed! Accuracy: {accuracy:.4f}")

        print("\n📊 Evaluating the model...")
        evaluate_model(model, X_test, y_test)
        print("✅ Model evaluation successful!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the prepare_data function")
    parser.add_argument("--train", type=str, required=True, help="Path to the training CSV file")
    parser.add_argument("--test", type=str, required=True, help="Path to the test CSV file")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--prepare", action="store_true", help="Only prepare the data, don't train the model")
    parser.add_argument("--predict", action="store_true", help="Run a prediction using a trained model")

    args = parser.parse_args()
    
    # Vérifier que train et test sont requis sauf si on fait uniquement une prédiction
    if not args.predict and (args.train is None or args.test is None):
        parser.error("❌ Les arguments --train et --test sont requis sauf si --predict est utilisé.")

    main(args.train, args.test, prepare_only_flag=args.prepare, predict_flag=args.predict)