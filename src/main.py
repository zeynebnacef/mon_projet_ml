import argparse
import mlflow
import mlflow.sklearn
from model_pipeline import prepare_data, train_model, save_model, check_or_assign_model_stage, load_model, evaluate_model, predict
from sklearn.metrics import accuracy_score
import numpy as np
# Configure MLflow to use PostgreSQL as the backend store

mlflow.set_tracking_uri("postgresql://mlflow_user:zeyneb@localhost:5432/mlflow_db2")

mlflow.set_experiment("new_experiment")

def prepare_only(train_path, test_path):
    """
    Prepare the data and log the shapes to MLflow.
    """
    X_train, X_test, y_train, y_test, X_cluster, y_cluster = prepare_data(train_path, test_path)
    print("\n✅ Data Preparation Completed!")
    print(f"📊 X_train shape: {X_train.shape}")
    print(f"📊 X_test shape: {X_test.shape}")

    # Log data preparation details to MLflow
    with mlflow.start_run():
        mlflow.log_param("X_train_shape", X_train.shape)
        mlflow.log_param("X_test_shape", X_test.shape)

def main(train_path, test_path, prepare_only_flag=False, predict_flag=False):
    if prepare_only_flag:
        prepare_only(train_path, test_path)
    elif predict_flag:
        print("\n🎯 Running Prediction Mode...")

        # Load the trained model
        loaded_model = load_model()

        # Simulate sample features for prediction (replace with actual data if needed)
        sample_features = np.random.rand(1, 8)  # Replace 8 with the correct feature dimension

        # Enregistrer le modèle dans MLflow
        with mlflow.start_run():
            mlflow.sklearn.log_model(loaded_model, "model")

            # Ajouter le modèle au Model Registry
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mlflow.register_model(model_uri, "gbm_model_predict")
            # Make a prediction
            prediction = predict(sample_features)
            print(f"\n🎯 Prediction Result: {prediction}")

       
    else:
        # Prepare the data
        X_train, X_test, y_train, y_test, X_cluster, y_cluster = prepare_data(train_path, test_path)
        print("\n✅ Data Preparation Completed!")

        # 🚀 *Lancement d'une expérience MLflow*
        with mlflow.start_run():
            print("\n🚀 Training Model...")
            
            # Entraîner le modèle
            model = train_model(X_train, y_train)
            
            # Logger les hyperparamètres
            mlflow.log_param("model_type", "RandomForest")  # Exemple, adapte selon ton modèle
            mlflow.log_param("train_size", X_train.shape[0])
            mlflow.log_param("test_size", X_test.shape[0])



            # Save the model
            save_model(model)

            # Load the model
            loaded_model = load_model()

            # Make predictions on the test data
            y_pred = loaded_model.predict(X_test)

            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            print(f"\n✅ Model Training Completed! Accuracy: {accuracy:.4f}")

            # Log the accuracy metric to MLflow
            mlflow.log_metric("accuracy", accuracy)

            # Log the model to MLflow
            mlflow.sklearn.log_model(loaded_model, "model")

            # Register the model in the Model Registry
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mlflow.register_model(model_uri, "gbm_model")

            # Evaluate the model and log additional metrics
            print("\n📊 Evaluating the model...")
            evaluate_model(model, X_test, y_test)
            print("✅ Model evaluation successful!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the prepare_data function")
    parser.add_argument("--train", type=str, required=True, help="Path to the training CSV file")
    parser.add_argument("--test", type=str, required=True, help="Path to the test CSV file")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--prepare", action="store_true", help="Only prepare the data, don't train the model")
    parser.add_argument("--predict", action="store_true", help="Run a prediction using a trained model")

    args = parser.parse_args()

    # Validate that --train and --test are required unless --predict is used
    if not args.predict and (args.train is None or args.test is None):
        parser.error("❌ The arguments --train and --test are required unless --predict is used.")

    # Check or assign model stage
    check_or_assign_model_stage(
        model_name="gbm_model",
        version=48,
        stage="Staging"  # Change to 'Production' if needed
    )

    # Call the main function
    main(args.train, args.test, prepare_only_flag=args.prepare, predict_flag=args.predict)