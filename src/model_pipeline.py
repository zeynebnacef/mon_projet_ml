import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import f_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def prepare_data(train_path, test_path):
    """
    Prepare the dataset for training and testing.
    """
    # 1. Load data
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    # 2. Copy data for preparation
    df_prep = df.copy()

    # 3. Convert binary categorical variables
    df_prep['International plan'] = df_prep['International plan'].map({'Yes': 1, 'No': 0})
    df_prep['Voice mail plan'] = df_prep['Voice mail plan'].map({'Yes': 1, 'No': 0})
    df_prep['Churn'] = df_prep['Churn'].astype(int)

    # 4. Target Encoding for 'State'
    target_mean = df_prep.groupby('State')['Churn'].mean()
    df_prep['STATE_TargetMean'] = df_prep['State'].map(target_mean)

    # 5. Label Encoding for 'State'
    label_encoder = LabelEncoder()
    df_prep['STATE_Label'] = label_encoder.fit_transform(df_prep['State'])
    df_prep = df_prep.drop(columns=['State'])

    # 6. Remove highly correlated columns
    corr_data = df_prep.corr()
    upper_triangle = corr_data.where(np.triu(np.ones(corr_data.shape), k=1).astype(bool))
    high_correlation_columns = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)]
    df_prep_dropped = df_prep.drop(columns=high_correlation_columns)

    # 7. Clip extreme values
    lower_limit = df_prep_dropped.quantile(0.05)
    upper_limit = df_prep_dropped.quantile(0.95)
    df_prep_clipped = df_prep_dropped.apply(lambda x: x.clip(lower_limit[x.name], upper_limit[x.name]))

    # 8. Separate features and target
    df_classif = df_prep_clipped.copy()
    X = df_classif.drop(columns=['Churn'])
    y = df_classif['Churn']

    df_cluster = df_prep_clipped.copy()
    X_cluster = df_cluster.drop(columns=['Churn'])
    y_cluster = df_cluster['Churn']

    # 9. Balance data with SMOTE
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
    X_cluster, y_cluster = smote.fit_resample(X_cluster, y_cluster)

    # 10. Feature selection with ANOVA F-score
    F_scores, p_values = f_classif(X, y)
    scores_df = pd.DataFrame({'Feature': X.columns, 'F-Score': F_scores, 'P-Value': p_values})
    significant_features = scores_df[scores_df['P-Value'] < 0.05]['Feature'].tolist()

    # 11. Drop non-significant columns
    columns_to_drop = ['STATE_TargetMean', 'STATE_Label', 'Account length',
                       'Total night calls', 'Area code', 'Total day calls',
                       'Total eve calls']
    X = X.drop(columns=columns_to_drop, errors='ignore')
    X_cluster = X_cluster.drop(columns=columns_to_drop, errors='ignore')

    # 12. Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 13. Normalize data with StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_cluster_scaled = scaler.fit_transform(X_cluster)

    return X_train_scaled, X_test_scaled, y_train, y_test, X_cluster_scaled, y_cluster


def train_model(X_train, y_train):
    """
    Train a Gradient Boosting Classifier with hyperparameter optimization.
    """
    # Define hyperparameter search space
    param_dist = {
        'n_estimators': randint(50, 200),
        'learning_rate': uniform(0.01, 0.2),
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }

    # Initialize the model
    gb_model = GradientBoostingClassifier(random_state=42)

    # Optimize hyperparameters with RandomizedSearchCV
    random_search = RandomizedSearchCV(
        gb_model,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )

    # Train the model
    random_search.fit(X_train, y_train)

    # Log hyperparameters and metrics to MLflow
    best_params = random_search.best_params_
    
    # Return the best model
    return random_search.best_estimator_


def evaluate_model(model, X_test, y_test):
  
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

        # Plot and log confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
 

    # Print results
    print(f"\n✅ Evaluation Completed!")
    print(f"📊 Accuracy: {accuracy:.4f}")
    print(f"📊 Classification Report:\n{report}")
    print(f"📊 Confusion Matrix:\n{cm}")


def save_model(model, filename="gbm_model.joblib"):
   
    joblib.dump(model, filename)
    print(f"\n💾 Model saved to '{filename}' and logged as an artifact.")


def load_model(filename="gbm_model.joblib"):
    """
    Load a saved model.
    """
    model = joblib.load(filename)
    print(f"\n📂 Model loaded from '{filename}'")
    return model
def predict(features):
    model = joblib.load("gbm_model.joblib")
    prediction = model.predict(features)
    print(f"\n✅ Prediction Completed! Prediction: {prediction}")
    return prediction
