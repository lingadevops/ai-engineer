import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

# Load the synthetic health insurance claims data
df = pd.read_csv('synthetic_health_claims.csv')

mlflow.set_tracking_uri("http://localhost:5000")  # Set your MLflow tracking URI

#features to use for the model
features = ['claim_amount', 'num_services', 'age', 'provider_id', 'days_since_last_claim']

# Split the data into training and testing sets
X_train, X_test = train_test_split(df[features], test_size=0.2, random_state=42)

#setup MLflow experiment
mlflow.set_experiment("Health Insurance Claims Anomaly Detection")

with mlflow.start_run():
    # Train the Isolation Forest model
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X_train)

    #predict on test set
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    #convert predictions to anomaly labels (1 for normal, -1 for anomaly)
    anomaly_score_train = (y_pred_train == -1).astype(int)
    anomaly_score_test = (y_pred_test == -1).astype(int)

    # Log the parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("contamination", 0.05)

    #log the metrics
    train_anomaly_percentage = anomaly_score_train.mean()*100
    test_anomaly_percentage = anomaly_score_test.mean()*100
    mlflow.log_metric("train_anomaly_percentage", train_anomaly_percentage)
    mlflow.log_metric("test_anomaly_percentage", test_anomaly_percentage)

    # Log the model
    mlflow.sklearn.log_model(model, "model")
    print(f"Model trained and logged with MLflow. Train anomaly percentage: {train_anomaly_percentage:.2f}%, Test anomaly percentage: {test_anomaly_percentage:.2f}%")

