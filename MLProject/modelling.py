import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load Data
df = pd.read_csv('titanic_preprocessing.csv')
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    print("Model Basic trained with Autolog.")

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )
    print("Model logged to MLflow.")

    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)