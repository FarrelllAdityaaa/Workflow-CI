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

# Set experiment
mlflow.set_experiment("CI_Retraining_Run")

# Autolog
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="CI_Base_Model"):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    print("Model CI Basic trained with Autolog.")