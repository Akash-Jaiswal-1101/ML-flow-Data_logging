import mlflow
import mlflow.sklearn

# import dagshub
# dagshub.init(repo_owner='Akash-Jaiswal-1101', repo_name='mlflow-dagshub-demo', mlflow=True)

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("iris-rf-experiment")
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the iris dataset
iris = load_iris()
X = iris.data   
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameters for the Random Forest model
max_depth = 5
n_estimators = 50 

mlflow.autolog()
# Start an MLflow run
with mlflow.start_run():
    # Train the Random Forest model
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)

    # Make predictions
    y_pred = rf.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Log the confusion matrix as an artifact
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact(__file__)

    # mlflow.sklearn.log_model(rf, name="random_forest_model")
    mlflow.set_tag("author", "king-akash")
    mlflow.set_tag("model_type", "Random Forest")


    