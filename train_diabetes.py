from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
df = pd.read_csv('https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master...')

# Splitting data into features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the RandomForestClassifier model
rf = RandomForestClassifier(random_state=42)

# Defining the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30]
}

# Applying GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

mlflow.set_experiment('diabetes-rf-hp')


with mlflow.start_run():
    grid_search.fit(X_train, y_train)

    # Displaying the best parameters and the best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    #params
    mlflow.log_params(best_params)


    #metrics
    mlflow.log_metric("accuracy", best_score)

    #data
    train_df = X_train
    train_df['Outcome'] = y_train
    mlflow.data.from_pandas(train_df)
    mlflow.log_inputs(train_df,"Training Data")

    test_df = X_test
    test_df['Outcome'] = y_test 
    mlflow.data.from_pandas(test_df)
    mlflow.log_inputs(test_df,"Testing Data")


    #source code
    mlflow.log_artifact(__file__)

    #model
    mlflow.sklearn.log_model(grid_search.best_estimator_, "random_forest_model")


    #tags
    mlflow.set_tag("author", "king-akash")

print(f"Best Parameters: {best_params}")
print(f"Best Score: {best_score}")