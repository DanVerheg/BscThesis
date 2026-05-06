import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import shap 
import os

results = "results_noyear.csv"

df = pd.read_csv("million_spotify_data.csv")
df = df.drop(columns = ["track_id", "artist_name", "track_name", "Unnamed: 0", "year"])
df = pd.get_dummies(df, columns=["key", "time_signature"], drop_first=True)

#print(df.columns.values)
#print(df['genre'].value_counts())

rock_df = df[df['genre'] == "rock"]
electronic_df = df[df['genre'] == "electronic"]
hiphop_df = df[df['genre'] == "hip-hop"]
folk_df = df[df['genre'] == "folk"]
pop_df = df[df['genre'] == "pop"]
classical_df = df[df['genre'] == "classical"]
jazz_df = df[df['genre'] == "jazz"]
country_df = df[df['genre'] == "country"]
soul_df = df[df['genre'] == "soul"]
blues_df = df[df['genre'] == "blues"]

subsets = {
    "rock": rock_df,
    "electronic": electronic_df,
    "hip-hop": hiphop_df,
    "folk": folk_df,
    "pop": pop_df,
    "classical": classical_df,
    "jazz": jazz_df,
    "country": country_df,
    "soul": soul_df,
    "blues": blues_df
}

#for name, df in subsets.items(): 
#    df.to_csv(f"{name}_subset.csv", index=False)

balanced_datasets = {}

for genre , subset in subsets.items(): #randomly samples all genres to match the smallest genre sample size 
    balanced_datasets[genre] = subset.sample(n = 3319, random_state = 42)
    
#for name, df in balanced_datasets.items(): 
#    df.to_csv(f"{name}_subset.csv", index=False)

#combines the balanced subsets into a new global dataset
balanced_datasets["global"] = pd.concat(balanced_datasets.values(), ignore_index = True) 

# check which genres are already completed so i can resume if the script is interrupted
completed_genres = set()
if os.path.exists(results):
    completed_genres = set(pd.read_csv(results, index_col="model").index.tolist())
    print(f"already completed: {completed_genres}")

for genre, dataset in balanced_datasets.items(): 

    # skip genre if already completed
    if genre in completed_genres:
        print(f"Skipping {genre}")
        continue

    #splits into the training and testing subsets
    if genre == "global":
        train_df, test_df = train_test_split(dataset, test_size = 0.2, random_state = 42, stratify = dataset["genre"]) 
    else: 
        train_df, test_df = train_test_split(dataset, test_size = 0.2, random_state = 42) 
        
    #These check the genre distribution of the subsets 
    #print(train_df["genre"].value_counts(normalize=True)) 
    #print(val_df["genre"].value_counts(normalize=True))
    #print(test_df["genre"].value_counts(normalize=True))

    #Measure mean and standard deviation of the popularity for each genre
    stats_per_genre = train_df.groupby("genre")["popularity"].agg(["mean", "std"])
    #print(stats_per_genre)

    #creates the training and testing split
    X_train = train_df.drop(columns=["popularity", "genre"])
    y_train = train_df["popularity"]

    X_test = test_df.drop(columns=["popularity", "genre"])
    y_test = test_df["popularity"]
    
    #parameters for the ExtraTreesRegressor 
    param_grid = {
    "n_estimators": [100, 300, 500, 700, 1000],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None]
}
    model = ExtraTreesRegressor(random_state=42)

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        n_jobs= -1, 
        verbose=2,
        cv = 5 #2 for testing 5 for the real deal 
    )

    #tunes the hyperparameters, fits the new test model 
    grid.fit(X_train, y_train)
    #print(grid.best_params_)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    
    #saves the results and other important values to the results dataframe
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred) 
    avg_std = stats_per_genre["std"].mean()
    
    #compute shap values for each feature 
    print("Starting SHAP")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)
    shap_importance = np.mean(np.abs(shap_values), axis=0)
    shap_vector = pd.Series(shap_importance, index=X_test.columns)
    print("SHAP Finished ")

    row = {
        "model": genre,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "n_estimators": grid.best_params_["n_estimators"],
        "max_depth": grid.best_params_["max_depth"],
        "min_samples_split": grid.best_params_["min_samples_split"],
        "min_samples_leaf": grid.best_params_["min_samples_leaf"],
        "max_features": grid.best_params_["max_features"],
    }

    for feature, importance in shap_vector.items():
        row[f"shap_{feature}"] = importance

    # Append this genre's row to the results file immediately
    row_df = pd.DataFrame([row]).set_index("model")
    if os.path.exists(results):
        row_df.to_csv(results, mode="a", header=False, index_label="model")
    else:
        row_df.to_csv(results, mode="w", header=True, index_label="model")
    print(f"Saved results for {genre}")