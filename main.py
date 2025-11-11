import json
import os.path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import optuna

MODEL_PATH = "alzheimers_model.pkl"
PARAMS_PATH = "optuna_params.json"


def load_data(filename):
    df = pd.read_csv(filename)
    X = df.drop("Diagnosis", axis=1)
    y = df["Diagnosis"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
    return X_train, X_test, y_train, y_test


def feature_engineering():
    # AI Disclosure: AI was used to determine the numerical and categorical features and turn them into a list
    numerical_features = ["Age", "BMI", "AlcoholConsumption", "PhysicalActivity",
                          "DietQuality", "SleepQuality",
                          "SystolicBP", "DiastolicBP", "CholesterolTotal",
                          "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides",
                          "MMSE", "FunctionalAssessment", "ADL"]

    categorical_features = ["Smoking", "FamilyHistoryAlzheimers", "CardiovascularDisease",
                            "Diabetes", "Depression", "HeadInjury", "Hypertension",
                            "MemoryComplaints", "BehavioralProblems",
                            "Confusion", "Disorientation", "PersonalityChanges",
                            "DifficultyCompletingTasks", "Forgetfulness"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    return preprocessor


def objective(trial, X_train, y_train, preprocessor):
    # AI Disclosure: ChatGPT was used to suggest Optuna parameter ranges
    n_estimators = trial.suggest_int("n_estimators", 100, 400)
    max_depth = trial.suggest_int("max_depth", 5, 50)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

    model = RandomForestClassifier(
        random_state=41,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight="balanced",
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="recall")
    return scores.mean()


def train_model(X_train, X_test, y_train, y_test, preprocessor):
    if os.path.exists(MODEL_PATH) and os.path.exists(PARAMS_PATH):
        pipe = joblib.load(MODEL_PATH)
    else:
        study = optuna.create_study(direction="maximize")
        # AI Disclosure: AI simplified this one line of code
        study.optimize(lambda trial: objective(trial, X_train, y_train, preprocessor), n_trials=20,
                       show_progress_bar=True)

        best_params = study.best_params
        with open(PARAMS_PATH, "w") as f:
            json.dump(best_params, f)

        # AI Disclosure: AI gave me the syntax for the **best_params for easily passing them in
        best_model = RandomForestClassifier(**best_params, random_state=41, n_jobs=-1, class_weight="balanced")
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", best_model)
        ])

        pipe.fit(X_train, y_train)
        joblib.dump(pipe, MODEL_PATH)

    y_preds = pipe.predict(X_test)
    acc = recall_score(y_test, y_preds)
    print("Recall score on test data", acc)
    print("Classification report:")
    print(classification_report(y_test, y_preds))


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data("alzheimers_disease_data.csv")
    preprocessor = feature_engineering()
    train_model(X_train, X_test, y_train, y_test, preprocessor)
