import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("wine-classification")

data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)

params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}

with mlflow.start_run(run_name="rf_n100_d5"):
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")

    
    mlflow.log_params(params)
    # 2. метрики
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_macro", f1)
    
    joblib.dump(model, "model.joblib")
    mlflow.log_artifact("model.joblib")
    mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name="WineClassifier",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    print(f"accuracy={acc:.3f}, f1_macro={f1:.3f}")