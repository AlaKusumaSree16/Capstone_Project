from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_all_models(X_train, y_train):
    models = {}

    # Logistic Regression
    models["Logistic Regression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500))
    ])

    # Decision Tree
    models["Decision Tree"] = DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    )

    # Random Forest
    models["Random Forest"] = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42
    )

    # SVM
    models["SVM"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf"))
    ])

    # Train all models
    for model in models.values():
        model.fit(X_train, y_train)

    return models
