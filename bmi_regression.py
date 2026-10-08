"""Train and evaluate a BMI linear-regression model from the supplied workbook.

The input data contains BMI, age, sleep_hours, and duration_minutes.  The
workbook does not contain a diet-score field, so this script does not create a
synthetic one.  If a future version contains ``diet_score``, it is included
automatically.
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DATA_FILE = Path("ps capstone dataset.xlsx")
TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_and_prepare_data(path: Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load the first worksheet and produce model-ready BMI features."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    df = pd.read_excel(path)
    required_columns = {"bmi", "age", "sleep_hours", "duration_minutes"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Required columns are missing: {sorted(missing)}")

    # The source stores exercise duration in minutes; express it in hours as
    # requested for the health-metric regression model.
    model_data = df[["bmi", "age", "sleep_hours", "duration_minutes"]].copy()
    model_data["exercise_hours"] = model_data.pop("duration_minutes") / 60

    features = ["age", "exercise_hours", "sleep_hours"]
    if "diet_score" in df.columns:
        model_data["diet_score"] = df["diet_score"]
        features.insert(0, "diet_score")
    else:
        print("Note: diet_score is not present in this dataset and was omitted.")

    model_data = model_data.dropna(subset=["bmi", *features])
    return model_data[features], model_data["bmi"], features


def main() -> None:
    X, y, feature_names = load_and_prepare_data(DATA_FILE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse**0.5
    r2 = r2_score(y_test, predictions)

    print(f"Records used: {len(X)} (train: {len(X_train)}, test: {len(X_test)})")
    print("Features used:", ", ".join(feature_names))
    print("\nTest-set evaluation")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2:  {r2:.4f}")

    coefficients = pd.DataFrame(
        {"feature": feature_names, "coefficient": model.coef_}
    )
    print(f"\nIntercept: {model.intercept_:.4f}")
    print("Coefficients:")
    print(coefficients.to_string(index=False))

    # Example prediction using the median profile in the dataset.
    example = X.median().to_frame().T
    print(f"\nExample BMI prediction for a median profile: {model.predict(example)[0]:.2f}")


if __name__ == "__main__":
    main()
