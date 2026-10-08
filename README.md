# BMI / Health Metric Prediction

This project trains a Linear Regression model to predict **BMI** from the supplied `ps capstone dataset.xlsx` file.

## Data mapping

| Requested input | Dataset field | Treatment |
| --- | --- | --- |
| BMI (target) | `bmi` | Predicted health metric |
| Age | `age` | Used directly |
| Exercise hours | `duration_minutes` | Converted from minutes to hours |
| Sleep duration | `sleep_hours` | Used directly |
| Diet score | Not available | Not present; deliberately not fabricated |

If a later dataset version supplies a `diet_score` column, the script uses it automatically.

## Run

Install the requirements, then run:

```powershell
py -m pip install -r requirements.txt
py bmi_regression.py
```

The script uses an 80/20 train-test split with a fixed random seed (`42`) and reports these test-set metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R^2 score

It also prints the learned coefficient for each predictor and an example BMI prediction based on the dataset's median profile.
