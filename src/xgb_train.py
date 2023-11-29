import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.paths import predict_performance_path

from predict_performance.pp_utils import (
    load_model_performance,
    vector_dict_process,
    fill_vectors,
    prepare_datasets,
)


def train_xgboost_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = xgb.XGBRegressor(objective="reg:squarederror")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {
        "y_test": y_test,
        "predictions": predictions,
        "mse": mse,
        "mae": mae,
        "r2": r2,
    }


def main(dataset_name, metrics):
    performance_df = load_model_performance()
    vector_df = vector_dict_process(dataset_name)
    joined_df = fill_vectors(performance_df, vector_df)
    joined_df, df_tuned = prepare_datasets(joined_df)

    all_results = []

    for metric in metrics:
        # Stack the vectors for the Train + Test model
        X1 = np.array(
            [v for v in joined_df["Train_Test_Vectors"].values if v is not None]
        )
        y1 = joined_df[joined_df["Train_Test_Vectors"].notnull()][metric].values
        results1 = train_xgboost_model(X1, y1)
        results1["model"] = "train_test"
        results1["metric"] = metric

        # Stack the vectors for the Train + Tune + Test model
        X2 = np.array(
            [v for v in df_tuned["Train_Tune_Test_Vectors"].values if v is not None]
        )
        y2 = df_tuned[df_tuned["Train_Tune_Test_Vectors"].notnull()][metric].values
        results2 = train_xgboost_model(X2, y2)
        results2["model"] = "train_tune_test"
        results2["metric"] = metric

        # join the results but add column for train_test or train_tune_test
        all_results.append(results1)
        all_results.append(results2)

    results_df = pd.DataFrame(all_results)
    return results_df


# ----------------------------------------------------------------------------------------------

if __name__ == "__main__":
    METRICS = ["Accuracy", "AUC-ROC", "Error Rate", "Calibration Error"]
    results_df = main("eicu_mortality", METRICS)
    results_df.to_pickle(predict_performance_path())
