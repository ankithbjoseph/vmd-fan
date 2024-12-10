import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sktime.transformations.series.vmd import VmdTransformer
from sktime.datatypes import check_raise
from sktime.split import temporal_train_test_split
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import torch
import logging
import torch.nn as nn
import numpy as np
from itertools import product
from sklearn.metrics import mean_absolute_error, mean_squared_error
from models.FAN import FANForecastingModel
from models.BaseNN import BaselineNN
from joblib import Parallel, delayed
import json
import warnings


warnings.filterwarnings("ignore")


def save_plot(data, title, xlabel, ylabel, save_path, visualize):
    """
    Save a plot to the specified directory if visualize is True.
    """
    if visualize:
        plt.figure(figsize=(10, 6))
        plt.plot(data, marker="o")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved to {save_path}")


def load_dataset(file_path):
    """
    Load dataset from the given file path

    """

    print("Loading dataset...")
    data = pd.read_csv(file_path)
    print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
    return data


def apply_eda(data, output_dir, visualize):
    """
    Perform exploratory data analysis on the dataset and generate visualizations if enabled.
    """
    print("\nDataset Info:")
    print(data.info())

    print("\nMissing Data:")
    print(data.isnull().sum())

    print("\nUnique Values in 'sitename':", data["sitename"].nunique())

    if visualize:
        os.makedirs(output_dir, exist_ok=True)
        save_plot(
            data["aqi"],
            "AQI Time Series",
            "Date",
            "AQI",
            os.path.join(output_dir, "eda_aqi_timeseries.png"),
            visualize,
        )


def clean_data(data, variable, output_dir, visualize):
    """
    Clean the dataset by removing missing values and negative values for the selected variable.
    Generate visualizations for the cleaned data.
    """
    print("\nCleaning data...")
    data = data[["date", "sitename", variable]].copy()
    data = data.dropna(subset=[variable])
    data = data[data[variable] > 0]
    print(f"Data cleaned. Remaining rows: {data.shape[0]}")

    if visualize:
        os.makedirs(output_dir, exist_ok=True)
        save_plot(
            data[variable],
            f"Cleaned {variable} Time Series",
            "Date",
            variable,
            os.path.join(output_dir, f"cleaned_{variable}_timeseries.png"),
            visualize,
        )
    return data


def validate_data_format(data, variable):
    """
    Validate input data format for sktime compatibility.

    """
    # Ensure the index is a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data["date"], errors="coerce")
        except Exception as e:
            raise ValueError(f"Failed to set DatetimeIndex: {e}")

    # Drop rows with invalid timestamps
    if data.index.isnull().any():
        print(f"Found {data.index.isnull().sum()} invalid timestamps. Dropping them.")
        data = data[~data.index.isnull()]

    # Sort the index
    if not data.index.is_monotonic_increasing:
        print("Sorting index to ensure monotonically increasing order.")
        data = data.sort_index()

    # Handle duplicate timestamps
    if not data.index.is_unique:
        print("Duplicate timestamps detected. Keeping the first occurrence.")
        data = data[~data.index.duplicated(keep="first")]

    # Validate sktime-compatible format
    try:
        check_raise(data, "pd.DataFrame")
    except Exception as e:
        raise ValueError(f"Input data is not sktime-compatible: {e}")

    return data


def segment_by_city(data, variable, cities, output_dir):
    """
    Segment data by 'sitename' and save each city's data to separate CSV files.

    """
    print("\nSegmenting data by city...")

    cities_list = cities.split(",") if cities else data["sitename"].unique()
    data = data[data["sitename"].isin(cities_list)]
    grouped = data.groupby("sitename")
    for city, group in grouped:
        os.makedirs(f"{output_dir}/{city}", exist_ok=True)
        city_path = os.path.join(output_dir, f"{city}/{city}_{variable}.csv")
        group = validate_data_format(group, variable)
        group[variable].to_csv(city_path, index=True)
        print(f"Saved data for {city} to {city_path}")


def create_sequences(data, window_size, forecast_horizon):
    sequences = []
    targets = []
    for i in range(len(data) - window_size - forecast_horizon + 1):
        sequences.append(data[i : i + window_size])
        targets.append(data[i + window_size : i + window_size + forecast_horizon])

    sequences = np.array(sequences)
    targets = np.array(targets)

    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(
        targets, dtype=torch.float32
    )


def forecast_data(
    data, model, window_size, forecast_horizon, device, visualize, output_dir
):
    model.eval()
    X_test, _ = create_sequences(data.values, window_size, forecast_horizon)
    X_test = X_test.to(device)
    with torch.no_grad():
        predictions = model(X_test).cpu().numpy()

    if visualize:
        os.makedirs(output_dir, exist_ok=True)
        save_plot(
            predictions.flatten(),
            "Forecasted AQI",
            "Time Steps",
            "AQI",
            os.path.join(output_dir, "forecasted_aqi.png"),
            visualize,
        )
    return predictions


def forecast_imfs_data(imfs, fan_models, window_size, forecast_horizon, device):
    predictions = {}
    for imf, model in fan_models.items():
        model.eval()
        imf_data = imfs[imf].values
        X_test, _ = create_sequences(imf_data, window_size, forecast_horizon)
        X_test = X_test.to(device)
        with torch.no_grad():
            pred = model(X_test).cpu().numpy()  # Shape: (num_samples, forecast_horizon)

        # Flatten predictions into a single sequence for compatibility
        pred_flattened = pred.sum(axis=1)  # Aggregate predictions over the horizon
        predictions[imf] = pred_flattened

    return pd.DataFrame(
        predictions, index=imfs.index[window_size + forecast_horizon - 1 :]
    )


def apply_vmd(city, data, variable, parameter_grid, output_dir, num_jobs, visualize):
    """
    Apply VMD transformation to the city's data for the selected variable in parallel.
    """
    save_dir = os.path.join(output_dir, city, "vmd_decompositions")
    os.makedirs(save_dir, exist_ok=True)

    def process_params(K, alpha, tau, DC, tol):
        params = {
            "K": K,
            "alpha": alpha,
            "tau": tau,
            "DC": DC,
            "tol": tol,
        }
        # print(f"Experiment: [ City:{city}, {params} ]")
        sub_dir = os.path.join(save_dir, f"K_{K}")
        os.makedirs(sub_dir, exist_ok=True)
        file_name = f"alpha_{alpha}_tau_{tau:.0e}_DC_{DC}_tol_{tol:.0e}.csv"
        full_path = os.path.join(sub_dir, file_name)

        if os.path.exists(full_path):
            print(f"Saved VMD decomposition for {city} with parameters: {params}")
            return

        try:
            vmd_transformer = VmdTransformer(**params)
            decomposed_data = vmd_transformer.fit_transform(data)
            decomposed_data.columns = [f"IMF_{i}" for i in range(1, K + 1)]
            decomposed_data.index = data.index

            decomposed_data.to_csv(full_path, index=True, chunksize=1000)
            print(f"Saved VMD decomposition for {city} with parameters: {params}")

            if visualize:
                for imf in decomposed_data.columns:
                    save_plot(
                        decomposed_data[imf],
                        f"{imf} Decomposition for {city}",
                        "Date",
                        "Value",
                        os.path.join(sub_dir, f"{imf}_decomposition.png"),
                        visualize,
                    )
        except Exception as e:
            print(f"Error processing {city} with parameters {params}: {e}")

    Parallel(n_jobs=num_jobs)(
        delayed(process_params)(K, alpha, tau, DC, tol)
        for K in parameter_grid["K"]
        for alpha in parameter_grid["alpha"]
        for tau in parameter_grid["tau"]
        for DC in parameter_grid["DC"]
        for tol in parameter_grid["tol"]
    )


def load_decomposition_file(base_dir, K, alpha, tau, DC, tol):
    """
    function to load a decomposition file based on given parameters.

    """

    if tau == 0:
        tau = "0e+00"

    K_folder = os.path.join(base_dir, f"K_{K}")
    filename = f"alpha_{alpha}_tau_{str(tau)}_DC_{DC}_tol_{str(tol)}.csv"
    file_path = os.path.join(K_folder, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        data = pd.read_csv(file_path)
        data["date"] = pd.to_datetime(data["date"])
        data.set_index("date", inplace=True)
        # print(f"File loaded successfully from: {file_path}")
        return data
    except Exception as e:
        raise Exception(f"An error occurred while loading the file: {e}")


def apply_fan(
    city, vmd, parameter_grid, output_dir, num_jobs, visualize, forecast_horizon
):
    """
    Apply FAN to each IMF of a city's data using the specified parameters.
    """

    num_epochs = 50
    batch_size = 512
    forecast_horizon = forecast_horizon

    results_file = f"{output_dir}/results.csv"

    existing_configs = set()
    if os.path.exists(results_file):
        existing_results = pd.read_csv(results_file)
        if not existing_results.empty:
            existing_configs = set(
                zip(
                    existing_results["City"],
                    existing_results["Model"],
                    existing_results["Parameters"],
                )
            )

    # Create results file with headers
    if not os.path.exists(results_file):
        with open(results_file, "w") as f:
            f.write("City,Model,Parameters,MAE,MSE,RMSE,MAPE\n")

    if vmd:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Model = "VMD+FAN"

        def process_vmdfan(
            window_size,
            p_ratio,
            fan_units,
            learning_rate,
            K,
            alpha,
            tau,
            DC,
            tol,
            city,
            Model,
        ):
            parameters = f"K:{K},alpha:{alpha},tau:{tau},DC:{DC},tol:{tol},window_size:{window_size}, learning_rate:{learning_rate}, p_ratio:{p_ratio}, fan_units:{fan_units}, forecast_horizon:{forecast_horizon}"

            print(f"Experiment: [ City:{city}, {parameters} ]")

            if (city, Model, parameters) in existing_configs:
                print(
                    f"Skipping already processed configuration for {city}, {Model}: {parameters}"
                )
                return

            file_name = f"{city}/{city}_aqi.csv"
            file_path = os.path.join(output_dir, file_name)

            try:
                city_data = pd.read_csv(file_path)
            except FileNotFoundError:
                return

            train_data, test_data = temporal_train_test_split(city_data, test_size=0.2)

            base_dir = f"./{output_dir}/{city}/vmd_decompositions"

            try:
                imfs_data = load_decomposition_file(base_dir, K, alpha, tau, DC, tol)
                imfs_train_data, imfs_test_data = temporal_train_test_split(
                    imfs_data, test_size=0.2
                )
            except Exception:
                return

            fan_models = {}

            for imf in imfs_train_data.columns:
                X_train, y_train = create_sequences(
                    imfs_train_data[imf].values, window_size, forecast_horizon
                )

                train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=False
                )

                model = FANForecastingModel(
                    input_dim=window_size,
                    output_dim=forecast_horizon,
                    p_ratio=p_ratio,
                    fan_units=fan_units,
                ).to(device)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                for epoch in range(num_epochs):
                    model.train()
                    for batch_X, batch_y in train_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()

                fan_models[imf] = model

            imf_predictions = forecast_imfs_data(
                imfs_test_data, fan_models, window_size, forecast_horizon, device
            )
            final_forecast = imf_predictions.sum(axis=1)

            # Metrics
            actual_aqi = np.array(
                [
                    test_data["aqi"]
                    .values[i + window_size : i + window_size + forecast_horizon]
                    .sum()
                    for i in range(len(final_forecast))
                ]
            )

            predicted_aqi = final_forecast.values

            mae = mean_absolute_error(actual_aqi, predicted_aqi)
            mse = mean_squared_error(actual_aqi, predicted_aqi)
            rmse = mse**0.5
            mape = np.mean(np.abs((actual_aqi - predicted_aqi) / actual_aqi)) * 100

            with open(results_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([city, Model, parameters, mae, mse, rmse, mape])

        param_combinations = [
            (window_size, p_ratio, fan_units, learning_rate, K, alpha, tau, DC, tol)
            for K in parameter_grid["VMD"]["K"]
            for alpha in parameter_grid["VMD"]["alpha"]
            for tau in parameter_grid["VMD"]["tau"]
            for DC in parameter_grid["VMD"]["DC"]
            for tol in parameter_grid["VMD"]["tol"]
            for window_size in parameter_grid["FAN"]["window_size"]
            for p_ratio in parameter_grid["FAN"]["p_ratio"]
            for fan_units in parameter_grid["FAN"]["fan_units"]
            for learning_rate in parameter_grid["FAN"]["learning_rate"]
        ]

        # Use joblib to parallelize the processing
        Parallel(n_jobs=num_jobs)(
            delayed(process_vmdfan)(
                window_size,
                p_ratio,
                fan_units,
                learning_rate,
                K,
                alpha,
                tau,
                DC,
                tol,
                city,
                Model,
            )
            for window_size, p_ratio, fan_units, learning_rate, K, alpha, tau, DC, tol in param_combinations
        )

    else:

        def process_fan(
            params, train_data, test_data, device, results_file, city, Model
        ):
            window_size, learning_rate, p_ratio, fan_units = params

            parameters = f"window_size:{window_size}, learning_rate:{learning_rate}, p_ratio:{p_ratio}, fan_units:{fan_units}, forecast_horizon:{forecast_horizon}"

            print(f"Experiment: [ City:{city}, {parameters} ]")

            if (city, Model, parameters) in existing_configs:
                print(
                    f"Skipping already processed configuration for {city}, {Model}: {parameters}"
                )
                return

            # Prepare training data
            X_train, y_train = create_sequences(
                train_data["aqi"].values, window_size, forecast_horizon
            )
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=False
            )

            # Initialize model
            model = FANForecastingModel(
                input_dim=window_size,
                output_dim=forecast_horizon,
                p_ratio=p_ratio,
                fan_units=fan_units,
            ).to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # Train model
            for epoch in range(num_epochs):
                model.train()
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            # Evaluate model on validation/test data
            predicted_aqi = forecast_data(
                test_data["aqi"],
                model,
                window_size,
                forecast_horizon,
                device,
                visualize,
                output_dir,
            )
            actual_aqi = np.array(
                [
                    test_data["aqi"].values[
                        i + window_size : i + window_size + forecast_horizon
                    ]
                    for i in range(len(predicted_aqi))
                ]
            )

            mae = mean_absolute_error(actual_aqi, predicted_aqi)
            mse = mean_squared_error(actual_aqi, predicted_aqi)
            rmse = mse**0.5
            mape = np.mean(np.abs((actual_aqi - predicted_aqi) / actual_aqi)) * 100

            # Log results
            with open(results_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([city, Model, parameters, mae, mse, rmse, mape])

            return params, mae, mse, rmse, mape

        Model = "FAN"
        # Define parameter grid
        fan_parameters = parameter_grid["FAN"]

        # Generate all combinations of parameters
        param_combinations = list(product(*fan_parameters.values()))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        file_name = f"{city}/{city}_aqi.csv"
        file_path = os.path.join(output_dir, file_name)

        try:
            city_data = pd.read_csv(file_path)
        except FileNotFoundError:
            return

        train_data, test_data = temporal_train_test_split(city_data, test_size=0.2)

        Parallel(n_jobs=num_jobs)(
            delayed(process_fan)(
                params, train_data, test_data, device, results_file, city, Model
            )
            for params in param_combinations
        )


def apply_baseNN(city, vmd, parameter_grid, output_dir, num_jobs, forecast_horizon):
    """
    Apply BaseNN to each IMF of a city's data using the specified parameters.
    """
    window_size = 12
    hidden_dim = 64
    learning_rate = 0.001
    num_epochs = 50
    batch_size = 512
    forecast_horizon = forecast_horizon

    results_file = f"{output_dir}/results.csv"

    existing_configs = set()
    if os.path.exists(results_file):
        existing_results = pd.read_csv(results_file)
        if not existing_results.empty:
            existing_configs = set(
                zip(
                    existing_results["City"],
                    existing_results["Model"],
                    existing_results["Parameters"],
                )
            )

    # Create results file with headers if it doesn't exist
    if not os.path.exists(results_file):
        with open(results_file, "w") as f:
            f.write("City,Model,Parameters,MAE,MSE,RMSE,MAPE\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if vmd:
        Model = "VMD+BaseNN"

        def process_vmdbasenn(
            window_size,
            hidden_dim,
            learning_rate,
            K,
            alpha,
            tau,
            DC,
            tol,
            city,
            Model,
        ):
            parameters = f"K:{K},alpha:{alpha},tau:{tau},DC:{DC},tol:{tol},window_size:{window_size},hidden_dim:{hidden_dim},learning_rate:{learning_rate},forecast_horizon:{forecast_horizon}"

            print(f"Experiment: [ City:{city}, {parameters} ]")

            if (city, Model, parameters) in existing_configs:
                print(
                    f"Skipping already processed configuration for {city}, {Model}: {parameters}"
                )
                return

            file_name = f"{city}/{city}_aqi.csv"
            file_path = os.path.join(output_dir, file_name)

            try:
                city_data = pd.read_csv(file_path)
            except FileNotFoundError:
                return

            train_data, test_data = temporal_train_test_split(city_data, test_size=0.2)

            base_dir = f"./{output_dir}/{city}/vmd_decompositions"

            try:
                imfs_data = load_decomposition_file(base_dir, K, alpha, tau, DC, tol)
                imfs_train_data, imfs_test_data = temporal_train_test_split(
                    imfs_data, test_size=0.2
                )
            except Exception:
                return

            base_models = {}

            for imf in imfs_train_data.columns:
                X_train, y_train = create_sequences(
                    imfs_train_data[imf].values, window_size, forecast_horizon
                )

                train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=False
                )

                model = BaselineNN(
                    input_dim=window_size,
                    hidden_dim=hidden_dim,
                    output_dim=forecast_horizon,
                ).to(device)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                for epoch in range(num_epochs):
                    model.train()
                    for batch_X, batch_y in train_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()

                base_models[imf] = model

            imf_predictions = forecast_imfs_data(
                imfs_test_data, base_models, window_size, forecast_horizon, device
            )
            final_forecast = imf_predictions.sum(axis=1)

            # Metrics
            actual_aqi = np.array(
                [
                    test_data["aqi"]
                    .values[i + window_size : i + window_size + forecast_horizon]
                    .sum()
                    for i in range(len(final_forecast))
                ]
            )

            predicted_aqi = final_forecast.values

            mae = mean_absolute_error(actual_aqi, predicted_aqi)
            mse = mean_squared_error(actual_aqi, predicted_aqi)
            rmse = mse**0.5
            mape = np.mean(np.abs((actual_aqi - predicted_aqi) / actual_aqi)) * 100

            with open(results_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([city, Model, parameters, mae, mse, rmse, mape])

        param_combinations = [
            (window_size, hidden_dim, learning_rate, K, alpha, tau, DC, tol)
            for K in parameter_grid["VMD"]["K"]
            for alpha in parameter_grid["VMD"]["alpha"]
            for tau in parameter_grid["VMD"]["tau"]
            for DC in parameter_grid["VMD"]["DC"]
            for tol in parameter_grid["VMD"]["tol"]
        ]

        # Use joblib to parallelize the processing
        Parallel(n_jobs=num_jobs)(
            delayed(process_vmdbasenn)(
                window_size,
                hidden_dim,
                learning_rate,
                K,
                alpha,
                tau,
                DC,
                tol,
                city,
                Model,
            )
            for window_size, hidden_dim, learning_rate, K, alpha, tau, DC, tol in param_combinations
        )

    else:
        # Original BaseNN without VMD
        Model = "BaseNN"
        parameters = f"window_size:{window_size}, hidden_dim:{hidden_dim}, learning_rate:{learning_rate}, forecast_horizon:{forecast_horizon}"

        if (city, Model, parameters) in existing_configs:
            print(
                f"Skipping already processed configuration for {city}, {Model}: {parameters}"
            )
            return

        file_name = f"{city}/{city}_aqi.csv"
        file_path = os.path.join(output_dir, file_name)

        try:
            city_data = pd.read_csv(file_path)
        except FileNotFoundError:
            return

        train_data, test_data = temporal_train_test_split(city_data, test_size=0.2)
        X_train, y_train = create_sequences(
            train_data["aqi"].values, window_size, forecast_horizon
        )
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False
        )

        model = BaselineNN(
            input_dim=window_size, hidden_dim=hidden_dim, output_dim=forecast_horizon
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        X_test, y_test = create_sequences(
            test_data["aqi"].values, window_size, forecast_horizon
        )
        X_test, y_test = X_test.to(device), y_test.to(device)

        model.eval()
        with torch.no_grad():
            predictions = model(X_test).cpu().numpy()
            actual_aqi = y_test.cpu().numpy()

        mae = mean_absolute_error(actual_aqi, predictions)
        mse = mean_squared_error(actual_aqi, predictions)
        rmse = mse**0.5
        mape = np.mean(np.abs((actual_aqi - predictions) / actual_aqi)) * 100

        with open(results_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([city, Model, parameters, mae, mse, rmse, mape])


def apply_arima(
    city, data, variable, output_dir, vmd=False, parameter_grid=None, num_jobs=1
):
    """
    Apply ARIMA with optional VMD decomposition and automatic (p, d, q) selection to forecast the given variable
    and evaluate its performance.
    """
    results_file = f"{output_dir}/results.csv"

    # Create results file with headers if it doesn't exist
    if not os.path.exists(results_file):
        with open(results_file, "w") as f:
            f.write("City,Model,Parameters,MAE,MSE,RMSE,MAPE\n")

    print(f"\nApplying ARIMA with auto (p, d, q) selection for city: {city}")

    # Split into training and testing data (80:20 split)
    train_data, test_data = temporal_train_test_split(data[variable], test_size=0.2)

    # Use auto_arima to find the best (p, d, q)
    print("Running auto_arima to find the best (p, d, q)...")
    arima_model = auto_arima(
        train_data,
        seasonal=True,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
    )

    # Extract the best parameters
    best_params = arima_model.order
    print(f"Selected (p, d, q): {best_params}")

    # Fit the ARIMA model with the best parameters
    model = ARIMA(train_data, order=best_params)
    fitted_model = model.fit()

    # Forecast the next steps
    forecast = fitted_model.forecast(steps=len(test_data))

    # Evaluate the results
    test_data = test_data.reset_index(drop=True)
    forecast = forecast.reset_index(drop=True)

    mae = mean_absolute_error(test_data, forecast)
    mse = mean_squared_error(test_data, forecast)
    rmse = mse**0.5
    mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100

    # Log results
    with open(results_file, "a") as f:
        f.write(
            f"{city},ARIMA,{best_params},{mae:.2f},{mse:.2f},{rmse:.2f},{mape:.2f}%\n"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Time-series forecasting for AQI data using various models."
    )

    parser.add_argument(
        "--eda", action="store_true", help="Perform exploratory data analysis."
    )
    parser.add_argument("--vmd", action="store_true", help="Enable VMD decomposition.")
    parser.add_argument(
        "--input_file",
        type=str,
        default="./dataset/air_quality_Taiwan.csv",
        help="Path to the input dataset CSV file.",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default="aqi",
        help="Target variable for analysis (default: 'aqi').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results and plots.",
    )
    parser.add_argument(
        "--cities",
        type=str,
        help="Comma-separated list of cities to process (default: all cities).",
    )
    parser.add_argument(
        "--parameter_grid",
        type=str,
        default="parameters.json",
        help="Path to JSON file with parameter grids.",
    )
    parser.add_argument(
        "--num_jobs", type=int, default=1, help="Number of parallel jobs (default: 1)."
    )
    parser.add_argument(
        "--forecast_horizon", type=int, default=1, help="Number of steps t forecast"
    )
    parser.add_argument(
        "--model", type=str, choices=["FAN", "BaseNN", "ARIMA"], help="Model to apply."
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate and save visualizations."
    )

    args = parser.parse_args()

    # Load parameter grid
    if args.parameter_grid:
        with open(args.parameter_grid, "r") as f:
            parameters = json.load(f)
    else:
        parameters = {}

    # Load dataset
    data = load_dataset(args.input_file)

    # EDA
    if args.eda:
        apply_eda(data, args.output_dir, args.visualize)

    # Data Cleaning
    data = clean_data(data, args.variable, args.output_dir, args.visualize)

    # Segment by city and save
    segment_by_city(data, args.variable, args.cities, args.output_dir)

    # Apply VMD if enabled
    if args.vmd:
        print("\nStarting VMD...")
        cities_list = (
            args.cities.split(",") if args.cities else data["sitename"].unique()
        )
        for city in cities_list:
            city_file = os.path.join(args.output_dir, f"{city}/{city}_aqi.csv")
            try:
                city_data = pd.read_csv(city_file)
                city_data["date"] = pd.to_datetime(city_data["date"])
                city_data.set_index("date", inplace=True)
                apply_vmd(
                    city,
                    city_data,
                    args.variable,
                    parameters["VMD"],
                    args.output_dir,
                    args.num_jobs,
                    args.visualize,
                )
            except FileNotFoundError:
                print(f"File not found for city: {city}")
            except Exception as e:
                print(f"Error processing city: {city}, Error: {e}")

    # Model application
    if args.model == "FAN":
        print("\nStarting FAN...")
        for city in (
            args.cities.split(",") if args.cities else data["sitename"].unique()
        ):
            apply_fan(
                city,
                args.vmd,
                parameters,
                args.output_dir,
                args.num_jobs,
                args.visualize,
                args.forecast_horizon,
            )
    elif args.model == "BaseNN":
        print("\nStarting BaseNN...")
        for city in (
            args.cities.split(",") if args.cities else data["sitename"].unique()
        ):
            apply_baseNN(
                city,
                args.vmd,
                parameters,
                args.output_dir,
                args.num_jobs,
                args.forecast_horizon,
            )
    elif args.model == "ARIMA":
        print("\nStarting ARIMA...")
        for city in (
            args.cities.split(",") if args.cities else data["sitename"].unique()
        ):
            city_file = os.path.join(args.output_dir, f"{city}/{city}_aqi.csv")
            try:
                city_data = pd.read_csv(city_file)
                city_data["date"] = pd.to_datetime(city_data["date"])
                city_data.set_index("date", inplace=True)
                apply_arima(
                    city,
                    city_data,
                    args.variable,
                    args.output_dir,
                    args.vmd,
                    parameters,
                    args.num_jobs,
                )
            except FileNotFoundError:
                print(f"File not found for city: {city}")
            except Exception as e:
                print(f"Error processing city: {city}, Error: {e}")


if __name__ == "__main__":
    main()
