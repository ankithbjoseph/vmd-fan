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
import torch.nn as nn
import numpy as np
from itertools import product
from sklearn.metrics import mean_absolute_error, mean_squared_error
from models.FAN import FANForecastingModel
from models.BaseNN import BaselineNN
from joblib import Parallel, delayed
import json
import warnings
import random
import requests
import zipfile


######################################
# MAIN UTILITY FUNCTIONS
######################################


def load_dataset(file_path):
    """
    Load the dataset from a file. If the file doesn't exist, download and extract it.
    """
    if not os.path.exists(file_path):
        print(f"Dataset not found at {file_path}. Downloading and extracting...")
        os.makedirs("dataset", exist_ok=True)
        download_url = "https://www.kaggle.com/api/v1/datasets/download/taweilo/taiwan-air-quality-data-20162024"
        zip_file_path = "dataset/taiwan-air-quality-data-20162024.zip"

        with requests.get(download_url, stream=True) as response:
            response.raise_for_status()
            with open(zip_file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"Dataset downloaded to {zip_file_path}")

        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall("dataset")

        extracted_file = "dataset/air_quality.csv"
        if not os.path.exists(extracted_file):
            raise FileNotFoundError(
                f"Expected file {extracted_file} not found in extracted archive."
            )
        os.rename(extracted_file, file_path)

    print(f"Loading dataset from {file_path}...")
    data = pd.read_csv(file_path)
    print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
    return data


def apply_eda(data):
    """
    Perform exploratory data analysis on the dataset
    """
    print("\nDataset Info:")
    print(data.info())

    print("\nMissing Data:")
    print(data.isnull().sum())

    print("\nUnique Values in 'sitename':", data["sitename"].nunique())


def clean_data(data, variable):
    """
    Clean the dataset by removing missing values and negative values for the selected variable.
    """
    print("\nCleaning data...")
    data = data[["date", "sitename", variable]].copy()
    data[variable] = pd.to_numeric(data[variable], errors="coerce")
    data = data.dropna(subset=[variable])
    data = data[data[variable] > 0]
    print(f"Data cleaned. Remaining rows: {data.shape[0]}")

    return data


def validate_data_format(data):
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


def segment_by_city(data, variable, cities, output_dir, visualize):
    """
    Segment data by 'sitename' and save each city's data to separate CSV files.

    """
    print("\nSegmenting data by city...")

    cities_list = cities.split(",") if cities else data["sitename"].unique()
    final_city_list = []
    data = data[data["sitename"].isin(cities_list)]
    grouped = data.groupby("sitename")
    for city, group in grouped:
        print(f"\nCity: {city}")

        city_path = os.path.join(output_dir, f"{city}/{city}_{variable}.csv")
        save_path = os.path.join(output_dir, f"{city}/{city}_{variable}_ts.png")
        group = validate_data_format(group)

        if group.shape[0] > 60000:
            os.makedirs(f"{output_dir}/{city}", exist_ok=True)
            final_city_list.append(city)
            group[variable].to_csv(city_path, index=True)
            print(f"Remaining rows: {group.shape[0]}")

            if visualize:
                if not os.path.exists(save_path):
                    plt.figure(figsize=(10, 6))
                    plt.plot(
                        group.index, group[variable], label="timeseries", color="blue"
                    )
                    plt.title(f"Time Series of {variable.upper()} (city: {city})")
                    plt.xlabel("Date")
                    plt.ylabel(variable)
                    plt.grid(True)
                    # plt.legend()
                    plt.savefig(save_path, dpi=600)
                    plt.close()
                    print(f"Plot saved to {save_path}")

            print(f"Saved data for {city} to {city_path}")
        else:
            print(f"Skipped City:{city} due to data points less than 60000")

    return final_city_list


def create_sequences(data, window_size, forecast_horizon):
    """
    Generate input-output sequence pairs for time series forecasting.

    """
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


def compute_diagonal_averages(matrix):
    """
    Calculate the average of elements along each diagonal of a matrix.

    """
    rows, cols = matrix.shape
    max_diagonals = rows + cols - 1
    result = []

    for diag in range(max_diagonals):
        elements = []
        for row in range(rows):
            col = diag - row
            if 0 <= col < cols:
                elements.append(matrix[row, col])
        if elements:
            try:
                avg = sum(elements) / len(elements)
            except TypeError:
                avg = elements
            result.append(avg)

    return np.array(result)


def forecast_data(data, model, window_size, forecast_horizon, device):
    """
    Generate forecasts using a trained model.

    """
    model.eval()
    X_test, _ = create_sequences(data.values, window_size, forecast_horizon)
    X_test = X_test.to(device)
    with torch.no_grad():
        predictions = model(X_test).cpu().numpy()

    predictions = compute_diagonal_averages(predictions)

    return predictions


def forecast_imfs_data(imfs, fan_models, window_size, forecast_horizon, device):
    """
    Generate forecasts using a trained imf models.

    """
    predictions = {}
    for imf, model in fan_models.items():
        pred = forecast_data(imfs[imf], model, window_size, forecast_horizon, device)
        predictions[imf] = pred

    return pd.DataFrame(predictions)


def plot_actual_vs_predicted(
    actual, predicted, output_dir, city_name, variable_name, model_name, parameters
):
    """
    Plots actual vs. predicted values and saves the plot.

    """
    savedir = f"{output_dir}/{city_name}/plots/{model_name}"
    os.makedirs(savedir, exist_ok=True)

    values = [param.split(":")[1] for param in parameters.split(",")]
    p = "_".join(values)
    plot_title = f"{model_name} Actual vs Predicted {variable_name} (City: {city_name})"
    plot_filename = f"{savedir}/{city_name}_{variable_name}_{p}_result.png"

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Truth", color="red", alpha=1)
    plt.plot(predicted, label="Prediction", color="blue", alpha=1)
    plt.title(plot_title)
    plt.xlabel("Datetime/hour")
    plt.ylabel(variable_name)
    plt.legend()
    plt.grid(True)

    plt.savefig(plot_filename, dpi=600)
    plt.close()
    print(f"Plot saved to {plot_filename}")


def load_decomposition_file(base_dir, K, alpha, tau, DC, tol, city):
    """
    function to load a decomposition file based on given parameters.

    """

    if tau == 0:
        tau = "0e+00"

    K_folder = os.path.join(base_dir, f"K_{K}")
    filename = f"{city}_K_{K}_alpha_{alpha}_tau_{str(tau)}_DC_{DC}_tol_{str(tol)}.csv"
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


######################################
# VMD
######################################


def apply_vmd(city, data, variable, parameter_grid, output_dir, num_jobs, visualize):
    """
    Apply VMD transformation to the city's data for the selected variable.
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
        file_name = f"{city}_K_{K}_alpha_{alpha}_tau_{tau:.0e}_DC_{DC}_tol_{tol:.0e}"
        full_path = os.path.join(sub_dir, file_name + ".csv")
        plot_path = os.path.join(sub_dir, file_name + ".png")

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
                if not os.path.exists(plot_path):
                    plt.figure(figsize=(12, K * 1.5))
                    total_subplots = K + 1
                    plt.subplot(total_subplots, 1, 1)
                    plt.plot(data.index, data["aqi"], color="red")
                    plt.text(
                        -0.035,
                        0.5,
                        "Original",
                        transform=plt.gca().transAxes,
                        fontsize=10,
                        ha="right",
                        rotation="vertical",
                        va="center",
                    )
                    plt.grid(True)
                    for i, column in enumerate(decomposed_data.columns, start=2):
                        plt.subplot(total_subplots, 1, i)
                        plt.plot(
                            decomposed_data.index, decomposed_data[column], color="blue"
                        )
                        plt.text(
                            -0.035,
                            0.5,
                            column,
                            transform=plt.gca().transAxes,
                            fontsize=10,
                            ha="right",
                            rotation="vertical",
                            va="center",
                        )
                        plt.grid(True)
                    plt.gcf().text(0.5, 0.01, "Datetime/hour", fontsize=12, ha="center")
                    plt.suptitle(
                        f"VMD Decomposition for City:'{city}' ({variable.upper()}) (Parameters: K={K}, alpha={alpha}, tau={tau}, tol={tol})",
                        fontsize=14,
                        y=0.95,
                    )
                    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
                    plt.savefig(plot_path, dpi=600)
                    plt.close()

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


######################################
# BASENN
######################################


def apply_baseNN(
    city,
    vmd,
    parameter_grid,
    output_dir,
    num_jobs,
    forecast_horizon,
    variable,
    visualize,
):
    """
    Apply BaseNN to each IMF of a city's data using the specified parameters.
    """
    hidden_dim = 64
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
            variable,
            visualize,
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
            except Exception as e:
                raise Exception(f"An error occurred: {e}")

            train_data, test_data = temporal_train_test_split(city_data, test_size=0.2)

            print(test_data)

            base_dir = f"./{output_dir}/{city}/vmd_decompositions"

            try:
                imfs_data = load_decomposition_file(
                    base_dir, K, alpha, tau, DC, tol, city
                )
                imfs_train_data, imfs_test_data = temporal_train_test_split(
                    imfs_data, test_size=0.2
                )
            except Exception as e:
                raise Exception(f"An error occurred: {e}")

            base_models = {}

            for imf in imfs_train_data.columns:
                X_train, y_train = create_sequences(
                    imfs_train_data[imf].values, window_size, forecast_horizon
                )

                train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=False,
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

            predicted_aqi = final_forecast.values

            actual_aqi = np.array(
                test_data["aqi"].values[window_size : window_size + len(predicted_aqi)]
            )

            # print(predicted_aqi)
            # print(len(predicted_aqi))

            mae = mean_absolute_error(actual_aqi, predicted_aqi)
            mse = mean_squared_error(actual_aqi, predicted_aqi)
            rmse = mse**0.5
            mape = np.mean(np.abs((actual_aqi - predicted_aqi) / actual_aqi)) * 100

            with open(results_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([city, Model, parameters, mae, mse, rmse, mape])

            if visualize:
                plot_actual_vs_predicted(
                    actual_aqi,
                    predicted_aqi,
                    output_dir,
                    city,
                    variable,
                    Model,
                    parameters,
                )

        param_combinations = [
            (
                window_size,
                hidden_dim,
                learning_rate,
                K,
                alpha,
                tau,
                DC,
                tol,
            )
            for K in parameter_grid["VMD"]["K"]
            for alpha in parameter_grid["VMD"]["alpha"]
            for tau in parameter_grid["VMD"]["tau"]
            for DC in parameter_grid["VMD"]["DC"]
            for tol in parameter_grid["VMD"]["tol"]
            for learning_rate in parameter_grid["BaseNN"]["learning_rate"]
            for window_size in parameter_grid["BaseNN"]["window_size"]
        ]

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
                variable,
                visualize,
            )
            for window_size, hidden_dim, learning_rate, K, alpha, tau, DC, tol in param_combinations
        )

    else:
        # BaseNN without VMD
        def process_basenn(window_size, hidden_dim, learning_rate, city):
            Model = "BaseNN"
            parameters = f"window_size:{window_size}, hidden_dim:{hidden_dim}, learning_rate:{learning_rate}, forecast_horizon:{forecast_horizon}"

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
            except Exception as e:
                raise Exception(f"An error occurred: {e}")

            train_data, test_data = temporal_train_test_split(city_data, test_size=0.2)
            X_train, y_train = create_sequences(
                train_data["aqi"].values, window_size, forecast_horizon
            )
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
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

            predicted_aqi = forecast_data(
                test_data["aqi"], model, window_size, forecast_horizon, device
            )

            actual_aqi = np.array(
                test_data["aqi"].values[window_size : window_size + len(predicted_aqi)]
            )

            # print(len(predicted_aqi), len(actual_aqi))
            # print(predicted_aqi)
            # print(actual_aqi)

            mae = mean_absolute_error(actual_aqi, predicted_aqi)
            mse = mean_squared_error(actual_aqi, predicted_aqi)
            rmse = mse**0.5
            mape = np.mean(np.abs((actual_aqi - predicted_aqi) / actual_aqi)) * 100

            with open(results_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([city, Model, parameters, mae, mse, rmse, mape])

            if visualize:
                plot_actual_vs_predicted(
                    actual_aqi,
                    predicted_aqi,
                    output_dir,
                    city,
                    variable,
                    Model,
                    parameters,
                )

        param_combinations = [
            (window_size, hidden_dim, learning_rate)
            for learning_rate in parameter_grid["BaseNN"]["learning_rate"]
            for window_size in parameter_grid["BaseNN"]["window_size"]
        ]

        Parallel(n_jobs=num_jobs)(
            delayed(process_basenn)(
                window_size,
                hidden_dim,
                learning_rate,
                city,
            )
            for window_size, hidden_dim, learning_rate in param_combinations
        )


######################################
# FAN
######################################


def apply_fan(
    city,
    vmd,
    parameter_grid,
    output_dir,
    num_jobs,
    visualize,
    forecast_horizon,
    variable,
):
    """
    Apply FAN to each IMF of a city's data using the specified parameters.
    """

    num_epochs = 50
    batch_size = 256
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
            except Exception as e:
                raise Exception(f"An error occurred : {e}")

            train_data, test_data = temporal_train_test_split(city_data, test_size=0.2)

            base_dir = f"./{output_dir}/{city}/vmd_decompositions"

            try:
                imfs_data = load_decomposition_file(
                    base_dir, K, alpha, tau, DC, tol, city
                )
                imfs_train_data, imfs_test_data = temporal_train_test_split(
                    imfs_data, test_size=0.2
                )
            except Exception as e:
                raise Exception(f"An error occurred : {e}")

            fan_models = {}

            for imf in imfs_train_data.columns:
                X_train, y_train = create_sequences(
                    imfs_train_data[imf].values, window_size, forecast_horizon
                )

                train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=False,
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
            predicted_aqi = final_forecast.values

            actual_aqi = np.array(
                test_data["aqi"].values[window_size : window_size + len(predicted_aqi)]
            )

            mae = mean_absolute_error(actual_aqi, predicted_aqi)
            mse = mean_squared_error(actual_aqi, predicted_aqi)
            rmse = mse**0.5
            mape = np.mean(np.abs((actual_aqi - predicted_aqi) / actual_aqi)) * 100

            with open(results_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([city, Model, parameters, mae, mse, rmse, mape])

            if visualize:
                plot_actual_vs_predicted(
                    actual_aqi,
                    predicted_aqi,
                    output_dir,
                    city,
                    variable,
                    Model,
                    parameters,
                )

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

            X_train, y_train = create_sequences(
                train_data["aqi"].values, window_size, forecast_horizon
            )
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
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

            predicted_aqi = forecast_data(
                test_data["aqi"], model, window_size, forecast_horizon, device
            )

            actual_aqi = np.array(
                test_data["aqi"].values[window_size : window_size + len(predicted_aqi)]
            )

            mae = mean_absolute_error(actual_aqi, predicted_aqi)
            mse = mean_squared_error(actual_aqi, predicted_aqi)
            rmse = mse**0.5
            mape = np.mean(np.abs((actual_aqi - predicted_aqi) / actual_aqi)) * 100

            with open(results_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([city, Model, parameters, mae, mse, rmse, mape])

            if visualize:
                plot_actual_vs_predicted(
                    actual_aqi,
                    predicted_aqi,
                    output_dir,
                    city,
                    variable,
                    Model,
                    parameters,
                )

            return params, mae, mse, rmse, mape

        Model = "FAN"
        fan_parameters = parameter_grid["FAN"]
        param_combinations = list(product(*fan_parameters.values()))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        file_name = f"{city}/{city}_aqi.csv"
        file_path = os.path.join(output_dir, file_name)

        try:
            city_data = pd.read_csv(file_path)
        except Exception as e:
            raise Exception(f"An error occurred : {e}")

        train_data, test_data = temporal_train_test_split(city_data, test_size=0.2)

        Parallel(n_jobs=num_jobs)(
            delayed(process_fan)(
                params, train_data, test_data, device, results_file, city, Model
            )
            for params in param_combinations
        )
