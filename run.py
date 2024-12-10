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

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def load_dataset(file_path):
    """
    Load the dataset from a file. If the file doesn't exist, download and extract it.
    """
    if not os.path.exists(file_path):
        print(f"Dataset not found at {file_path}. Downloading and extracting...")
        download_url = "https://www.kaggle.com/api/v1/datasets/download/taweilo/taiwan-air-quality-data-20162024"
        zip_file_path = os.path.expanduser(
            "~/Downloads/taiwan-air-quality-data-20162024.zip"
        )
        download_dir = os.path.dirname(file_path)

        # Create the download directory if it doesn't exist
        os.makedirs(download_dir, exist_ok=True)

        # Download the file using requests
        with requests.get(download_url, stream=True) as response:
            response.raise_for_status()
            with open(zip_file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"Dataset downloaded to {zip_file_path}")

        # Extract the ZIP file
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(download_dir)

        print(f"Dataset extracted to {download_dir}")

        # Assuming the dataset has a specific file within the extracted folder
        extracted_file = os.path.join(
            download_dir, "air_quality.csv"
        )  # Change to the actual file name
        if not os.path.exists(extracted_file):
            raise FileNotFoundError(
                f"Expected file {extracted_file} not found in extracted archive."
            )

        os.rename(
            extracted_file, file_path
        )  # Move/rename extracted file to the expected location

    # Load the dataset
    print(f"Loading dataset from {file_path}...")
    data = pd.read_csv(file_path)
    print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
    return data


def apply_eda(data):
    """
    Perform exploratory data analysis on the dataset and generate visualizations if enabled.
    """
    print("\nDataset Info:")
    print(data.info())

    print("\nMissing Data:")
    print(data.isnull().sum())

    print("\nUnique Values in 'sitename':", data["sitename"].nunique())


def clean_data(data, variable, output_dir, visualize):
    """
    Clean the dataset by removing missing values and negative values for the selected variable.
    Generate visualizations for the cleaned data.
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
        os.makedirs(f"{output_dir}/{city}", exist_ok=True)
        city_path = os.path.join(output_dir, f"{city}/{city}_{variable}.csv")
        save_path = os.path.join(output_dir, f"{city}/{city}_{variable}_ts.png")
        group = validate_data_format(group)

        if group.shape[0] > 60000:
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
        pass
    return predictions


def forecast_imfs_data(imfs, fan_models, window_size, forecast_horizon, device):
    predictions = {}
    for imf, model in fan_models.items():
        model.eval()
        imf_data = imfs[imf].values
        X_test, _ = create_sequences(imf_data, window_size, forecast_horizon)
        X_test = X_test.to(device)
        with torch.no_grad():
            pred = model(X_test).cpu().numpy()

        pred_flattened = pred.sum(axis=1)
        predictions[imf] = pred_flattened

    return pd.DataFrame(
        predictions, index=imfs.index[window_size + forecast_horizon - 1 :]
    )


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
    plt.plot(
        range(len(predicted)), predicted, label="Prediction", color="blue", alpha=1
    )
    plt.plot(range(len(actual)), actual, label="Truth", color="red", alpha=1)
    plt.title(plot_title)
    plt.xlabel("Datetime/hour")
    plt.ylabel(variable_name)
    plt.legend()
    plt.grid(True)

    # Save and show the plot
    plt.savefig(plot_filename, dpi=600)
    plt.close()  # Close the plot to free memory
    print(f"Plot saved to {plot_filename}")


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
                    plt.figure(
                        figsize=(12, K * 1.5)
                    )  # Adjust height dynamically for readability
                    total_subplots = K + 1  # K IMFs + Original signal

                    # Plot original data on the first subplot
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

                    # Plot each IMF in its own subplot
                    for i, column in enumerate(
                        decomposed_data.columns, start=2
                    ):  # Start from 2nd subplot
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

                    # Add a single x-axis label for the entire figure
                    plt.gcf().text(0.5, 0.01, "Datetime/hour", fontsize=12, ha="center")

                    # Add a central title for the entire figure
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
            except FileNotFoundError:
                return

            train_data, test_data = temporal_train_test_split(city_data, test_size=0.2)

            base_dir = f"./{output_dir}/{city}/vmd_decompositions"

            try:
                imfs_data = load_decomposition_file(
                    base_dir, K, alpha, tau, DC, tol, city
                )
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
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
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
    hidden_dim = 32
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
            except FileNotFoundError:
                return

            train_data, test_data = temporal_train_test_split(city_data, test_size=0.2)

            base_dir = f"./{output_dir}/{city}/vmd_decompositions"

            try:
                imfs_data = load_decomposition_file(
                    base_dir, K, alpha, tau, DC, tol, city
                )
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
                variable,
                visualize,
            )
            for window_size, hidden_dim, learning_rate, K, alpha, tau, DC, tol in param_combinations
        )

    else:
        # Original BaseNN without VMD
        def process_basenn(window_size, hidden_dim, learning_rate, city):
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

            X_test, y_test = create_sequences(
                test_data["aqi"].values, window_size, forecast_horizon
            )
            X_test, y_test = X_test.to(device), y_test.to(device)

            model.eval()
            with torch.no_grad():
                predicted_aqi = model(X_test).cpu().numpy()
                actual_aqi = y_test.cpu().numpy()

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

        # Use joblib to parallelize the processing
        Parallel(n_jobs=num_jobs)(
            delayed(process_basenn)(
                window_size,
                hidden_dim,
                learning_rate,
                city,
            )
            for window_size, hidden_dim, learning_rate in param_combinations
        )


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
            f"{city},ARIMA,'{best_params}',{mae:.2f},{mse:.2f},{rmse:.2f},{mape:.2f}\n"
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
        "--model", type=str, choices=["FAN", "BaseNN", "ARIMA"], help="Model to apply."
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate and save visualizations."
    )

    parser.add_argument(
        "--forecast_horizon", type=int, default=1, help="Number of steps to forecast"
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
        apply_eda(data)

    # Data Cleaning
    data = clean_data(data, args.variable, args.output_dir, args.visualize)

    # Segment by city and save
    cities_list = segment_by_city(
        data, args.variable, args.cities, args.output_dir, args.visualize
    )

    # Apply VMD if enabled
    if args.vmd:
        print("\nStarting VMD...")
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
        for city in cities_list:
            apply_fan(
                city,
                args.vmd,
                parameters,
                args.output_dir,
                args.num_jobs,
                args.visualize,
                args.forecast_horizon,
                args.variable,
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
                args.variable,
                args.visualize,
            )
    elif args.model == "ARIMA":
        print("\nStarting ARIMA...")
        for city in cities_list:
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
