import argparse
import os
import pandas as pd
import torch
import numpy as np
import json
import warnings
import random
from utils import *

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


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
        "--model", type=str, choices=["FAN", "BaseNN"], help="Model to apply."
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
    data = clean_data(data, args.variable)

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


if __name__ == "__main__":
    main()
