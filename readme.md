python run.py --input_file dataset/air_quality_Taiwan.csv --output_dir processed_data --variable aqi --eda --remove_outliers --cities "Annan" --segment --visualize --vmd


python run.py --input_file dataset/air_quality_Taiwan.csv --output_dir processed_data --variable aqi --cities "Annan,Sanyi" --vmd


python run.py --cities "Annan" --parameter_grid ptest.json --visualize --vmd --num_jobs 2 --
model FAN --forecast_horizon 1