import os
import json
import csv

# Define the root folder
root_folder = "experiments_scaling"

# Output CSV file
output_csv = "scaling_experiment_summary.csv"

# Training time in itwinai-friendly format
mlpf_csv = "mlpf_train_time.csv"

# Prepare the CSV header
header = ["num_nodes", "epoch_id", "train_time", "valid_time", "tot_time"]

# Open the CSV file for writing
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)

    # Iterate over subfolders matching the pattern
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)

        # Skip non-directory entries
        if not os.path.isdir(subfolder_path):
            continue

        # Extract the number of nodes from the folder name
        if "scaling_bl_ray_N_" in subfolder:
            try:
                num_nodes = int(subfolder.split("scaling_bl_ray_N_")[1].split("_")[0])
            except (ValueError, IndexError):
                print(f"Skipping folder due to parsing error: {subfolder}")
                continue
        else:
            continue

        # Navigate to the 'history' folder
        history_folder = os.path.join(subfolder_path, "history")
        if not os.path.exists(history_folder):
            continue

        # Process each JSON file in the history folder
        for json_file in os.listdir(history_folder):
            if json_file.startswith("epoch_") and json_file.endswith(".json"):
                epoch_id = json_file.split("epoch_")[1].split(".json")[0]

                # Attempt to parse the epoch ID
                try:
                    epoch_id = int(epoch_id)
                except ValueError:
                    print(f"Skipping file due to invalid epoch number: {json_file}")
                    continue

                json_path = os.path.join(history_folder, json_file)

                # Read and parse the JSON file
                with open(json_path, "r") as f:
                    data = json.load(f)
                    train_time = data.get("epoch_train_time", 0)
                    valid_time = data.get("epoch_valid_time", 0)
                    tot_time = data.get("epoch_total_time", 0)

                # Write the row to the CSV file
                writer.writerow([num_nodes, epoch_id, train_time, valid_time, tot_time])


# Generate the mlpf_baseline summary CSV file
with open(output_csv, "r") as csvfile, open(mlpf_csv, "w", newline="") as mlpf_file:
    reader = csv.DictReader(csvfile)
    mlpf_writer = csv.writer(mlpf_file)
    mlpf_writer.writerow(["name", "nodes", "epoch_id", "time"])

    for row in reader:
        name = "mlpf_baseline"
        nodes = row["num_nodes"]
        epoch_id = row["epoch_id"]
        time = row["train_time"]
        mlpf_writer.writerow([name, nodes, epoch_id, time])

print(f"CSV summary saved to {output_csv}")
print(f"mlpf_baseline summary saved to {mlpf_csv}")

import pandas as pd

from itwinai.scalability import (
        create_absolute_plot,
        create_relative_plot,
    )

combined_df = pd.read_csv(mlpf_csv)

print("Merged CSV:")
print(combined_df)

avg_time_df = (
    combined_df.drop(columns="epoch_id")
    .groupby(["name", "nodes"])
    .mean()
    .reset_index()
)
print("\nAvg over name and nodes:")
print(avg_time_df.rename(columns=dict(time="avg(time)")))

create_absolute_plot(avg_time_df)
create_relative_plot(avg_time_df)