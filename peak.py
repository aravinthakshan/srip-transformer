import os
import pandas as pd

# Path to your directory with the CSV files
csv_dir = "/home/aravinthakshan/Projects/main-srip/src/output_csv"  # change this to your actual path

# Iterate over each CSV file
for filename in os.listdir(csv_dir):
    if filename.endswith(".csv"):
        filepath = os.path.join(csv_dir, filename)
        df = pd.read_csv(filepath)

        # Compute % of peak captured
        df['% of peak captured'] = (df['predicted'] / df['ground_truth']) * 100
        df['% of peak captured'] = df['% of peak captured'].clip(upper=100)

        # Save the modified CSV
        df.to_csv(filepath, index=False)

        # Print average peak capture percentage
        avg_peak = df['% of peak captured'].mean()
        print(f"{filename}: Average % of peak captured = {avg_peak:.2f}%")
