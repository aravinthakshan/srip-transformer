import pandas as pd
import numpy as np

def nse(obs, sim):
    obs = np.array(obs)
    sim = np.array(sim)
    return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)

# Change this path to your CSV file
csv_path = "/home/aravinthakshan/Projects/main-srip/src/output_csv/predictions_combined_Handia_streamflow_final.csv"

df = pd.read_csv(csv_path)

# If you want to filter by dataset (e.g., 'train', 'First 10 Years', etc.), you can do so:
# df = df[df['dataset'] == 'train']

for horizon in sorted(df['horizon'].unique()):
    sub = df[df['horizon'] == horizon]
    nse_val = nse(sub['ground_truth'], sub['predicted'])
    print(f"Horizon {horizon}: NSE = {nse_val:.4f}")