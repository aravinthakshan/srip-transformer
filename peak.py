# import os
# import pandas as pd
# import re
# import numpy as np

# csv_dir = "/home/aravinthakshan/Projects/main-srip/src/output_csv"

# for filename in os.listdir(csv_dir):
#     if filename.endswith(".csv") and "streamflow" in filename:
#         match = re.search(r'predictions_(\d+)_streamflow_final\.csv', filename)
#         if match:
#             number = int(match.group(1))
#             if number % 2 == 0:
#                 filepath = os.path.join(csv_dir, filename)
#                 df = pd.read_csv(filepath)

#                 # Avoid division by zero
#                 df = df[df['ground_truth'] != 0]

#                 if df.empty:
#                     print(f"{filename}: No valid rows (ground_truth != 0) for peak capture calculation.")
#                     continue

#                 # Compute peak percentage safely
#                 df['% of peak captured'] = (df['predicted'] / df['ground_truth']) * 100
#                 df['% of peak captured'] = df['% of peak captured'].clip(lower=0, upper=100)  # Avoid negative %s

#                 # Replace inf/-inf with NaN (no chaining)
#                 df['% of peak captured'] = df['% of peak captured'].replace([np.inf, -np.inf], np.nan)

#                 # Save back
#                 df.to_csv(filepath, index=False)

#                 # Average only non-NaN
#                 if df['% of peak captured'].notna().any():
#                     avg_peak = df['% of peak captured'].mean()
#                     print(f"{filename}: Average % of peak captured = {avg_peak:.2f}%")
#                 else:
#                     print(f"{filename}: All % peak values are NaN after cleaning.")

# import os
# import pandas as pd
# import re
# import numpy as np

# csv_dir = "/home/aravinthakshan/Projects/main-srip/src/output_csv"

# # Define bins and labels
# bins = [0, 20, 40, 60, 80, 100]
# labels = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]

# # List to collect all peak % values
# all_peak_values = []

# for filename in os.listdir(csv_dir):
#     if filename.endswith(".csv") and "streamflow" in filename:
#         match = re.search(r'predictions_(\d+)_streamflow_final\.csv', filename)
#         if match:
#             number = int(match.group(1))
#             if number % 2 == 0:
#                 filepath = os.path.join(csv_dir, filename)
#                 df = pd.read_csv(filepath)

#                 # Avoid division by zero
#                 df = df[df['ground_truth'] != 0]

#                 if df.empty:
#                     continue

#                 # Compute % peak safely
#                 df['% of peak captured'] = (df['predicted'] / df['ground_truth']) * 100
#                 df['% of peak captured'] = df['% of peak captured'].clip(lower=0, upper=100)
#                 df['% of peak captured'] = df['% of peak captured'].replace([np.inf, -np.inf], np.nan)

#                 all_peak_values.extend(df['% of peak captured'].dropna().tolist())

# # Convert to DataFrame for binning
# all_peak_series = pd.Series(all_peak_values)

# # Bin the values
# binned = pd.cut(all_peak_series, bins=bins, labels=labels, right=True)

# # Print results
# print(f"\nTotal average % of peak captured = {all_peak_series.mean():.2f}%")
# print("Total bin distribution:")
# for label, count in binned.value_counts().sort_index().items():
#     print(f"  {label}: {count}")

# import os
# import pandas as pd
# import re
# import numpy as np

# csv_dir = "/home/aravinthakshan/Projects/main-srip/src/output_csv"

# # Define bins and labels
# bins = [0, 20, 40, 60, 80, 100]
# labels = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]

# # List to collect peak % values from 7-day windows
# rolling_peak_values = []

# for filename in os.listdir(csv_dir):
#     if filename.endswith(".csv") and "streamflow" in filename:
#         match = re.search(r'predictions_(\d+)_streamflow_final\.csv', filename)
#         if match:
#             number = int(match.group(1))
#             if number % 2 == 0:
#                 filepath = os.path.join(csv_dir, filename)
#                 df = pd.read_csv(filepath)
                
#                 # Avoid division by zero
#                 df = df[df['ground_truth'] != 0]
#                 if df.empty:
#                     continue
                
#                 # Ensure date column exists and is datetime
#                 if 'date' in df.columns:
#                     df['date'] = pd.to_datetime(df['date'])
#                     df = df.sort_values('date')
#                 elif 'Date' in df.columns:
#                     df['Date'] = pd.to_datetime(df['Date'])
#                     df = df.sort_values('Date').rename(columns={'Date': 'date'})
#                 else:
#                     # If no date column, create index-based rolling windows
#                     df = df.reset_index(drop=True)
#                     window_size = 7
                    
#                     for i in range(len(df) - window_size + 1):
#                         window_data = df.iloc[i:i+window_size]
#                         # Find the row with highest ground_truth in this window
#                         peak_idx = window_data['ground_truth'].idxmax()
#                         peak_row = df.loc[peak_idx]
                        
#                         # Calculate % of peak captured for this peak
#                         peak_captured = (peak_row['predicted'] / peak_row['ground_truth']) * 100
#                         peak_captured = np.clip(peak_captured, 0, 100)
                        
#                         if not np.isnan(peak_captured) and not np.isinf(peak_captured):
#                             rolling_peak_values.append(peak_captured)
#                     continue
                
#                 # Date-based rolling windows (7 days)
#                 start_date = df['date'].min()
#                 end_date = df['date'].max()
#                 current_date = start_date
                
#                 while current_date <= end_date - pd.Timedelta(days=6):
#                     window_end = current_date + pd.Timedelta(days=6)
#                     window_data = df[(df['date'] >= current_date) & (df['date'] <= window_end)]
                    
#                     if not window_data.empty:
#                         # Find the row with highest ground_truth in this 7-day window
#                         peak_idx = window_data['ground_truth'].idxmax()
#                         peak_row = window_data.loc[peak_idx]
                        
#                         # Calculate % of peak captured for this peak
#                         peak_captured = (peak_row['predicted'] / peak_row['ground_truth']) * 100
#                         peak_captured = np.clip(peak_captured, 0, 100)
                        
#                         if not np.isnan(peak_captured) and not np.isinf(peak_captured):
#                             rolling_peak_values.append(peak_captured)
                    
#                     current_date += pd.Timedelta(days=1)  # Move window by 1 day

# # Convert to DataFrame for binning
# rolling_peak_series = pd.Series(rolling_peak_values)

# # Bin the values
# binned = pd.cut(rolling_peak_series, bins=bins, labels=labels, right=True)

# # Print results
# print(f"\nTotal 7-day rolling peak analysis:")
# print(f"Number of peaks analyzed: {len(rolling_peak_values)}")
# print(f"Average % of peak captured = {rolling_peak_series.mean():.2f}%")
# print("Peak capture distribution:")
# for label, count in binned.value_counts().sort_index().items():
#     percentage = (count / len(rolling_peak_values)) * 100
#     print(f" {label}: {count} ({percentage:.1f}%)")

import os
import pandas as pd
import re
import numpy as np

csv_dir = "/home/aravinthakshan/Projects/main-srip/src/output_csv"

# Define bins and labels
bins = [0, 20, 40, 60, 80, 100]
labels = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]

# List to collect peak % values from 7-day windows
rolling_peak_values = []

for filename in os.listdir(csv_dir):
    if filename.endswith(".csv") and "streamflow" in filename:
        match = re.search(r'predictions_(\d+)_streamflow_final\.csv', filename)
        if match:
            number = int(match.group(1))
            if number % 2 == 0:
                filepath = os.path.join(csv_dir, filename)
                df = pd.read_csv(filepath)
                
                # Avoid division by zero
                df = df[df['ground_truth'] != 0]
                if df.empty:
                    continue
                
                # Ensure date column exists and is datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                elif 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.sort_values('Date').rename(columns={'Date': 'date'})
                else:
                    # If no date column, create index-based rolling windows
                    df = df.reset_index(drop=True)
                    window_size = 7
                    
                    for i in range(len(df) - window_size + 1):
                        window_data = df.iloc[i:i+window_size]
                        # Find the row with highest ground_truth in this window
                        peak_idx = window_data['ground_truth'].idxmax()
                        peak_row = df.loc[peak_idx]
                        
                        # Calculate % of peak captured for this peak
                        peak_captured = (peak_row['predicted'] / peak_row['ground_truth']) * 100
                        peak_captured = np.clip(peak_captured, 0, 100)
                        
                        if not np.isnan(peak_captured) and not np.isinf(peak_captured):
                            rolling_peak_values.append(peak_captured)
                    continue
                
                # Date-based rolling windows (7 days)
                start_date = df['date'].min()
                end_date = df['date'].max()
                current_date = start_date
                
                while current_date <= end_date - pd.Timedelta(days=6):
                    window_end = current_date + pd.Timedelta(days=6)
                    window_data = df[(df['date'] >= current_date) & (df['date'] <= window_end)]
                    
                    if not window_data.empty:
                        # Find the row with highest ground_truth in this 7-day window
                        peak_idx = window_data['ground_truth'].idxmax()
                        peak_row = window_data.loc[peak_idx]
                        
                        # Calculate % of peak captured for this peak
                        peak_captured = (peak_row['predicted'] / peak_row['ground_truth']) * 100
                        peak_captured = np.clip(peak_captured, 0, 100)
                        
                        if not np.isnan(peak_captured) and not np.isinf(peak_captured):
                            rolling_peak_values.append(peak_captured)
                    
                    current_date += pd.Timedelta(days=1)  # Move window by 1 day

# Convert to DataFrame for binning
rolling_peak_series = pd.Series(rolling_peak_values)

# Bin the values
binned = pd.cut(rolling_peak_series, bins=bins, labels=labels, right=True)

# Print results
print(f"\nTotal 7-day rolling peak analysis:")
print(f"Number of peaks analyzed: {len(rolling_peak_values)}")
print(f"Average % of peak captured = {rolling_peak_series.mean():.2f}%")
print("Peak capture distribution:")
for label, count in binned.value_counts().sort_index().items():
    percentage = (count / len(rolling_peak_values)) * 100
    print(f" {label}: {count} ({percentage:.1f}%)")

# Plotting
import matplotlib.pyplot as plt

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Bar chart of bin distribution
bin_counts = binned.value_counts().sort_index()
colors = ['#ff4444', '#ff8800', '#ffcc00', '#88cc00', '#00cc44']
bars = ax1.bar(range(len(bin_counts)), bin_counts.values, color=colors)
ax1.set_xlabel('Peak Capture Range')
ax1.set_ylabel('Number of Peaks')
ax1.set_title('Distribution of 7-Day Rolling Peak Capture')
ax1.set_xticks(range(len(bin_counts)))
ax1.set_xticklabels(bin_counts.index, rotation=45)

# Add value labels on bars
for bar, count in zip(bars, bin_counts.values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5, str(count), 
             ha='center', va='bottom')

# Plot 2: Histogram of all peak values
ax2.hist(rolling_peak_series, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
ax2.axvline(rolling_peak_series.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {rolling_peak_series.mean():.1f}%')
ax2.set_xlabel('% Peak Captured')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Peak Capture Percentages')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Summary statistics
print(f"\nSummary Statistics:")
print(f"Mean: {rolling_peak_series.mean():.2f}%")
print(f"Median: {rolling_peak_series.median():.2f}%")
print(f"Std Dev: {rolling_peak_series.std():.2f}%")
print(f"Min: {rolling_peak_series.min():.2f}%")
print(f"Max: {rolling_peak_series.max():.2f}%")