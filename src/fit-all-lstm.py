import subprocess
import os
import pandas as pd
import re
from datetime import datetime

def parse_run_info(run_info_path):
    parsed_data = {}
    current_section = None
    current_test_period = None
    current_horizon = None
    prev_line = ""
    prev_line_section = ""
    
    test_metrics_data = {
        'first_10': {'T+1': {}, 'T+2': {}, 'T+3': {}},
        'last_10': {'T+1': {}, 'T+2': {}, 'T+3': {}},
        'first_10_top10_nse': [0, 0, 0],
        'last_10_top10_nse': [0, 0, 0]
    }
    
    try:
        with open(run_info_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            
            if "Run Description:" in line:
                parsed_data['Run_Description'] = line.split(":", 1)[1].strip()
                parts = parsed_data['Run_Description'].split('_')
                parsed_data['Station_Name'] = parts[0]
                parsed_data['Target'] = parts[1]
                parsed_data['Feature_Set_Name'] = "_".join(parts[2:])
                
            elif "Station File:" in line:
                parsed_data['Station_File'] = line.split(":", 1)[1].strip()
                
            elif "Target Variable:" in line:
                parsed_data['Target_Variable'] = line.split(":", 1)[1].strip()
                
            elif "Features:" in line:
                parsed_data['Features'] = line.split(":", 1)[1].strip()
                
            elif "TRAINING METRICS" in line:
                current_section = "TRAINING_METRICS"
                
            elif "TEST METRICS" in line:
                current_section = "TEST_METRICS"
                
            elif "TOP 10% NSE SCORES" in line:
                current_section = "TOP10_NSE"
                
            elif line.startswith("LSTM T+"):
                if current_section == "TRAINING_METRICS":
                    horizon = line.split(":")[0].strip().replace("LSTM ", "")
                    parsed_data[f'Train_{horizon}_NSE'] = 0
                    parsed_data[f'Train_{horizon}_R2'] = 0
                    parsed_data[f'Train_{horizon}_PBIAS'] = 0
                    parsed_data[f'Train_{horizon}_KGE'] = 0
                    
            elif line.startswith("T+"):
                if current_section == "TEST_METRICS":
                    horizon_match = re.match(r"(T\+\d) Metrics:", line)
                    if horizon_match:
                        current_horizon = horizon_match.group(1)
                        
                elif current_section == "TOP10_NSE":
                    horizon_match = re.match(r"(T\+\d): ([-+]?\d*\.\d+|\d+)", line)
                    if horizon_match:
                        horizon = horizon_match.group(1)
                        nse_value = float(horizon_match.group(2))
                        horizon_idx = int(horizon[-1]) - 1
                        if "First 10 Years:" in prev_line_section:
                            test_metrics_data['first_10_top10_nse'][horizon_idx] = nse_value
                        elif "Last 10 Years:" in prev_line_section:
                            test_metrics_data['last_10_top10_nse'][horizon_idx] = nse_value
                            
            elif "First 10 Years" in line and current_section == "TEST_METRICS":
                current_test_period = 'first_10'
                
            elif "Last 10 Years" in line and current_section == "TEST_METRICS":
                current_test_period = 'last_10'
                
            elif "First 10 Years:" in line and current_section == "TOP10_NSE":
                prev_line_section = "First 10 Years:"
                
            elif "Last 10 Years:" in line and current_section == "TOP10_NSE":
                prev_line_section = "Last 10 Years:"
                
            elif current_section == "TEST_METRICS" and ": " in line and not line.startswith("T+") and not "Years" in line:
                metric_name_val_match = re.match(r"([A-Za-z_]+): ([-+]?\d*\.?\d+)", line.strip())
                if metric_name_val_match and current_test_period and current_horizon:
                    metric = metric_name_val_match.group(1).strip()
                    value = float(metric_name_val_match.group(2))
                    test_metrics_data[current_test_period][current_horizon][metric] = value
                    
            elif current_section == "TRAINING_METRICS" and ": " in line and not line.startswith("LSTM"):
                metric_name_val_match = re.match(r"([A-Za-z_]+): ([-+]?\d*\.?\d+)", line.strip())
                if metric_name_val_match and "LSTM " in prev_line:
                    metric = metric_name_val_match.group(1).strip()
                    value = float(metric_name_val_match.group(2))
                    train_horizon_match = re.search(r"LSTM (T\+\d):", prev_line)
                    if train_horizon_match:
                        horizon_key = train_horizon_match.group(1)
                        parsed_data[f'Train_{horizon_key}_{metric}'] = value
                        
            prev_line = line 
        for period_key, period_data in test_metrics_data.items():
            if period_key in ['first_10', 'last_10']:
                for horizon, metrics in period_data.items():
                    for metric_name, value in metrics.items():
                        col_prefix = "First10" if period_key == 'first_10' else "Last10"
                        horizon_num = horizon.replace("+", "").lower()
                        parsed_data[f'{col_prefix}_{horizon_num}_{metric_name}'] = value
                        
            elif period_key in ['first_10_top10_nse', 'last_10_top10_nse']:
                col_prefix = "Top10_First" if period_key == 'first_10_top10_nse' else "Top10_Last"
                for i, nse_val in enumerate(period_data):
                    parsed_data[f'{col_prefix}_t{i+1}_NSE'] = nse_val
                    
    except Exception as e:
        print(f"Error parsing {run_info_path}: {e}")
        print(f"Line causing error: {line if 'line' in locals() else 'Unknown'}")
        return None
    
    return parsed_data

def run_experiment(master_output_dir_for_run, station_file, features, target_variable, epochs=25):
    station_name = os.path.basename(station_file).replace("new_", "").replace(".csv", "").replace("_with_hierarchical_quantiles", "")
    
    feature_set_name = "custom_features"
    if 'modelq' in features and 'waterlevel_upstream' in features:
        feature_set_name = "Everything-w-Modelq"
    elif 'waterlevel_upstream' in features:
        feature_set_name = "Everything-w/o-Modelq"
    elif 'modelq' in features:
        feature_set_name = "basic_plus_modelq"
    elif 'VIC_W_dam' in features:
        feature_set_name = "basic_plus_VIC_W_Dam"
    
    description = f"{station_name}_{target_variable}_{feature_set_name}"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    command = [
        "python",
        "fitlstm.py",
        "--epochs", str(epochs),
        "--desc", description,
        "--station_file", station_file,
        "--target_variable", target_variable,
        "--features", ",".join(features),
        "--output_base_dir", master_output_dir_for_run
    ]
    
    print(f"--- Running experiment: {description} ---")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, cwd=script_dir)
        print(f"Successfully completed: {description}")
        print("STDOUT:\n", result.stdout)
        if result.stderr:
            print("STDERR:\n", result.stderr)
        
        run_info_path_match = re.search(r"Sequential run information saved to: (.+?run_info\.txt)", result.stdout)
        if run_info_path_match:
            return run_info_path_match.group(1).strip()
        else:
            print(f"Warning: Could not find 'run_info.txt' path in stdout for {description}.")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment {description}:")
        print(f"Command: {' '.join(e.cmd)}")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
        return None
    except FileNotFoundError:
        print(f"Error: Python interpreter or 'Fit-lstm.py' not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred for experiment {description}: {e}")
        return None

### ------------------------ This function is the main entry point for running all experiments and plotting ------------------------ ###
def main():
    from datetime import datetime
    from misc import RatingCurveFitter
    from misc import create_run_directory, save_run_info
    new_station_files = [ # station files here
        'new_Barmanghat_with_hierarchical_quantiles', 
        'new_Garudeshwar_with_hierarchical_quantiles',
        'new_Handia_with_hierarchical_quantiles',
        'new_Hoshangabad_with_hierarchical_quantiles',
        'new_Mandleshwar_with_hierarchical_quantiles',
        'new_Manot_with_hierarchical_quantiles',
        'new_Sandia_with_hierarchical_quantiles'
    ]
    
    feature_sets = { # station feature sets here
        'everything_w_modelq': [
            'rainfall', 'tmin', 'tmax', 'waterlevel_final', 'streamflow_final',
            'Rain_cumulative_7d', 'Rain_cumulative_3d', 'hierarchical_feature_quantile_0.5',
            'waterlevel_upstream', 'streamflow_upstream'
        ],
    }
    
    target_variables = ["waterlevel_final", "streamflow_final"] # target variables here
    Desc = "Everything"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    master_output_dir = os.path.join(script_dir, f"Fit_lstm_runs_{Desc}")
    os.makedirs(master_output_dir, exist_ok=True)
    print(f"All sequential LSTM experiment results will be saved under: {master_output_dir}")
    
    station_data_path = r'/home/aravinthakshan/Projects/main-srip/hierarchial'
    
    all_run_metrics = []
    
    for feature_set_name, features in feature_sets.items():
        print(f"\n--- Running experiments for Feature Set: {feature_set_name} ---")
        current_feature_set_run_metrics = []
        
        for station_file_base_name in new_station_files:
            full_station_path = os.path.join(station_data_path, station_file_base_name + '.csv')
            
            if not os.path.exists(full_station_path):
                print(f"Warning: Station file not found: {full_station_path}. Skipping experiments for this station.")
                continue
            
            # Verify that required columns are present for streamflow predictions
            if 'streamflow_final' in target_variables:
                try:
                    df = pd.read_csv(full_station_path)
                    if 'waterlevel_final' not in df.columns or 'streamflow_final' not in df.columns:
                        print(f"Error: 'waterlevel_final' and 'streamflow_final' are required for streamflow prediction in {full_station_path}. Skipping.")
                        continue
                except Exception as e:
                    print(f"Error reading {full_station_path}: {e}. Skipping.")
                    continue
            
            for target_variable in target_variables:
                run_info_file_path = run_experiment(
                    master_output_dir,
                    full_station_path,
                    features,
                    target_variable,
                    epochs=1 # number of epochs here 
                )
                
                if run_info_file_path:
                    parsed_data = parse_run_info(run_info_file_path)
                    if parsed_data:
                        current_feature_set_run_metrics.append(parsed_data)
                        all_run_metrics.append(parsed_data)
        
        if current_feature_set_run_metrics:
            df_metrics = pd.DataFrame(current_feature_set_run_metrics)
            df_metrics['Target'] = df_metrics['Target'].str.replace('_final', '')
            aggregated_csv_path = os.path.join(master_output_dir, f"aggregated_metrics_{feature_set_name}.csv")
            df_metrics.to_csv(aggregated_csv_path, index=False)
            print(f"\nAggregated metrics for {feature_set_name} saved to: {aggregated_csv_path}")
            
            try:
                print(f"\nGenerating plots for {feature_set_name}...")
                import sys
                sys.path.append(script_dir)
                from plots import plot_station_comparison_combined
                
                plot_station_comparison_combined(
                    csv_path=aggregated_csv_path,
                    base_output_dir=master_output_dir,
                    feature_set_desc=feature_set_name.replace('_', ' ').title()
                )
                print(f"Plotting completed for {feature_set_name}.")
                
            except ImportError as e:
                print(f"Error importing plots module: {e}")
                print("Falling back to subprocess call...")
                
                try:
                    plot_command = [
                        "python",
                        "plots-fit.py",
                        aggregated_csv_path,
                        master_output_dir,
                        feature_set_name.replace('_', ' ').title()
                    ]
                    plot_result = subprocess.run(plot_command, check=True, capture_output=True, text=True, cwd=script_dir)
                    print(f"plots-fit.py completed for {feature_set_name}.")
                    print("plots-fit.py STDOUT:\n", plot_result.stdout)
                    if plot_result.stderr:
                        print("plots-fit.py STDERR:\n", plot_result.stderr)
                        
                except subprocess.CalledProcessError as e:
                    print(f"Error calling plots-fit.py for {feature_set_name}:")
                    print(f"Command: {' '.join(e.cmd)}")
                    print("plots-fit.py STDOUT:\n", e.stdout)
                    print("plots-fit.py STDERR:\n", e.stderr)
                except Exception as e:
                    print(f"An unexpected error occurred while calling plots-fit.py for {feature_set_name}: {e}")
                    
            except Exception as e:
                print(f"An unexpected error occurred while generating plots for {feature_set_name}: {e}")
        else:
            print(f"No metrics collected for feature set {feature_set_name}. Skipping aggregation and plotting.")
    
    if all_run_metrics:
        df_all_metrics = pd.DataFrame(all_run_metrics)
        df_all_metrics['Target'] = df_all_metrics['Target'].str.replace('_final', '')
        overall_csv_path = os.path.join(master_output_dir, "aggregated_metrics_all.csv")
        df_all_metrics.to_csv(overall_csv_path, index=False)
        print(f"\nOverall aggregated metrics saved to: {overall_csv_path}")
        try:
            print(f"\nGenerating overall comparison plots...")
            import sys
            sys.path.append(script_dir)
            from plots import plot_station_comparison_combined
            
            plot_station_comparison_combined(
                csv_path=overall_csv_path,
                base_output_dir=master_output_dir,
                feature_set_desc="All Feature Sets"
            )
            print(f"Overall plotting completed.")
            
        except Exception as e:
            print(f"Error generating overall plots: {e}")
    
    print("\nAll sequential LSTM experiments and plotting completed!")

if __name__ == "__main__":
    main()