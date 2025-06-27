import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
np.random.seed(42) # Add this line at the very top of your script
from catboost import CatBoostRegressor
# --- Configuration ---
CSV_FILE_PATH = "/home/aravinthakshan/Projects/main-srip/hierarchial/new_Handia_with_hierarchical_quantiles.csv"  # <--- IMPORTANT: REPLACE WITH YOUR CSV FILE PATH
TARGET_STREAMFLOW_COLUMN = "streamflow_final" # <--- IMPORTANT: REPLACE WITH YOUR TARGET STREAMFLOW COLUMN NAME
DATE_COLUMN_NAME = "date" # <--- Adjust if your date column has a different name

LAG_K = 1 # Lead time (k days: T-k data predicts T)

# Define the names of your input features from the CSV
INPUT_RAINFALL_COLUMN = 'rainfall' # check this once
INPUT_TMAX_COLUMN = 'tmax'
INPUT_TMIN_COLUMN = 'tmin'
INPUT_WATERLEVEL_UPSTREAM_COLUMN = 'waterlevel_upstream' # I Guess this doesnt matter 
INPUT_STREAMFLOW_UPSTREAM_COLUMN = 'streamflow_upstream' # This will be our Q_feeder
# --- Evaluation Metrics ---

# CatBoost training parameters (can be tuned)
CATBOOST_PARAMS = {
    'iterations': 500, # Number of boosting iterations
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3, # L2 regularization coefficient
    'loss_function': 'RMSE', # For regression
    'verbose': 0, # Suppress training output for cleaner console
    'random_seed': 42
}

# --- Evaluation Metrics ---
def calculate_nse(observations, simulations):
    """
    Calculates Nash-Sutcliffe Efficiency (NSE).
    NSE = 1 - (sum((observations - simulations)^2) / sum((observations - mean(observations))^2))
    """
    numerator = np.sum((observations - simulations)**2)
    denominator = np.sum((observations - np.mean(observations))**2)
    if denominator == 0:
        return np.nan # Avoid division by zero, though unlikely for streamflow
    return 1 - (numerator / denominator)

def calculate_pbias(observations, simulations):
    """
    Calculates Percent Bias (PBIAS).
    PBIAS = (sum(simulations - observations) * 100) / sum(observations)
    """
    numerator = np.sum(simulations - observations) * 100
    denominator = np.sum(observations)
    if denominator == 0:
        return np.nan # Avoid division by zero
    return numerator / denominator

# --- Data Loading and Preparation ---
def load_and_prepare_data(csv_path, target_col, date_col, k_lag):
    """
    Loads data, extracts year, creates lagged variables, and handles NaNs.
    """
    df = pd.read_csv(csv_path)

    # Convert date column to datetime and extract year
    df[date_col] = pd.to_datetime(df[date_col], format="%d-%m-%Y") # Assuming DD-MM-YYYY
    df['year'] = df[date_col].dt.year

    # Create lagged features for all required inputs
    df[f'{INPUT_RAINFALL_COLUMN}_lag{k_lag}'] = df[INPUT_RAINFALL_COLUMN].shift(k_lag)
    df[f'{INPUT_TMAX_COLUMN}_lag{k_lag}'] = df[INPUT_TMAX_COLUMN].shift(k_lag)
    df[f'{INPUT_TMIN_COLUMN}_lag{k_lag}'] = df[INPUT_TMIN_COLUMN].shift(k_lag)
    df[f'{INPUT_WATERLEVEL_UPSTREAM_COLUMN}_lag{k_lag}'] = df[INPUT_WATERLEVEL_UPSTREAM_COLUMN].shift(k_lag)
    df[f'{INPUT_STREAMFLOW_UPSTREAM_COLUMN}_lag{k_lag}'] = df[INPUT_STREAMFLOW_UPSTREAM_COLUMN].shift(k_lag)

    # Drop rows with NaN values resulting from the shift operation
    df.dropna(inplace=True)

    print(f"Data loaded and prepared. Shape: {df.shape}")
    print(f"Columns after lagging: {df.columns.tolist()}")
    return df

# --- Parameter Estimation using CatBoost ---
def estimate_catboost_models(df, target_col, feeder_col_lagged, rainfall_col_lagged, covariate_cols_lagged, cb_params):
    """
    Estimates mean and variance models using CatBoost Regressor.
    Returns trained CatBoost models.
    """
    trained_models = {
        'mean_P_eq_0': None,
        'mean_P_gt_0': None,
        'variance_P_eq_0': None,
        'variance_P_gt_0': None
    }

    # Split data based on lagged rainfall
    df_p_eq_0 = df[df[rainfall_col_lagged] == 0].copy()
    df_p_gt_0 = df[df[rainfall_col_lagged] > 0].copy()

    # --- Mean Model Estimation ---
    print("\n--- Training CatBoost Mean Models (Training Data) ---")

    # Predictors for mean model: Q_feeder, and X (covariates), potentially rainfall for P>0
    # CatBoost handles categorical features if specified. Here we treat all as numerical.
    cat_features = [] # Assuming all inputs are numerical for now. Modify if you have true categorical inputs.

    # Case 1: Rainfall == 0
    if not df_p_eq_0.empty:
        print("Training Mean Model for P = 0 case...")
        X_p_eq_0 = df_p_eq_0[[feeder_col_lagged] + covariate_cols_lagged]
        y_p_eq_0 = df_p_eq_0[target_col]
        
        mean_model_p_eq_0 = CatBoostRegressor(**cb_params)
        mean_model_p_eq_0.fit(X_p_eq_0, y_p_eq_0, cat_features=cat_features)
        trained_models['mean_P_eq_0'] = mean_model_p_eq_0
        # Calculate residuals for variance model
        residuals_p_eq_0 = y_p_eq_0 - mean_model_p_eq_0.predict(X_p_eq_0)
    else:
        print("No data for P = 0 case in training set for mean model.")
        residuals_p_eq_0 = pd.Series([])

    # Case 2: Rainfall > 0
    if not df_p_gt_0.empty:
        print("Training Mean Model for P > 0 case...")
        X_p_gt_0 = df_p_gt_0[[feeder_col_lagged, rainfall_col_lagged] + covariate_cols_lagged]
        y_p_gt_0 = df_p_gt_0[target_col]
        
        mean_model_p_gt_0 = CatBoostRegressor(**cb_params)
        mean_model_p_gt_0.fit(X_p_gt_0, y_p_gt_0, cat_features=cat_features)
        trained_models['mean_P_gt_0'] = mean_model_p_gt_0
        # Calculate residuals for variance model
        residuals_p_gt_0 = y_p_gt_0 - mean_model_p_gt_0.predict(X_p_gt_0)
    else:
        print("No data for P > 0 case in training set for mean model.")
        residuals_p_gt_0 = pd.Series([])

    # --- Variance Model Estimation (Approximation) ---
    print("\n--- Training CatBoost Variance Models (Training Data) ---")
    print("NOTE: CatBoost is used to model log(squared_residuals) as a proxy for log(variance).")
    print("This is a simplified approach, not full Bayesian inference.")

    # Predictors for variance model: Q_feeder, and X (covariates), potentially rainfall for P>0
    
    # Case 1: Rainfall == 0
    if not residuals_p_eq_0.empty and len(residuals_p_eq_0) > 1: # Need at least 2 samples for regression
        print("Training Variance Model for P = 0 case...")
        # Dependent variable: log of squared residuals
        dependent_var_phi_eq_0 = np.log(residuals_p_eq_0**2 + 1e-6) # Add epsilon to avoid log(0)
        X_phi_eq_0 = df_p_eq_0[[feeder_col_lagged] + covariate_cols_lagged]
        
        variance_model_p_eq_0 = CatBoostRegressor(**cb_params)
        variance_model_p_eq_0.fit(X_phi_eq_0, dependent_var_phi_eq_0, cat_features=cat_features)
        trained_models['variance_P_eq_0'] = variance_model_p_eq_0
    else:
        print("Insufficient data for P = 0 case for variance model.")

    # Case 2: Rainfall > 0
    if not residuals_p_gt_0.empty and len(residuals_p_gt_0) > 1: # Need at least 2 samples
        print("Training Variance Model for P > 0 case...")
        dependent_var_phi_gt_0 = np.log(residuals_p_gt_0**2 + 1e-6)
        X_phi_gt_0 = df_p_gt_0[[feeder_col_lagged, rainfall_col_lagged] + covariate_cols_lagged]
        
        variance_model_p_gt_0 = CatBoostRegressor(**cb_params)
        variance_model_p_gt_0.fit(X_phi_gt_0, dependent_var_phi_gt_0, cat_features=cat_features)
        trained_models['variance_P_gt_0'] = variance_model_p_gt_0
    else:
        print("Insufficient data for P > 0 case for variance model.")

    return trained_models

# --- Prediction and Simulation (Forward Pass for Test Data) ---
def predict_and_simulate_streamflow_catboost(df_test, trained_models,
                                   target_col, feeder_col_lagged, rainfall_col_lagged, covariate_cols_lagged):
    """
    Predicts mean and standard deviation using trained CatBoost models,
    then simulates streamflow for a test DataFrame.
    """
    predicted_means = []
    predicted_stds = []
    simulated_qs = []

    for index, row in df_test.iterrows():
        q_feeder_input = row[feeder_col_lagged]
        rainfall_input = row[rainfall_col_lagged]
        
        # Prepare input features for CatBoost prediction for this single row
        # Ensure the order and names match the training data
        input_features_p_eq_0 = pd.DataFrame([[q_feeder_input] + [row[col] for col in covariate_cols_lagged]],
                                             columns=[feeder_col_lagged] + covariate_cols_lagged)
        input_features_p_gt_0 = pd.DataFrame([[q_feeder_input, rainfall_input] + [row[col] for col in covariate_cols_lagged]],
                                             columns=[feeder_col_lagged, rainfall_col_lagged] + covariate_cols_lagged)


        # Determine which set of models to use based on rainfall_input
        if rainfall_input == 0:
            mean_model = trained_models['mean_P_eq_0']
            variance_model = trained_models['variance_P_eq_0']
            current_input_features = input_features_p_eq_0
        else:
            mean_model = trained_models['mean_P_gt_0']
            variance_model = trained_models['variance_P_gt_0']
            current_input_features = input_features_p_gt_0

        mu_streamflow = np.nan
        sigma_streamflow = np.nan
        simulated_q = np.nan

        if mean_model is not None:
            # Predict mean streamflow using the trained CatBoost model
            mu_streamflow = mean_model.predict(current_input_features)[0]
            mu_streamflow = max(0.1, mu_streamflow) # Ensure positive
            if variance_model is not None:
                # Predict log(sigma^2) using the trained CatBoost variance model
                log_sigma_sq = variance_model.predict(current_input_features)[0]
                sigma_squared_approx = np.exp(log_sigma_sq)
                sigma_streamflow = np.sqrt(sigma_squared_approx)
                sigma_streamflow = max(0.01, sigma_streamflow) # Ensure positive and non-zero

                # # --- Simulate Streamflow using Lognormal Distribution ---
                # if mu_streamflow > 0 and sigma_streamflow > 0:
                #     # Convert mu and sigma of Q to mu_lnQ and sigma_lnQ for log-normal distribution
                #     sigma_lnQ_sq = np.log(1 + (sigma_streamflow / mu_streamflow)**2)
                #     sigma_lnQ = np.sqrt(sigma_lnQ_sq)
                #     mu_lnQ = np.log(mu_streamflow) - 0.5 * sigma_lnQ_sq

                #     # z_score = norm.rvs() # Generate a random value from a standard normal distribution
                #     z_score = norm.ppf(0.5)
                #     log_simulated_q = mu_lnQ + sigma_lnQ * z_score
                #     simulated_q = np.exp(log_simulated_q)

                if mu_streamflow > 0 and sigma_streamflow > 0:
                    # Convert mu and sigma of Q to mu_lnQ and sigma_lnQ for log-normal distribution
                    sigma_lnQ_sq = np.log(1 + (sigma_streamflow / mu_streamflow)**2)
                    sigma_lnQ = np.sqrt(sigma_lnQ_sq)
                    mu_lnQ = np.log(mu_streamflow) - 0.5 * sigma_lnQ_sq

                    # Use quantiles instead of random sampling
                    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]  # 10th, 50th, 90th percentiles
                    simulated_values = []
                    
                    for q in quantiles:
                        z_score = norm.ppf(q)  # Deterministic quantile
                        log_simulated_q = mu_lnQ + sigma_lnQ * z_score
                        simulated_values.append(np.exp(log_simulated_q))
                    
                    # Use median (50th percentile) as main prediction
                    simulated_q = simulated_values[-1]  # 90th percentile

        
        predicted_means.append(mu_streamflow)
        predicted_stds.append(sigma_streamflow)
        simulated_qs.append(simulated_q)

    df_test['predicted_mean_Q'] = predicted_means
    df_test['predicted_std_Q'] = predicted_stds
    df_test['simulated_Q'] = simulated_qs
    return df_test

# --- Main Execution Block ---
if __name__ == "__main__":
    # 1. Load and Prepare Data
    print(f"Loading data from: {CSV_FILE_PATH}")
    full_df = load_and_prepare_data(CSV_FILE_PATH, TARGET_STREAMFLOW_COLUMN, DATE_COLUMN_NAME, LAG_K)

    # Define lagged column names based on configuration
    RAIN_LAGGED = f'{INPUT_RAINFALL_COLUMN}_lag{LAG_K}'
    FEEDER_LAGGED = f'{INPUT_STREAMFLOW_UPSTREAM_COLUMN}_lag{LAG_K}'
    COVARIATES_LAGGED = [
        f'{INPUT_TMAX_COLUMN}_lag{LAG_K}',
        f'{INPUT_TMIN_COLUMN}_lag{LAG_K}',
        f'{INPUT_WATERLEVEL_UPSTREAM_COLUMN}_lag{LAG_K}'
    ]

    # 2. Temporal Split
    print("\n--- Splitting Data into Training and Test Sets ---")
    training_mask = (full_df['year'] > 1970) & (full_df['year'] < 2011)
    test_first_10_mask = (full_df['year'] >= 1961) & (full_df['year'] <= 1970)
    test_last_10_mask = (full_df['year'] >= 2011) & (full_df['year'] <= 2020)

    df_train = full_df[training_mask].copy()
    df_test_first_10 = full_df[test_first_10_mask].copy()
    df_test_last_10 = full_df[test_last_10_mask].copy()

    print(f"Training set size: {len(df_train)} days")
    print(f"Test Set (1961-1970) size: {len(df_test_first_10)} days")
    print(f"Test Set (2011-2020) size: {len(df_test_last_10)} days")

    # 3. Estimate Parameters (Train CatBoost Models) on Training Data
    if df_train.empty:
        print("\nERROR: Training data is empty. Cannot train models.")
        trained_catboost_models = {}
    else:
        trained_catboost_models = estimate_catboost_models(
            df_train, TARGET_STREAMFLOW_COLUMN, FEEDER_LAGGED, RAIN_LAGGED, COVARIATES_LAGGED, CATBOOST_PARAMS
        )

        print("\n\n--- CatBoost Models Trained ---")
        for model_name, model_obj in trained_catboost_models.items():
            if model_obj:
                print(f"  {model_name}: Trained")
            else:
                print(f"  {model_name}: Not trained (due to insufficient data or error)")

        # 4. Predict and Simulate Streamflow on Test Sets
        print("\n--- Making Predictions and Simulations on Test Sets ---")

        if trained_catboost_models['mean_P_eq_0'] is not None and trained_catboost_models['mean_P_gt_0'] is not None:
            print("Predicting for Test Set: 1961-1970")
            df_test_first_10_results = predict_and_simulate_streamflow_catboost(
                df_test_first_10, trained_catboost_models,
                TARGET_STREAMFLOW_COLUMN, FEEDER_LAGGED, RAIN_LAGGED, COVARIATES_LAGGED
            )

            print("Predicting for Test Set: 2011-2020")
            df_test_last_10_results = predict_and_simulate_streamflow_catboost(
                df_test_last_10, trained_catboost_models,
                TARGET_STREAMFLOW_COLUMN, FEEDER_LAGGED, RAIN_LAGGED, COVARIATES_LAGGED
            )

            # 5. Evaluate Performance
            print("\n--- Model Performance Evaluation ---")

            # Evaluate First 10 Years Test Set
            obs_first_10 = df_test_first_10_results[TARGET_STREAMFLOW_COLUMN]
            sim_first_10 = df_test_first_10_results['simulated_Q']
            valid_indices_first_10 = sim_first_10.notna() & obs_first_10.notna() # Check both
            obs_first_10_valid = obs_first_10[valid_indices_first_10]
            sim_first_10_valid = sim_first_10[valid_indices_first_10]

            if not obs_first_10_valid.empty:
                nse_first_10 = calculate_nse(obs_first_10_valid, sim_first_10_valid)
                pbias_first_10 = calculate_pbias(obs_first_10_valid, sim_first_10_valid)
                print(f"\nResults for Test Set (1961-1970):")
                print(f"  NSE: {nse_first_10:.4f}")
                print(f"  PBIAS: {pbias_first_10:.4f} %")
            else:
                print("\nNo valid data points for evaluation in Test Set (1961-1970).")


            # Evaluate Last 10 Years Test Set
            obs_last_10 = df_test_last_10_results[TARGET_STREAMFLOW_COLUMN]
            sim_last_10 = df_test_last_10_results['simulated_Q']
            valid_indices_last_10 = sim_last_10.notna() & obs_last_10.notna() # Check both
            obs_last_10_valid = obs_last_10[valid_indices_last_10]
            sim_last_10_valid = sim_last_10[valid_indices_last_10]

            if not obs_last_10_valid.empty:
                nse_last_10 = calculate_nse(obs_last_10_valid, sim_last_10_valid)
                pbias_last_10 = calculate_pbias(obs_last_10_valid, sim_last_10_valid)
                print(f"\nResults for Test Set (2011-2020):")
                print(f"  NSE: {nse_last_10:.4f}")
                print(f"  PBIAS: {pbias_last_10:.4f} %")
            else:
                print("\nNo valid data points for evaluation in Test Set (2011-2020).")
        else:
            print("\nCannot perform predictions. Mean models for P=0 or P>0 were not trained.")

    print("\n--- Script Finished ---")
    import matplotlib.pyplot as plt

    # --- 6. Simulate on the Training Set with CatBoost ---
    df_train_preds_cb = predict_and_simulate_streamflow_catboost(
        df_train, trained_catboost_models,
        TARGET_STREAMFLOW_COLUMN, FEEDER_LAGGED, RAIN_LAGGED, COVARIATES_LAGGED
    )

    # --- 7. Index by date for plotting ---
    for df in (df_train_preds_cb, df_test_first_10_results, df_test_last_10_results):
        df[DATE_COLUMN_NAME] = pd.to_datetime(df[DATE_COLUMN_NAME], format="%d-%m-%Y")
        df.set_index(DATE_COLUMN_NAME, inplace=True)

    # --- 8. Plotting helper ---
    def plot_streamflow(df, title):
        plt.figure()
        plt.plot(df.index, df[TARGET_STREAMFLOW_COLUMN], label="Observed")
        plt.plot(df.index, df["simulated_Q"], label="Simulated (CatBoost)", alpha=0.7)
        plt.xlabel("Date")
        plt.ylabel("Streamflow")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # # --- 9. Generate the three time‐series plots ---
    plot_streamflow(df_train_preds_cb,         "Training (1971–2010) — CatBoost")
    plot_streamflow(df_test_first_10_results,  "Test (1961–1970) — CatBoost")
    plot_streamflow(df_test_last_10_results,   "Test (2011–2020) — CatBoost")

    # 6. Add hierarchical_streamflow predictions to full dataset
    # print("\n--- Adding Hierarchical Predictions to Full Dataset ---")
    
    # # Create a copy of the full dataset for predictions
    # full_df_with_predictions = full_df.copy()
    
    # # Initialize the hierarchical_streamflow column with NaN
    # full_df_with_predictions['hierarchical_streamflow'] = np.nan
    
    # if trained_catboost_models['mean_P_eq_0'] is not None and trained_catboost_models['mean_P_gt_0'] is not None:
    #     # Predict on the entire dataset
    #     full_df_predictions = predict_and_simulate_streamflow_catboost(
    #         full_df_with_predictions, trained_catboost_models,
    #         TARGET_STREAMFLOW_COLUMN, FEEDER_LAGGED, RAIN_LAGGED, COVARIATES_LAGGED
    #     )
        
    #     # Copy the simulated values to hierarchical_streamflow column
    #     full_df_predictions['hierarchical_streamflow'] = full_df_predictions['simulated_Q']
        
    #     # Save to CSV with hierarchical predictions
    #     output_csv_path = CSV_FILE_PATH.replace('.csv', '_with_hierarchical_predictions.csv')
    #     full_df_predictions.to_csv(output_csv_path, index=False)
        
    #     print(f"Full dataset with hierarchical predictions saved to: {output_csv_path}")
    #     print(f"Added hierarchical_streamflow column with {full_df_predictions['hierarchical_streamflow'].notna().sum()} predictions")
        
    #     # Display sample of results
    #     print("\nSample predictions:")
    #     sample_cols = [DATE_COLUMN_NAME, TARGET_STREAMFLOW_COLUMN, 'hierarchical_streamflow', RAIN_LAGGED]
    #     print(full_df_predictions[sample_cols].head(10))
        
    # else:
    #     print("Cannot generate hierarchical predictions - models not properly trained")
    #     full_df_predictions = full_df_with_predictions



# --- Model Performance Evaluation --- Streamflow
# Results for Test Set (1961-1970):
#   NSE: 0.8836
#   PBIAS: 14.7970 %

# Results for Test Set (2011-2020):
#   NSE: 0.7590
#   PBIAS: 1.3489 %

# --- Model Performance Evaluation --- Waterlevel
# Results for Test Set (1961-1970):
#   NSE: 0.7214
#   PBIAS: 0.3264 %

# Results for Test Set (2011-2020):
#   NSE: 0.7643
#   PBIAS: -0.0833 %