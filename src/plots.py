import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

def create_output_directory(base_output_dir):
    output_dir = os.path.join(base_output_dir, "ppt_graphs_combined")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def plot_individual_nse_graphs_combined(df, output_dir, feature_set_desc):
    periods = ['First10', 'Last10']
    period_labels = ['First 10 Years (1961-1970)', 'Last 10 Years (2011-2020)']
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    for period, period_label in zip(periods, period_labels):
        for t in [1, 2, 3]:
            col_name = f'{period}_t{t}_NSE'
            
            if col_name not in df.columns:
                print(f"Column '{col_name}' not found in DataFrame. Skipping plot.")
                continue
            
            plot_data = df[['Station_Name', 'Target', col_name]].dropna(subset=[col_name])
            
            if plot_data.empty:
                print(f"No data for {col_name} across all targets. Skipping plot.")
                continue
            
            g = sns.catplot(
                data=plot_data,
                x='Station_Name',
                y=col_name,
                col='Target',
                kind='bar',
                height=8, aspect=1.2,
                palette='viridis',
                sharey=True
            )
            
            g.set_axis_labels("Stations", "NSE Score", fontsize=16)
            g.set_titles("Target: {col_name}", fontsize=18, fontweight='bold')
            g.set_xticklabels(rotation=45, ha='right', fontsize=12)
            g.set_yticklabels(fontsize=12)
            g.set(ylim=(-0.2, 1.0))
            
            g.fig.suptitle(f'NSE t+{t} - {period_label} Performance ({feature_set_desc})', fontsize=20, fontweight='bold', y=1.03)
            
            for ax in g.axes.flat:
                for p in ax.patches:
                    height = p.get_height()
                    if not np.isnan(height):
                        ax.text(p.get_x() + p.get_width() / 2.,
                                height + 0.01,
                                f'{height:.3f}',
                                ha='center', va='bottom', fontsize=10, color='black')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            filename = f'{output_dir}/combined_nse_individual_{period.lower()}_t{t}_{feature_set_desc.replace(" ", "_")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(g.fig)
            print(f"Saved: {filename}")
    
    for period, period_label in zip(['First', 'Last'], period_labels):
        for t in [1, 2, 3]:
            col_name = f'Top10_{period}_t{t}_NSE'
            
            if col_name not in df.columns:
                print(f"Column '{col_name}' not found in DataFrame. Skipping plot.")
                continue
            
            plot_data = df[['Station_Name', 'Target', col_name]].dropna(subset=[col_name])
            
            if plot_data.empty:
                print(f"No data for {col_name} across all targets. Skipping plot.")
                continue
            
            g = sns.catplot(
                data=plot_data,
                x='Station_Name',
                y=col_name,
                col='Target',
                kind='bar',
                height=8, aspect=1.2,
                palette='plasma',
                sharey=True
            )
            
            g.set_axis_labels("Stations", "NSE Score", fontsize=16)
            g.set_titles("Target: {col_name}", fontsize=18, fontweight='bold')
            g.set_xticklabels(rotation=45, ha='right', fontsize=12)
            g.set_yticklabels(fontsize=12)
            g.set(ylim=(-0.2, 1.0))
            
            g.fig.suptitle(f'Top 10% NSE t+{t} - {period_label} Performance ({feature_set_desc})', fontsize=20, fontweight='bold', y=1.03)
            
            for ax in g.axes.flat:
                for p in ax.patches:
                    height = p.get_height()
                    if not np.isnan(height):
                        ax.text(p.get_x() + p.get_width() / 2.,
                                height + 0.01,
                                f'{height:.3f}',
                                ha='center', va='bottom', fontsize=10, color='black')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            filename = f'{output_dir}/combined_nse_top10_individual_{period.lower()}_t{t}_{feature_set_desc.replace(" ", "_")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(g.fig)
            print(f"Saved: {filename}")

def plot_pbias_combined(df, output_dir, feature_set_desc):
    periods = ['First10', 'Last10']
    period_labels = ['First 10 Years (1961-1970)', 'Last 10 Years (2011-2020)']
    
    for period, period_label in zip(periods, period_labels):
        pbias_cols = [f'{period}_t1_PBIAS', f'{period}_t2_PBIAS', f'{period}_t3_PBIAS']
        
        missing_pbias_cols = [col for col in pbias_cols if col not in df.columns]
        if missing_pbias_cols:
            print(f"Missing PBIAS columns for {period_label}: {missing_pbias_cols}. Skipping plot.")
            continue
        
        plot_data_melted = df[['Station_Name', 'Target'] + pbias_cols].dropna(subset=pbias_cols, how='all').copy()
        
        if plot_data_melted.empty:
            print(f"No PBIAS data available for {period_label} across any target. Skipping plot.")
            continue
        
        plot_data_melted = plot_data_melted.melt(id_vars=['Station_Name', 'Target'],
                                                  var_name='Forecast_Horizon',
                                                  value_name='PBIAS_Value')
        
        plot_data_melted.dropna(subset=['PBIAS_Value'], inplace=True)
        
        if plot_data_melted.empty:
            print(f"No valid PBIAS data after dropping NaNs for {period_label}. Skipping plot.")
            continue
        
        horizon_map = {
            f'{period}_t1_PBIAS': 't+1 Day',
            f'{period}_t2_PBIAS': 't+2 Days',
            f'{period}_t3_PBIAS': 't+3 Days'
        }
        plot_data_melted['Forecast_Horizon'] = plot_data_melted['Forecast_Horizon'].map(horizon_map)
        
        plot_data_melted = plot_data_melted.sort_values(by=['Station_Name', 'Target', 'Forecast_Horizon'])
        
        g = sns.catplot(
            x='Station_Name',
            y='PBIAS_Value',
            hue='Forecast_Horizon',
            col='Target',
            data=plot_data_melted,
            kind='bar',
            palette='dark',
            height=8, aspect=1.5,
            edgecolor='black',
            linewidth=0.7,
            sharey=True
        )
        
        g.set_axis_labels("Stations", "PBIAS (%)", fontsize=18)
        g.set_titles("Target: {col_name}", fontsize=20, fontweight='bold')
        g.set_xticklabels(rotation=45, ha='right', fontsize=14)
        g.set_yticklabels(fontsize=14)
        
        g.fig.suptitle(f'PBIAS Performance Across Stations and Target Types - {period_label} ({feature_set_desc})',
                       fontsize=24, fontweight='bold', y=1.02)
        
        for ax in g.axes.flat:
            min_val = plot_data_melted['PBIAS_Value'].min()
            max_val = plot_data_melted['PBIAS_Value'].max()
            y_lim_min = min(min_val - 10, -30)
            y_lim_max = max(max_val + 10, 30)
            ax.set_ylim([y_lim_min, y_lim_max])
            
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='±10% threshold')
            ax.axhline(y=-10, color='orange', linestyle='--', alpha=0.5)
            ax.axhline(y=25, color='red', linestyle='--', alpha=0.5, label='±25% threshold')
            ax.axhline(y=-25, color='red', linestyle='--', alpha=0.5)
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            
            for container in ax.containers:
                for bar in container:
                    height = bar.get_height()
                    if not np.isnan(height):
                        text_va = 'bottom' if height >= 0 else 'top'
                        text_y_offset = 2 if height >= 0 else -2
                        ax.text(bar.get_x() + bar.get_width() / 2, height + text_y_offset,
                                f'{height:.1f}%', ha='center', va=text_va,
                                fontsize=9, color='black')
            
            if ax == g.axes.flat[0]:
                ax.legend(title='Forecast Horizon', title_fontsize='16', fontsize='14', loc='best')
            else:
                if ax.get_legend():
                    ax.get_legend().remove()
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        filename = f'{output_dir}/grouped_pbias_combined_{period.lower()}_{feature_set_desc.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(g.fig)
        print(f"Saved: {filename}")

def plot_grouped_nse_comparison_combined(df, output_dir, feature_set_desc):
    periods = ['First10', 'Last10']
    period_labels = ['First 10 Years (1961-1970)', 'Last 10 Years (2011-2020)']

    for period, period_label in zip(periods, period_labels):
        nse_cols = [f'{period}_t1_NSE', f'{period}_t2_NSE', f'{period}_t3_NSE']

        missing_nse_cols = [col for col in nse_cols if col not in df.columns]
        if missing_nse_cols:
            print(f"Missing NSE columns for {period_label}: {missing_nse_cols}. Skipping plot.")
            continue

        plot_data_melted = df[['Station_Name', 'Target'] + nse_cols].dropna(subset=nse_cols, how='all').copy()

        if plot_data_melted.empty:
            print(f"No NSE data available for {period_label} across any target to create combined grouped bar plot. Skipping.")
            continue

        plot_data_melted = plot_data_melted.melt(id_vars=['Station_Name', 'Target'],
                                                  var_name='Forecast_Horizon',
                                                  value_name='NSE_Score')

        plot_data_melted.dropna(subset=['NSE_Score'], inplace=True)

        if plot_data_melted.empty:
            print(f"No valid NSE data after dropping NaNs for {period_label}. Skipping plot.")
            continue

        horizon_map = {
            f'{period}_t1_NSE': 't+1 Day',
            f'{period}_t2_NSE': 't+2 Days',
            f'{period}_t3_NSE': 't+3 Days'
        }
        plot_data_melted['Forecast_Horizon'] = plot_data_melted['Forecast_Horizon'].map(horizon_map)

        plot_data_melted = plot_data_melted.sort_values(by=['Station_Name', 'Target', 'Forecast_Horizon'])

        g = sns.catplot(
            x='Station_Name',
            y='NSE_Score',
            hue='Forecast_Horizon',
            col='Target',
            data=plot_data_melted,
            kind='bar',
            palette='viridis',
            height=8, aspect=1.5,
            edgecolor='black',
            linewidth=0.7,
            sharey=True
        )

        g.set_axis_labels("Station", "NSE Score", fontsize=18)
        g.set_titles("Target: {col_name}", fontsize=20, fontweight='bold')
        g.set_xticklabels(rotation=45, ha='right', fontsize=14)
        g.set_yticklabels(fontsize=14)
        g.set(ylim=(-0.2, 1.05))

        g.fig.suptitle(f'NSE Performance Across Stations and Target Types - {period_label} ({feature_set_desc})',
                       fontsize=24, fontweight='bold', y=1.02)

        for ax in g.axes.flat:
            ax.axhline(y=0.75, color='forestgreen', linestyle='--', linewidth=1.5, label='Very Good (NSE > 0.75)', alpha=0.8)
            ax.axhline(y=0.5, color='darkorange', linestyle=':', linewidth=1.5, label='Good (NSE > 0.5)', alpha=0.8)
            ax.grid(axis='y', linestyle='--', alpha=0.6)

            for container in ax.containers:
                for bar in container:
                    height = bar.get_height()
                    if not np.isnan(height):
                        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                                f'{height:.2f}', ha='center', va='bottom',
                                fontsize=9, color='black')
            if ax == g.axes.flat[0]:
                ax.legend(title='Forecast Horizon', title_fontsize='16', fontsize='14', loc='lower right')
            else:
                ax.get_legend().remove()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        filename = f'{output_dir}/grouped_nse_combined_{period.lower()}_{feature_set_desc.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(g.fig)
        print(f"Saved: {filename}")

def plot_horizontal_nse_comparison_combined(df, output_dir, metric_col, title_suffix, feature_set_desc):
    if metric_col not in df.columns:
        print(f"Column '{metric_col}' not found in DataFrame. Skipping plot.")
        return

    plot_data = df[['Station_Name', 'Target', metric_col]].dropna(subset=[metric_col]).copy()

    if plot_data.empty:
        print(f"No valid data after dropping NaNs for {metric_col}. Skipping plot.")
        return

    plot_data['sort_key'] = plot_data.groupby('Target')[metric_col].transform(lambda x: x.rank(method='first', ascending=False))
    plot_data = plot_data.sort_values(by=['Target', 'sort_key'], ascending=[True, True])

    g = sns.catplot(
        x=metric_col,
        y='Station_Name',
        col='Target',
        data=plot_data,
        kind='bar',
        palette='magma',
        height=8, aspect=1.5,
        edgecolor='black',
        linewidth=0.7,
        sharex=True
    )

    g.set_axis_labels("NSE Score", "Station", fontsize=18)
    g.set_titles("Target: {col_name}", fontsize=20, fontweight='bold')
    g.set_xticklabels(fontsize=14)
    g.set_yticklabels(fontsize=14)
    g.set(xlim=(-0.2, 1.05))

    g.fig.suptitle(f'NSE {title_suffix} Across Stations and Target Types ({feature_set_desc})',
                   fontsize=24, fontweight='bold', y=1.02)

    for ax in g.axes.flat:
        ax.axvline(x=0.75, color='forestgreen', linestyle='--', linewidth=1.5, label='Very Good (NSE > 0.75)', alpha=0.8)
        ax.axvline(x=0.5, color='darkorange', linestyle=':', linewidth=1.5, label='Good (NSE > 0.5)', alpha=0.8)
        ax.grid(axis='x', linestyle='--', alpha=0.6)

        for p in ax.patches:
            x_value = p.get_width()
            y_value = p.get_y() + p.get_height() / 2
            if not np.isnan(x_value):
                ax.text(x_value + 0.01, y_value,
                        f'{x_value:.2f}', va='center', ha='left', fontsize=10, color='black')

        if ax == g.axes.flat[0]:
            ax.legend(fontsize=12)
        else:
            ax.get_legend().remove()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = f'{output_dir}/horizontal_nse_combined_{metric_col.lower()}_{feature_set_desc.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(g.fig)
    print(f"Saved: {filename}")

def plot_modelq_nse_comparison(df, output_dir, feature_set_desc):
    periods = ['First10', 'Last10']
    period_labels = ['First 10 Years (1961-1970)', 'Last 10 Years (2011-2020)']
    
    # Filter for streamflow target only
    plot_data = df[df['Target'] == 'streamflow'].copy()
    
    if plot_data.empty:
        print("No streamflow data available for modelq comparison. Skipping plot.")
        return
    
    # Filter for feature sets containing "Modelq" and without
    modelq_data = plot_data[plot_data['Feature_Set_Name'].str.contains('Modelq', case=False)]
    non_modelq_data = plot_data[~plot_data['Feature_Set_Name'].str.contains('Modelq', case=False)]
    
    if modelq_data.empty or non_modelq_data.empty:
        print("Insufficient data for modelq comparison (need both Modelq and non-Modelq feature sets). Skipping plot.")
        return
    
    for period, period_label in zip(periods, period_labels):
        for t in [1, 2, 3]:
            col_name = f'{period}_t{t}_NSE'
            
            if col_name not in plot_data.columns:
                print(f"Column '{col_name}' not found in DataFrame. Skipping modelq comparison plot.")
                continue
            
            # Combine modelq and non-modelq data
            plot_data_melted = plot_data[['Station_Name', 'Feature_Set_Name', col_name]].dropna(subset=[col_name])
            plot_data_melted = plot_data_melted.melt(id_vars=['Station_Name', 'Feature_Set_Name'],
                                                     var_name='Metric',
                                                     value_name='NSE_Score')
            
            if plot_data_melted.empty:
                print(f"No valid NSE data for {col_name} in streamflow modelq comparison. Skipping plot.")
                continue
            
            # Categorize feature sets as Modelq or Non-Modelq
            plot_data_melted['Feature_Type'] = plot_data_melted['Feature_Set_Name'].apply(
                lambda x: 'With Modelq' if 'Modelq' in x else 'Without Modelq'
            )
            
            g = sns.catplot(
                x='Station_Name',
                y='NSE_Score',
                hue='Feature_Type',
                data=plot_data_melted,
                kind='bar',
                palette='Set2',
                height=8, aspect=1.5,
                edgecolor='black',
                linewidth=0.7
            )
            
            g.set_axis_labels("Stations", "NSE Score", fontsize=18)
            g.set_titles(f"Streamflow NSE t+{t} - {period_label}", fontsize=20, fontweight='bold')
            g.set_xticklabels(rotation=45, ha='right', fontsize=14)
            g.set_yticklabels(fontsize=14)
            g.set(ylim=(-0.2, 1.05))
            
            g.fig.suptitle(f'Streamflow NSE t+{t} Comparison: With vs Without Modelq - {period_label} ({feature_set_desc})',
                           fontsize=24, fontweight='bold', y=1.02)
            
            for ax in g.axes.flat:
                ax.axhline(y=0.75, color='forestgreen', linestyle='--', linewidth=1.5, label='Very Good (NSE > 0.75)', alpha=0.8)
                ax.axhline(y=0.5, color='darkorange', linestyle=':', linewidth=1.5, label='Good (NSE > 0.5)', alpha=0.8)
                ax.grid(axis='y', linestyle='--', alpha=0.6)
                
                for container in ax.containers:
                    for bar in container:
                        height = bar.get_height()
                        if not np.isnan(height):
                            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                                    f'{height:.2f}', ha='center', va='bottom',
                                    fontsize=9, color='black')
                ax.legend(title='Feature Type', title_fontsize='16', fontsize='14', loc='lower right')
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            filename = f'{output_dir}/modelq_nse_comparison_{period.lower()}_t{t}_{feature_set_desc.replace(" ", "_")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(g.fig)
            print(f"Saved: {filename}")

def plot_station_comparison_combined(csv_path, base_output_dir, feature_set_desc=""):
    print(f"Attempting to read CSV from: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}. Cannot generate plots.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return

    if 'Target' not in df.columns:
        print("Error: 'Target' column not found in the CSV. Please ensure your data has a 'Target' column.")
        return
    df['Target'] = df['Target'].astype(str).str.lower()

    df_filtered_targets = df[df['Target'].isin(['waterlevel', 'streamflow'])].copy()

    if df_filtered_targets.empty:
        print(f"No data found for 'waterlevel' or 'streamflow' targets in the CSV after filtering.")
        return

    print(f"Found data for {len(df_filtered_targets['Station_Name'].unique())} stations across 'waterlevel' and 'streamflow' targets.")

    output_dir = create_output_directory(base_output_dir)
    print(f"Graphs will be saved to: {output_dir}/")

    print("\nGenerating combined individual NSE graphs (faceted by Target)...")
    plot_individual_nse_graphs_combined(df_filtered_targets, output_dir, feature_set_desc)

    print("\nGenerating combined grouped NSE comparison graphs (faceted by Target)...")
    plot_grouped_nse_comparison_combined(df_filtered_targets, output_dir, feature_set_desc)

    print("\nGenerating modelq NSE comparison graphs for streamflow...")
    plot_modelq_nse_comparison(df_filtered_targets, output_dir, feature_set_desc)

    print("\nGenerating combined horizontal NSE graphs (example: First10_t1_NSE, faceted by Target)...")
    plot_horizontal_nse_comparison_combined(df_filtered_targets, output_dir,
                                            metric_col='First10_t1_NSE', title_suffix='t+1 Day (First 10 Years)', feature_set_desc=feature_set_desc)
    print("\nGenerating combined horizontal NSE graphs (example: Last10_t3_NSE, faceted by Target)...")
    plot_horizontal_nse_comparison_combined(df_filtered_targets, output_dir,
                                            metric_col='Last10_t3_NSE', title_suffix='t+3 Days (Last 10 Years)', feature_set_desc=feature_set_desc)

    print("\nGenerating combined PBIAS graphs (faceted by Target and grouped by Horizon)...")
    plot_pbias_combined(df_filtered_targets, output_dir, feature_set_desc)

    print(f"\n=== SUMMARY STATISTICS ===")
    nse_cols_to_summarize = [
        col for col in df_filtered_targets.columns
        if 'NSE' in col and (
            'First10_t1' in col or 'First10_t2' in col or 'First10_t3' in col or
            'Last10_t1' in col or 'Last10_t2' in col or 'Last10_t3' in col or
            'Top10_First' in col or 'Top10_Last' in col
        )
    ]
    if nse_cols_to_summarize:
        print("\nNSE Statistics by Target:")
        for target_val in df_filtered_targets['Target'].unique():
            print(f"\n--- {target_val.upper()} ---")
            df_sub = df_filtered_targets[df_filtered_targets['Target'] == target_val]
            for col in nse_cols_to_summarize:
                if not df_sub[col].isna().all():
                    print(f"{col}: Mean={df_sub[col].mean():.3f}, Max={df_sub[col].max():.3f}, Min={df_sub[col].min():.3f}")

    pbias_cols_to_summarize = [
        col for col in df_filtered_targets.columns
        if 'PBIAS' in col and ('First10' in col or 'Last10' in col)
    ]
    if pbias_cols_to_summarize:
        print("\nPBIAS Statistics by Target:")
        for target_val in df_filtered_targets['Target'].unique():
            print(f"\n--- {target_val.upper()} ---")
            df_sub = df_filtered_targets[df_filtered_targets['Target'] == target_val]
            for col in pbias_cols_to_summarize:
                if not df_sub[col].isna().all():
                    print(f"{col}: Mean={df_sub[col].mean():.1f}%, Abs Mean={df_sub[col].abs().mean():.1f}%")

    # Add modelq vs non-modelq comparison for streamflow NSE
    if 'Feature_Set_Name' in df_filtered_targets.columns:
        print("\nStreamflow NSE Comparison: With vs Without Modelq")
        modelq_data = df_filtered_targets[(df_filtered_targets['Target'] == 'streamflow') & 
                                         (df_filtered_targets['Feature_Set_Name'].str.contains('Modelq', case=False))]
        non_modelq_data = df_filtered_targets[(df_filtered_targets['Target'] == 'streamflow') & 
                                             (~df_filtered_targets['Feature_Set_Name'].str.contains('Modelq', case=False))]
        for period in ['First10', 'Last10']:
            for t in [1, 2, 3]:
                col = f'{period}_t{t}_NSE'
                if col in df_filtered_targets.columns:
                    modelq_mean = modelq_data[col].mean() if not modelq_data[col].isna().all() else np.nan
                    non_modelq_mean = non_modelq_data[col].mean() if not non_modelq_data[col].isna().all() else np.nan
                    if not (np.isnan(modelq_mean) or np.isnan(non_modelq_mean)):
                        print(f"{col}: With Modelq Mean={modelq_mean:.3f}, Without Modelq Mean={non_modelq_mean:.3f}, Improvement={modelq_mean - non_modelq_mean:.3f}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python plots.py <csv_path> <base_output_dir> <feature_set_desc>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    base_output_dir = sys.argv[2]
    feature_set_desc = sys.argv[3]
    plot_station_comparison_combined(csv_path, base_output_dir, feature_set_desc)