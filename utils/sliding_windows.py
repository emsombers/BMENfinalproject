import numpy as np
import pandas as pd

def create_windows(anxiety_csv_path, output_csv_path, window_size, step_size):
    df = pd.read_csv(anxiety_csv_path)
    df.columns = df.columns.str.strip().str.lower().str.replace("-", "_")

    anxiety_z = df["anxiety"].values
    anxiety_scaled = df["scaled_anxiety"].values
    fmri_tr = df["fmri_tr"].values

    rows = []
    first_valid = np.where(~np.isnan(anxiety_z))[0][0]

    for start in range(first_valid, len(anxiety_z) - window_size + 1, step_size):
        end = start + window_size
        anx_window = anxiety_z[start:end]
        scaled_window = anxiety_scaled[start:end]

        if np.isnan(anx_window).any() or np.isnan(scaled_window).any():
            continue

        # ✅ Consistent naming
        row = {
        "window_start_index": start,
        "window_start_TR": fmri_tr[start],
        "mean_anxiety": np.mean(anx_window),
        "mean_scaled_anxiety": np.mean(scaled_window),
        "median_anxiety": np.median(anx_window),
        "median_scaled_anxiety": np.median(scaled_window)
    }
        print("DEBUG ROW:", row)  # Add this line
        rows.append(row)

    # ✅ Safe DataFrame creation
    result_df = pd.DataFrame(rows)

    # Binning function
    def bin_zscore(x):
        if x <= -0.5:
            return "low"
        elif x >= 0.5:
            return "high"
        else:
            return "medium"

    # Apply both binning labels
    result_df["binned_by_mean"] = result_df["mean_anxiety"].apply(bin_zscore)
    result_df["binned_by_median"] = result_df["median_anxiety"].apply(bin_zscore)

    # ✅ Save and confirm
    result_df.to_csv(output_csv_path, index=False)
    print(f"✅ Saved: {output_csv_path} with {len(result_df)} windows")
    print("📊 Columns:", result_df.columns.tolist())
    print("📊 Bin counts (mean):")
    print(result_df["binned_by_mean"].value_counts())
    print("📊 Bin counts (median):")
    print(result_df["binned_by_median"].value_counts())