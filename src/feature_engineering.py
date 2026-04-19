import numpy as np

def add_features(df):
    df = df.copy()

    df["growth_dist_interaction"] = df["area_growth_rate_ha_per_h"] * df["dist_min_ci_0_5h"]
    df["speed_alignment"] = df["centroid_speed_m_per_h"] * df["alignment_cos"]
    df["closing_ratio"] = df["closing_speed_m_per_h"] / (df["dist_min_ci_0_5h"] + 1e-6)
    df["growth_acceleration"] = df["area_growth_rate_ha_per_h"] - df["area_growth_rel_0_5h"]
    df["log_dist"] = np.log1p(df["dist_min_ci_0_5h"])
    df["inv_dist"] = 1 / (df["dist_min_ci_0_5h"] + 1)
    df["growth_over_dist"] = df["area_growth_rate_ha_per_h"] / (df["dist_min_ci_0_5h"] + 1)
    df["hour_bucket"] = (df["event_start_hour"] // 6).astype(int)
    df["movement_consistency"] = df["dist_std_ci_0_5h"] / (df["dist_min_ci_0_5h"] + 1)
    df["area_ratio"] = df["area_growth_abs_0_5h"] / (df["area_first_ha"] + 1)

    df["directional_risk"] = df["alignment_cos"] * df["closing_speed_m_per_h"]
    df["growth_norm"] = df["area_growth_rate_ha_per_h"] / (df["area_first_ha"] + 1)
    df["movement_strength"] = np.abs(df["alignment_cos"]) * df["centroid_speed_m_per_h"]

    return df