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

def add_features_v2(df):
    df = df.copy()
    eps = 1e-6

    # -------------------------------------------------
    # 1) observation quality / reliability
    # -------------------------------------------------
    df["is_single_perimeter"] = (df["num_perimeters_0_5h"] <= 1).astype(int)
    df["is_low_dt"] = (df["dt_first_last_0_5h"] < 0.5).astype(int)
    df["is_low_quality_obs"] = (
        (df["num_perimeters_0_5h"] <= 1) |
        (df["dt_first_last_0_5h"] < 0.5)
    ).astype(int)

    df["observation_quality"] = (
        np.log1p(df["num_perimeters_0_5h"]) *
        np.log1p(df["dt_first_last_0_5h"] + 1.0)
    )

    df["obs_density"] = df["num_perimeters_0_5h"] / (df["dt_first_last_0_5h"] + 0.5)
    df["obs_strength"] = df["num_perimeters_0_5h"] * df["dt_first_last_0_5h"]

    # -------------------------------------------------
    # 2) safer distance / risk transforms
    # -------------------------------------------------
    df["log_dist"] = np.log1p(df["dist_min_ci_0_5h"])
    df["inv_dist"] = 1.0 / (df["dist_min_ci_0_5h"] + 1.0)
    df["dist_km"] = df["dist_min_ci_0_5h"] / 1000.0

    # -------------------------------------------------
    # 3) interaction features (more interpretable)
    # -------------------------------------------------
    df["closing_ratio"] = df["closing_speed_m_per_h"] / (df["dist_min_ci_0_5h"] + eps)
    df["growth_over_dist"] = df["area_growth_rate_ha_per_h"] / (df["dist_min_ci_0_5h"] + 1.0)

    df["directional_risk"] = df["alignment_cos"] * df["closing_speed_m_per_h"]
    df["movement_strength"] = np.abs(df["alignment_cos"]) * df["centroid_speed_m_per_h"]
    df["growth_norm"] = df["area_growth_rate_ha_per_h"] / (df["area_first_ha"] + 1.0)

    df["closing_x_inv_dist"] = df["closing_speed_m_per_h"] * df["inv_dist"]
    df["growth_x_inv_dist"] = df["area_growth_rate_ha_per_h"] * df["inv_dist"]
    df["speed_x_inv_dist"] = df["centroid_speed_m_per_h"] * df["inv_dist"]

    # observation-aware interactions
    df["growth_x_dt"] = df["area_growth_rate_ha_per_h"] * df["dt_first_last_0_5h"]
    df["closing_x_dt"] = df["closing_speed_m_per_h"] * df["dt_first_last_0_5h"]
    df["speed_x_dt"] = df["centroid_speed_m_per_h"] * df["dt_first_last_0_5h"]

    df["growth_x_quality"] = df["area_growth_rate_ha_per_h"] * df["observation_quality"]
    df["closing_x_quality"] = df["closing_speed_m_per_h"] * df["observation_quality"]
    df["speed_x_quality"] = df["centroid_speed_m_per_h"] * df["observation_quality"]

    # -------------------------------------------------
    # 4) extra stable summary features
    # -------------------------------------------------
    df["perimeter_density"] = df["num_perimeters_0_5h"] / (df["dt_first_last_0_5h"] + 0.5)
    df["spread_potential"] = df["area_growth_rate_ha_per_h"] * np.maximum(df["alignment_cos"], 0.0)
    df["relative_radial_growth"] = df["radial_growth_rate_m_per_h"] / np.sqrt(df["area_first_ha"] + 1.0)

    # -------------------------------------------------
    # 5) cyclical temporal encoding
    # -------------------------------------------------
    df["hour_sin"] = np.sin(2 * np.pi * df["event_start_hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["event_start_hour"] / 24.0)

    df["month_sin"] = np.sin(2 * np.pi * (df["event_start_month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["event_start_month"] - 1) / 12.0)

    # -------------------------------------------------
    # 6) masked dynamic features for low-quality observations
    #    for tree models, NaN is often better than fake average
    # -------------------------------------------------
    lowq = df["is_low_quality_obs"] == 1

    dynamic_cols = [
        "area_growth_abs_0_5h",
        "area_growth_rate_ha_per_h",
        "radial_growth_m",
        "radial_growth_rate_m_per_h",
        "area_growth_rel_0_5h",
        "log1p_growth",
        "log_area_ratio_0_5h",
        "centroid_speed_m_per_h",
        "closing_speed_m_per_h",
        "dist_slope_ci_0_5h",
        "dist_change_ci_0_5h",
        "along_track_speed",
        "directional_risk",
        "movement_strength",
        "growth_norm",
        "growth_over_dist",
        "closing_ratio",
        "closing_x_inv_dist",
        "growth_x_inv_dist",
        "speed_x_inv_dist",
        "growth_x_dt",
        "closing_x_dt",
        "speed_x_dt",
        "growth_x_quality",
        "closing_x_quality",
        "speed_x_quality",
        "spread_potential",
        "relative_radial_growth",
    ]

    for col in dynamic_cols:
        masked_col = f"{col}_masked"
        df[masked_col] = df[col]
        df.loc[lowq, masked_col] = np.nan

    return df

def add_features_v3(df):
    df = df.copy()

    # ----------------------------
    # 1) 基本旗標
    # ----------------------------
    df["is_single_obs"] = (df["num_perimeters_0_5h"] == 1).astype(int)
    df["has_dynamic"] = (df["num_perimeters_0_5h"] > 1).astype(int)

    # 低資訊觀測
    df["is_low_info_obs"] = (
        (df["num_perimeters_0_5h"] == 1) |
        (df["dt_first_last_0_5h"] < 0.5)
    ).astype(int)

    # ----------------------------
    # 2) 單次觀測也可靠的特徵
    # ----------------------------
    df["log_dist"] = np.log1p(df["dist_min_ci_0_5h"])
    df["inv_dist"] = 1.0 / (df["dist_min_ci_0_5h"] + 1.0)
    df["dist_km"] = df["dist_min_ci_0_5h"] / 1000.0

    # ----------------------------
    # 3) dynamic feature gating
    #    不是直接相信原本的 0，而是乘上 has_dynamic
    # ----------------------------
    dynamic_cols = [
        "area_growth_abs_0_5h",
        "area_growth_rate_ha_per_h",
        "radial_growth_m",
        "radial_growth_rate_m_per_h",
        "area_growth_rel_0_5h",
        "log1p_growth",
        "log_area_ratio_0_5h",
        "centroid_speed_m_per_h",
        "closing_speed_m_per_h",
        "dist_slope_ci_0_5h",
        "dist_change_ci_0_5h",
        "along_track_speed",
    ]

    for col in dynamic_cols:
        df[f"{col}_gated"] = df[col] * df["has_dynamic"]

    # ----------------------------
    # 4) gated interaction features
    # ----------------------------
    df["closing_ratio_gated"] = (
        df["closing_speed_m_per_h_gated"] / (df["dist_min_ci_0_5h"] + 1e-6)
    )
    df["growth_over_dist_gated"] = (
        df["area_growth_rate_ha_per_h_gated"] / (df["dist_min_ci_0_5h"] + 1.0)
    )
    df["directional_risk_gated"] = (
        df["alignment_cos"] * df["closing_speed_m_per_h_gated"]
    )
    df["growth_norm_gated"] = (
        df["area_growth_rate_ha_per_h_gated"] / (df["area_first_ha"] + 1.0)
    )
    df["movement_strength_gated"] = (
        np.abs(df["alignment_cos"]) * df["centroid_speed_m_per_h_gated"]
    )

    df["closing_x_inv_dist_gated"] = (
        df["closing_speed_m_per_h_gated"] * df["inv_dist"]
    )
    df["growth_x_inv_dist_gated"] = (
        df["area_growth_rate_ha_per_h_gated"] * df["inv_dist"]
    )
    df["speed_x_inv_dist_gated"] = (
        df["centroid_speed_m_per_h_gated"] * df["inv_dist"]
    )

    # ----------------------------
    # 5) 單次觀測 fallback 特徵
    # ----------------------------
    df["single_obs_area_risk"] = df["is_single_obs"] * df["log1p_area_first"]
    df["single_obs_dist_risk"] = df["is_single_obs"] * df["inv_dist"]
    df["single_obs_alignment"] = df["is_single_obs"] * df["alignment_cos"]

    return df