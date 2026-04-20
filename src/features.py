def get_GBM_feature_columns():
    return [
        "num_perimeters_0_5h",
        "dt_first_last_0_5h",
        "area_first_ha",
        "area_growth_abs_0_5h",
        "area_growth_rate_ha_per_h",
        "radial_growth_m",
        "radial_growth_rate_m_per_h",
        "area_growth_rel_0_5h",
        "log1p_area_first",
        "log1p_growth",
        "log_area_ratio_0_5h",
        "centroid_speed_m_per_h",
        "dist_min_ci_0_5h",
        "closing_speed_m_per_h",
        "dist_slope_ci_0_5h",
        "dist_change_ci_0_5h",
        "dist_std_ci_0_5h",
        "dist_fit_r2_0_5h",
        "alignment_cos",
        "along_track_speed",
        "event_start_hour",
        "event_start_dayofweek",
        "event_start_month",

        "growth_dist_interaction",
        "speed_alignment",
        "closing_ratio",
        "growth_acceleration",
        "log_dist",
        "inv_dist",
        "growth_over_dist",
        "hour_bucket",
        "movement_consistency",
        "area_ratio",
        "directional_risk",
        "growth_norm",
        "movement_strength",
    ]

def get_NN_feature_columns():
    return [
        "num_perimeters_0_5h",
        "dt_first_last_0_5h",
        "log1p_area_first",
        "log1p_growth",
        "log_area_ratio_0_5h",
        "area_growth_rate_ha_per_h",
        "radial_growth_rate_m_per_h",
        "area_growth_rel_0_5h",
        "centroid_speed_m_per_h",
        "dist_min_ci_0_5h",
        "closing_speed_m_per_h",
        "dist_slope_ci_0_5h",
        "dist_std_ci_0_5h",
        "alignment_cos",
        "along_track_speed",
        "event_start_hour",
        "event_start_dayofweek",
        "event_start_month",

        "growth_dist_interaction",
        "speed_alignment",
        "closing_ratio",
        "growth_acceleration",
        "log_dist",
        "inv_dist",
        "growth_over_dist",
        "hour_bucket",
        "movement_consistency",
        "area_ratio",
        "directional_risk",
        "growth_norm",
        "movement_strength",
    ]

def get_GBM_feature_columns_v2():
    return [
        # ---------------------------------------------
        # base coverage
        # ---------------------------------------------
        "num_perimeters_0_5h",
        "dt_first_last_0_5h",

        # ---------------------------------------------
        # original static / semi-static features
        # ---------------------------------------------
        "area_first_ha",
        "dist_min_ci_0_5h",
        "dist_std_ci_0_5h",
        "dist_fit_r2_0_5h",
        "alignment_cos",
        "event_start_hour",
        "event_start_dayofweek",
        "event_start_month",

        # ---------------------------------------------
        # observation quality
        # ---------------------------------------------
        "is_single_perimeter",
        "is_low_dt",
        "is_low_quality_obs",
        "observation_quality",
        "obs_density",
        "obs_strength",
        "perimeter_density",

        # ---------------------------------------------
        # safe transforms
        # ---------------------------------------------
        "log1p_area_first",
        "log_dist",
        "inv_dist",
        "dist_km",

        # ---------------------------------------------
        # temporal cyclic encoding
        # ---------------------------------------------
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",

        # ---------------------------------------------
        # masked dynamic features
        # ---------------------------------------------
        "area_growth_abs_0_5h_masked",
        "area_growth_rate_ha_per_h_masked",
        "radial_growth_m_masked",
        "radial_growth_rate_m_per_h_masked",
        "area_growth_rel_0_5h_masked",
        "log1p_growth_masked",
        "log_area_ratio_0_5h_masked",
        "centroid_speed_m_per_h_masked",
        "closing_speed_m_per_h_masked",
        "dist_slope_ci_0_5h_masked",
        "dist_change_ci_0_5h_masked",
        "along_track_speed_masked",

        # ---------------------------------------------
        # masked engineered interactions
        # ---------------------------------------------
        "directional_risk_masked",
        "movement_strength_masked",
        "growth_norm_masked",
        "growth_over_dist_masked",
        "closing_ratio_masked",
        "closing_x_inv_dist_masked",
        "growth_x_inv_dist_masked",
        "speed_x_inv_dist_masked",
        "growth_x_dt_masked",
        "closing_x_dt_masked",
        "speed_x_dt_masked",
        "growth_x_quality_masked",
        "closing_x_quality_masked",
        "speed_x_quality_masked",
        "spread_potential_masked",
        "relative_radial_growth_masked",
    ]

def get_NN_feature_columns_v2():
    return [
        # base coverage
        "num_perimeters_0_5h",
        "dt_first_last_0_5h",

        # static / low-risk features
        "log1p_area_first",
        "dist_min_ci_0_5h",
        "dist_std_ci_0_5h",
        "dist_fit_r2_0_5h",
        "alignment_cos",
        "event_start_hour",
        "event_start_dayofweek",
        "event_start_month",

        # observation quality
        "is_single_perimeter",
        "is_low_dt",
        "is_low_quality_obs",
        "observation_quality",
        "obs_density",
        "obs_strength",
        "perimeter_density",

        # transforms
        "log_dist",
        "inv_dist",
        "dist_km",

        # original dynamic features
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

        # engineered features
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

        # cyclic
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
    ]

def get_GBM_feature_columns_v3():
    return [
        # ---------------------------------
        # base / reliable for all rows
        # ---------------------------------
        "num_perimeters_0_5h",
        "dt_first_last_0_5h",
        "is_single_obs",
        "has_dynamic",
        "is_low_info_obs",

        "area_first_ha",
        "log1p_area_first",
        "dist_min_ci_0_5h",
        "dist_std_ci_0_5h",
        "dist_fit_r2_0_5h",
        "alignment_cos",
        "event_start_hour",
        "event_start_dayofweek",
        "event_start_month",

        "log_dist",
        "inv_dist",
        "dist_km",

        # ---------------------------------
        # gated dynamic features
        # ---------------------------------
        "area_growth_abs_0_5h_gated",
        "area_growth_rate_ha_per_h_gated",
        "radial_growth_m_gated",
        "radial_growth_rate_m_per_h_gated",
        "area_growth_rel_0_5h_gated",
        "log1p_growth_gated",
        "log_area_ratio_0_5h_gated",
        "centroid_speed_m_per_h_gated",
        "closing_speed_m_per_h_gated",
        "dist_slope_ci_0_5h_gated",
        "dist_change_ci_0_5h_gated",
        "along_track_speed_gated",

        # ---------------------------------
        # gated engineered features
        # ---------------------------------
        "closing_ratio_gated",
        "growth_over_dist_gated",
        "directional_risk_gated",
        "growth_norm_gated",
        "movement_strength_gated",
        "closing_x_inv_dist_gated",
        "growth_x_inv_dist_gated",
        "speed_x_inv_dist_gated",

        # ---------------------------------
        # single-observation fallback
        # ---------------------------------
        "single_obs_area_risk",
        "single_obs_dist_risk",
        "single_obs_alignment",
    ]

def get_NN_feature_columns_v3():
    return [
        "num_perimeters_0_5h",
        "dt_first_last_0_5h",
        "is_single_obs",
        "has_dynamic",
        "is_low_info_obs",

        "log1p_area_first",
        "dist_min_ci_0_5h",
        "dist_std_ci_0_5h",
        "dist_fit_r2_0_5h",
        "alignment_cos",
        "event_start_hour",
        "event_start_dayofweek",
        "event_start_month",

        "log_dist",
        "inv_dist",
        "dist_km",

        "area_growth_abs_0_5h_gated",
        "area_growth_rate_ha_per_h_gated",
        "radial_growth_m_gated",
        "radial_growth_rate_m_per_h_gated",
        "area_growth_rel_0_5h_gated",
        "log1p_growth_gated",
        "log_area_ratio_0_5h_gated",
        "centroid_speed_m_per_h_gated",
        "closing_speed_m_per_h_gated",
        "dist_slope_ci_0_5h_gated",
        "dist_change_ci_0_5h_gated",
        "along_track_speed_gated",

        "closing_ratio_gated",
        "growth_over_dist_gated",
        "directional_risk_gated",
        "growth_norm_gated",
        "movement_strength_gated",
        "closing_x_inv_dist_gated",
        "growth_x_inv_dist_gated",
        "speed_x_inv_dist_gated",

        "single_obs_area_risk",
        "single_obs_dist_risk",
        "single_obs_alignment",
    ]