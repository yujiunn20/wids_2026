def build_targets(train):
    y_12 = (train["time_to_hit_hours"] <= 12).astype(int)
    y_24 = (train["time_to_hit_hours"] <= 24).astype(int)
    y_48 = (train["time_to_hit_hours"] <= 48).astype(int)
    y_72 = train["event"].astype(int)

    targets = {
        "prob_12h": y_12,
        "prob_24h": y_24,
        "prob_48h": y_48,
        "prob_72h": y_72,
    }

    return targets