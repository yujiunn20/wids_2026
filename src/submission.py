import pandas as pd


def save_submission(event_ids, pred_array, output_path="data/submissions/submission.csv"):
    submission = pd.DataFrame({
        "event_id": event_ids,
        "prob_12h": pred_array[:, 0],
        "prob_24h": pred_array[:, 1],
        "prob_48h": pred_array[:, 2],
        "prob_72h": pred_array[:, 3],
    })

    submission.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")