import pandas as pd

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from hybrid_matcher import generate_hybrid_ranking


EVALUATION_DIR = Path("data/evaluation")

LABELS_FILE = (
    EVALUATION_DIR / "labeled_matches.csv"
)

PREDICTIONS_FILE = (
    EVALUATION_DIR / "predictions.csv"
)


def load_labels():
    """Load manually created evaluation labels."""
    if not LABELS_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation labels not found at: {LABELS_FILE}"
        )

    return pd.read_csv(LABELS_FILE)


def generate_predictions():
    """
    Run the hybrid matcher on every evaluation resume.
    The resulting scores are saved so that we don't have
    to run the Transformer again when testing thresholds.
    """
    labels_df = load_labels()

    all_predictions = []

    resume_ids = labels_df["resume_id"].unique()

    for resume_id in resume_ids:

        resume_path = (
            EVALUATION_DIR / f"{resume_id}.docx"
        )

        if not resume_path.exists():
            print(
                f"Warning: {resume_path} not found."
            )
            continue

        print("\n" + "=" * 70)

        print(
            f"Evaluating {resume_id}"
        )

        print("=" * 70)

        results, _ = generate_hybrid_ranking(
            str(resume_path)
        )

        for _, row in results.iterrows():

            all_predictions.append(
                {
                    "resume_id": resume_id,
                    "job_id": row["job_id"],
                    "predicted_score": row["final_score"],
                }
            )

    predictions_df = pd.DataFrame(
        all_predictions
    )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    print(
        f"\nPredictions saved to: "
        f"{PREDICTIONS_FILE}"
    )

    return predictions_df


def get_predictions():
    """
    Use previously saved predictions if available.
    Otherwise run the hybrid matcher.
    """
    if PREDICTIONS_FILE.exists():

        print(
            "\nExisting predictions found."
        )

        print(
            "Loading predictions instead "
            "of running Transformer again..."
        )

        return pd.read_csv(
            PREDICTIONS_FILE
        )

    print(
        "\nNo saved predictions found."
    )

    print(
        "Running hybrid matcher..."
    )

    return generate_predictions()


def prepare_evaluation_data():
    """
    Merge ground-truth labels with model predictions.
    """
    labels_df = load_labels()

    predictions_df = get_predictions()

    evaluation_df = labels_df.merge(
        predictions_df,
        on=[
            "resume_id",
            "job_id"
        ],
        how="inner",
    )

    if evaluation_df.empty:
        raise ValueError(
            "No matching resume/job pairs found "
            "between labels and predictions."
        )

    return evaluation_df


def evaluate_thresholds(
    evaluation_df
):
    """
    Test thresholds from 20% to 50%
    in increments of 2%.
    """
    thresholds = list(
        range(20, 51, 2)
    )

    results = []

    y_true = evaluation_df[
        "relevance_label"
    ]

    for threshold in thresholds:

        y_pred = (
            evaluation_df[
                "predicted_score"
            ]
            >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        results.append(
            {
                "threshold": threshold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return pd.DataFrame(
        results
    )


if __name__ == "__main__":

    print("\n")

    print("=" * 75)

    print(
        "AI RESUME-JOB MATCHING — "
        "THRESHOLD TUNING"
    )

    print("=" * 75)

    evaluation_df = (
        prepare_evaluation_data()
    )

    print(
        f"\nTotal evaluation pairs: "
        f"{len(evaluation_df)}"
    )

    threshold_results = (
        evaluate_thresholds(
            evaluation_df
        )
    )

    print("\n")

    print("=" * 75)

    print("THRESHOLD COMPARISON")

    print("=" * 75)

    print(
        "\nThreshold | Accuracy | Precision | Recall | F1"
    )

    print("-" * 75)

    for _, row in (
        threshold_results.iterrows()
    ):

        print(
            f"{row['threshold']:9.0f} | "
            f"{row['accuracy']:8.3f} | "
            f"{row['precision']:9.3f} | "
            f"{row['recall']:6.3f} | "
            f"{row['f1']:5.3f}"
        )

    best_row = threshold_results.loc[
        threshold_results["f1"].idxmax()
    ]

    best_threshold = (
        best_row["threshold"]
    )

    print("\n")

    print("=" * 75)

    print("BEST THRESHOLD")

    print("=" * 75)

    print(
        f"\nThreshold : "
        f"{best_threshold:.0f}%"
    )

    print(
        f"Accuracy  : "
        f"{best_row['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{best_row['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_row['recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{best_row['f1']:.4f}"
    )

    y_true = evaluation_df[
        "relevance_label"
    ]

    y_pred = (
        evaluation_df[
            "predicted_score"
        ]
        >= best_threshold
    ).astype(int)

    matrix = confusion_matrix(
        y_true,
        y_pred
    )

    print("\n")

    print("=" * 75)

    print("CONFUSION MATRIX")

    print("=" * 75)

    print(
        "\n[[TN FP]"
    )

    print(
        " [FN TP]]"
    )

    print(
        "\n",
        matrix
    )

    print("\n")

    print("=" * 75)

    print("EVALUATION COMPLETE")

    print("=" * 75)