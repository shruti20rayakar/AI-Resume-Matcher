import pandas as pd

from pathlib import Path


EVALUATION_DIR = Path("data/evaluation")

LABELS_FILE = (
    EVALUATION_DIR / "labeled_matches.csv"
)

PREDICTIONS_FILE = (
    EVALUATION_DIR / "predictions.csv"
)


def load_data():
    """Load ground-truth labels and model predictions."""

    if not LABELS_FILE.exists():
        raise FileNotFoundError(
            f"Labels file not found: {LABELS_FILE}"
        )

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_FILE}"
        )

    labels = pd.read_csv(
        LABELS_FILE
    )

    predictions = pd.read_csv(
        PREDICTIONS_FILE
    )

    data = labels.merge(
        predictions,
        on=[
            "resume_id",
            "job_id"
        ],
        how="inner"
    )

    if data.empty:
        raise ValueError(
            "No matching resume-job pairs found."
        )

    return data


def precision_at_k(
    ranked_jobs,
    k
):
    """
    Calculate Precision@K.

    Measures how many of the top K recommended
    jobs are actually relevant.
    """

    top_k = ranked_jobs.head(k)

    if len(top_k) == 0:
        return 0.0

    relevant = (
        top_k["relevance_label"]
        .sum()
    )

    return relevant / len(top_k)


def recall_at_k(
    ranked_jobs,
    k
):
    """
    Calculate Recall@K.

    Measures how many relevant jobs were found
    within the top K recommendations.
    """

    total_relevant = (
        ranked_jobs[
            "relevance_label"
        ].sum()
    )

    if total_relevant == 0:
        return 0.0

    top_k = ranked_jobs.head(k)

    relevant_found = (
        top_k[
            "relevance_label"
        ].sum()
    )

    return (
        relevant_found
        / total_relevant
    )


def reciprocal_rank(
    ranked_jobs
):
    """
    Calculate Reciprocal Rank.

    If the first relevant job appears at:
        position 1 -> 1.0
        position 2 -> 0.5
        position 3 -> 0.333
        etc.
    """

    for position, (_, row) in enumerate(
        ranked_jobs.iterrows(),
        start=1
    ):

        if row["relevance_label"] == 1:

            return 1.0 / position

    return 0.0


def evaluate_resume(
    resume_data
):
    """
    Evaluate ranking quality for one resume.
    """

    ranked_jobs = resume_data.sort_values(
        by="predicted_score",
        ascending=False
    ).reset_index(
        drop=True
    )

    p3 = precision_at_k(
        ranked_jobs,
        3
    )

    r3 = recall_at_k(
        ranked_jobs,
        3
    )

    p5 = precision_at_k(
        ranked_jobs,
        5
    )

    r5 = recall_at_k(
        ranked_jobs,
        5
    )

    mrr = reciprocal_rank(
        ranked_jobs
    )

    return {
        "precision_at_3": p3,
        "recall_at_3": r3,
        "precision_at_5": p5,
        "recall_at_5": r5,
        "mrr": mrr,
    }


def run_evaluation():

    data = load_data()

    resume_ids = (
        data["resume_id"]
        .unique()
    )

    results = []

    print("\n")
    print("=" * 75)
    print(
        "AI RESUME-JOB MATCHING — "
        "RANKING EVALUATION"
    )
    print("=" * 75)

    for resume_id in resume_ids:

        resume_data = data[
            data["resume_id"]
            == resume_id
        ]

        metrics = evaluate_resume(
            resume_data
        )

        results.append({
            "resume_id": resume_id,
            **metrics
        })

        print("\n")
        print(
            f"Resume: {resume_id}"
        )

        print(
            f"Precision@3 : "
            f"{metrics['precision_at_3']:.3f}"
        )

        print(
            f"Recall@3    : "
            f"{metrics['recall_at_3']:.3f}"
        )

        print(
            f"Precision@5 : "
            f"{metrics['precision_at_5']:.3f}"
        )

        print(
            f"Recall@5    : "
            f"{metrics['recall_at_5']:.3f}"
        )

        print(
            f"MRR         : "
            f"{metrics['mrr']:.3f}"
        )

    results_df = pd.DataFrame(
        results
    )

    overall = {
        "precision_at_3":
            results_df[
                "precision_at_3"
            ].mean(),

        "recall_at_3":
            results_df[
                "recall_at_3"
            ].mean(),

        "precision_at_5":
            results_df[
                "precision_at_5"
            ].mean(),

        "recall_at_5":
            results_df[
                "recall_at_5"
            ].mean(),

        "mrr":
            results_df[
                "mrr"
            ].mean(),
    }

    print("\n")
    print("=" * 75)
    print(
        "OVERALL RANKING PERFORMANCE"
    )
    print("=" * 75)

    print(
        f"\nPrecision@3 : "
        f"{overall['precision_at_3']:.4f}"
    )

    print(
        f"Recall@3    : "
        f"{overall['recall_at_3']:.4f}"
    )

    print(
        f"Precision@5 : "
        f"{overall['precision_at_5']:.4f}"
    )

    print(
        f"Recall@5    : "
        f"{overall['recall_at_5']:.4f}"
    )

    print(
        f"MRR         : "
        f"{overall['mrr']:.4f}"
    )

    output_file = (
        EVALUATION_DIR
        / "ranking_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\n")
    print(
        f"Detailed results saved to:"
    )

    print(
        output_file
    )

    print("\n")
    print("=" * 75)
    print(
        "RANKING EVALUATION COMPLETE"
    )
    print("=" * 75)


if __name__ == "__main__":
    run_evaluation()