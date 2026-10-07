import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import PROCESSED_DATA_DIR
from resume_parser import extract_resume_text


MODEL_NAME = "all-MiniLM-L6-v2"


def load_jobs():
    """Load the processed job dataset."""

    jobs_path = PROCESSED_DATA_DIR / "jobs.csv"

    if not jobs_path.exists():
        raise FileNotFoundError(
            f"Job dataset not found at: {jobs_path}"
        )

    return pd.read_csv(jobs_path)


def build_job_text(job):
    """
    Combine the main information from a job
    into one text representation.
    """

    description = str(job["description"])
    required_skills = str(job["required_skills"])
    preferred_skills = str(job["preferred_skills"])

    return (
        f"Job Description: {description}. "
        f"Required Skills: {required_skills}. "
        f"Preferred Skills: {preferred_skills}."
    )


def create_model():
    """Load the Sentence Transformer model."""

    print("Loading Transformer model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Transformer model loaded.")

    return model


def calculate_semantic_similarity(
    model,
    resume_text,
    job_texts
):
    """
    Convert resume and jobs into embeddings
    and calculate cosine similarity.
    """

    resume_embedding = model.encode(
        [resume_text],
        normalize_embeddings=True
    )

    job_embeddings = model.encode(
        job_texts,
        normalize_embeddings=True
    )

    similarities = cosine_similarity(
        resume_embedding,
        job_embeddings
    )[0]

    return similarities


def rank_jobs(resume_text):
    """
    Rank jobs according to Transformer-based
    semantic similarity.
    """

    jobs_df = load_jobs()

    jobs_df["job_text"] = jobs_df.apply(
        build_job_text,
        axis=1
    )

    model = create_model()

    similarities = calculate_semantic_similarity(
        model,
        resume_text,
        jobs_df["job_text"].tolist()
    )

    jobs_df["semantic_score"] = similarities * 100

    ranked_jobs = jobs_df.sort_values(
        by="semantic_score",
        ascending=False
    ).reset_index(drop=True)

    return ranked_jobs


if __name__ == "__main__":

    resume_path = "data/raw/test_resume.docx"

    print("\nReading resume...")

    resume_text = extract_resume_text(resume_path)

    print("Resume successfully loaded.")

    results = rank_jobs(resume_text)

    print("\n" + "=" * 70)
    print("TRANSFORMER — SEMANTIC JOB MATCHING")
    print("=" * 70)

    for index, row in results.iterrows():

        print(
            f"\n{index + 1}. "
            f"{row['title']} — {row['company']}"
        )

        print(
            f"   Semantic Score: "
            f"{row['semantic_score']:.2f}%"
        )

    print("\n" + "=" * 70)