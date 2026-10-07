import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import PROCESSED_DATA_DIR
from resume_parser import extract_resume_text


def load_jobs():
    """Load job descriptions from the processed dataset."""

    jobs_path = PROCESSED_DATA_DIR / "jobs.csv"

    if not jobs_path.exists():
        raise FileNotFoundError(
            f"Job dataset not found at: {jobs_path}"
        )

    return pd.read_csv(jobs_path)


def build_job_text(job):
    """
    Combine the important parts of a job description.
    """

    description = str(job["description"])
    required_skills = str(job["required_skills"])
    preferred_skills = str(job["preferred_skills"])

    return (
        f"{description} "
        f"{required_skills} "
        f"{required_skills} "
        f"{preferred_skills}"
    )


def calculate_similarity(resume_text, job_texts):
    """
    Calculate TF-IDF cosine similarity between
    the resume and all job descriptions.
    """

    documents = [resume_text] + job_texts

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    resume_vector = tfidf_matrix[0]
    job_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(
        resume_vector,
        job_vectors
    )[0]

    return similarities


def rank_jobs(resume_text):
    """
    Rank all available jobs against a resume.
    """

    jobs_df = load_jobs()

    jobs_df["job_text"] = jobs_df.apply(
        build_job_text,
        axis=1
    )

    similarities = calculate_similarity(
        resume_text,
        jobs_df["job_text"].tolist()
    )

    jobs_df["match_score"] = similarities * 100

    ranked_jobs = jobs_df.sort_values(
        by="match_score",
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
    print("TF-IDF BASELINE — JOB RANKING")
    print("=" * 70)

    for index, row in results.iterrows():

        print(
            f"\n{index + 1}. {row['title']}"
            f" — {row['company']}"
        )

        print(
            f"   Match Score: "
            f"{row['match_score']:.2f}%"
        )

    print("\n" + "=" * 70)