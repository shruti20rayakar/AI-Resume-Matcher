import re

from resume_analyzer import analyze_resume


def normalize(text):
    """Convert text to lowercase and normalize whitespace."""

    if not text:
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def calculate_experience_relevance(resume_experience, job_description):
    """
    Estimate how relevant the candidate's experience is
    to the job description using keyword overlap.
    """

    experience = normalize(resume_experience)
    job = normalize(job_description)

    if not experience or not job:
        return 0.0

    # Important technical/job-related terms.
    keywords = [
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "python",
        "sql",
        "data analysis",
        "data science",
        "natural language processing",
        "nlp",
        "transformers",
        "computer vision",
        "neural networks",
        "tensorflow",
        "pytorch",
        "scikit-learn",
        "opencv",
        "statistics",
        "model training",
        "model evaluation",
        "data preprocessing",
    ]

    matched_keywords = [
        keyword
        for keyword in keywords
        if keyword in experience and keyword in job
    ]

    job_keywords = [
        keyword
        for keyword in keywords
        if keyword in job
    ]

    if not job_keywords:
        return 0.0

    score = (
        len(matched_keywords)
        / len(job_keywords)
    ) * 100

    return min(score, 100.0)


def calculate_education_relevance(resume_education, job_description):
    """
    Estimate whether the candidate's education contains
    relevant technical background.
    """

    education = normalize(resume_education)
    job = normalize(job_description)

    if not education:
        return 0.0

    technical_terms = [
        "computer science",
        "information technology",
        "artificial intelligence",
        "machine learning",
        "data science",
        "software engineering",
        "electronics",
        "mathematics",
        "statistics",
    ]

    relevant_terms = [
        term
        for term in technical_terms
        if term in education and term in job
    ]

    # If the job doesn't explicitly mention education fields,
    # give a reasonable baseline for a technical degree.
    if not any(term in job for term in technical_terms):
        if any(term in education for term in technical_terms):
            return 70.0

        return 0.0

    job_terms = [
        term
        for term in technical_terms
        if term in job
    ]

    if not job_terms:
        return 0.0

    score = (
        len(relevant_terms)
        / len(job_terms)
    ) * 100

    return min(score, 100.0)


def analyze_background(resume_path, job):
    """
    Calculate experience and education relevance
    for one job.
    """

    resume_data = analyze_resume(resume_path)

    experience_score = calculate_experience_relevance(
        resume_data["experience"],
        job["description"]
    )

    education_score = calculate_education_relevance(
        resume_data["education"],
        job["description"]
    )

    return {
        "experience_score": experience_score,
        "education_score": education_score,
    }


if __name__ == "__main__":

    import pandas as pd

    from config import PROCESSED_DATA_DIR

    resume_path = "data/raw/test_resume.docx"

    resume_data = analyze_resume(resume_path)

    jobs_path = PROCESSED_DATA_DIR / "jobs.csv"

    jobs_df = pd.read_csv(jobs_path)

    print("\n" + "=" * 75)
    print("EXPERIENCE & EDUCATION RELEVANCE")
    print("=" * 75)

    for _, job in jobs_df.iterrows():

        result = analyze_background(
            resume_path,
            job
        )

        print(
            f"\n{job['title']} — {job['company']}"
        )

        print(
            f"Experience Relevance: "
            f"{result['experience_score']:.2f}%"
        )

        print(
            f"Education Relevance: "
            f"{result['education_score']:.2f}%"
        )

    print("\n" + "=" * 75)