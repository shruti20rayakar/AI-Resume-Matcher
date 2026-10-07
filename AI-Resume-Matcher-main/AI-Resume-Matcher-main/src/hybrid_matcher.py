import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from config import PROCESSED_DATA_DIR
from resume_parser import extract_resume_text
from resume_analyzer import analyze_resume
from experience_education_matcher import (
    calculate_experience_relevance,
    calculate_education_relevance,
)



MODEL_NAME = "all-MiniLM-L6-v2"

TFIDF_WEIGHT = 0.15
SKILL_WEIGHT = 0.35
SEMANTIC_WEIGHT = 0.30
EXPERIENCE_WEIGHT = 0.10
EDUCATION_WEIGHT = 0.10


_model = None


def get_model():
    """
    Load the Transformer model only once.

    After the first call, the already-loaded model
    is reused.
    """

    global _model

    if _model is None:

        print("Loading Transformer model...")

        _model = SentenceTransformer(
            MODEL_NAME
        )

        print("Transformer model loaded.")

    return _model


def load_jobs():
    """Load the job dataset."""

    jobs_path = (
        PROCESSED_DATA_DIR
        / "jobs.csv"
    )

    if not jobs_path.exists():
        raise FileNotFoundError(
            f"Job dataset not found at: {jobs_path}"
        )

    return pd.read_csv(
        jobs_path
    )


def build_job_text(job):
    """Combine important job information."""

    return (
        f"Job Title: {job['title']}. "
        f"Job Description: {job['description']}. "
        f"Required Skills: {job['required_skills']}. "
        f"Preferred Skills: {job['preferred_skills']}."
    )


def calculate_tfidf_scores(
    resume_text,
    job_texts
):
    """Calculate TF-IDF cosine similarity."""

    documents = [
        resume_text
    ] + job_texts

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(
        documents
    )

    resume_vector = matrix[0]

    job_vectors = matrix[1:]

    scores = cosine_similarity(
        resume_vector,
        job_vectors
    )[0]

    return scores * 100


def parse_skills(skill_text):
    """Convert comma-separated skills into a set."""

    if not skill_text:
        return set()

    return {
        skill.strip().lower()
        for skill in str(
            skill_text
        ).split(",")
        if skill.strip()
    }


def calculate_skill_score(
    candidate_skills,
    job
):
    """Calculate skill compatibility."""

    candidate_skills = {
        skill.lower().strip()
        for skill in candidate_skills
    }

    required_skills = parse_skills(
        job["required_skills"]
    )

    preferred_skills = parse_skills(
        job["preferred_skills"]
    )

    matched_required = (
        candidate_skills
        & required_skills
    )

    missing_required = (
        required_skills
        - candidate_skills
    )

    matched_preferred = (
        candidate_skills
        & preferred_skills
    )

    missing_preferred = (
        preferred_skills
        - candidate_skills
    )

    if required_skills:

        required_score = (
            len(matched_required)
            / len(required_skills)
        ) * 100

    else:

        required_score = 0

    if preferred_skills:

        preferred_score = (
            len(matched_preferred)
            / len(preferred_skills)
        ) * 100

    else:

        preferred_score = 0

    skill_score = (
        required_score * 0.75
        + preferred_score * 0.25
    )

    return {
        "skill_score": skill_score,
        "required_score": required_score,
        "preferred_score": preferred_score,
        "matched_required":
            sorted(matched_required),
        "missing_required":
            sorted(missing_required),
        "matched_preferred":
            sorted(matched_preferred),
        "missing_preferred":
            sorted(missing_preferred),
    }


def calculate_semantic_scores(
    resume_text,
    job_texts
):
    """
    Calculate semantic similarity using the
    cached Transformer model.
    """

    model = get_model()

    resume_embedding = model.encode(
        [resume_text],
        normalize_embeddings=True
    )

    job_embeddings = model.encode(
        job_texts,
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        resume_embedding,
        job_embeddings
    )[0]

    return scores * 100


def generate_hybrid_ranking(
    resume_path
):
    """
    Generate final job ranking using:

    TF-IDF
    Skills
    Transformer semantics
    Experience
    Education
    """

    print("\nReading resume...")

    resume_text = extract_resume_text(
        resume_path
    )

    print("Resume loaded.")

    resume_data = analyze_resume(
        resume_path
    )

    candidate_skills = set(
        resume_data["skills"]
    )

    jobs_df = load_jobs()

    jobs_df["job_text"] = jobs_df.apply(
        build_job_text,
        axis=1
    )

    job_texts = jobs_df[
        "job_text"
    ].tolist()

    
    print(
        "Calculating TF-IDF scores..."
    )

    tfidf_scores = (
        calculate_tfidf_scores(
            resume_text,
            job_texts
        )
    )

    print(
        "Calculating Transformer scores..."
    )

    semantic_scores = (
        calculate_semantic_scores(
            resume_text,
            job_texts
        )
    )

    skill_results = []

    experience_scores = []

    education_scores = []

    for _, job in jobs_df.iterrows():

        skill_result = (
            calculate_skill_score(
                candidate_skills,
                job
            )
        )

        skill_results.append(
            skill_result
        )

        experience_score = (
            calculate_experience_relevance(
                resume_data["experience"],
                job["description"]
            )
        )

        education_score = (
            calculate_education_relevance(
                resume_data["education"],
                job["description"]
            )
        )

        experience_scores.append(
            experience_score
        )

        education_scores.append(
            education_score
        )

    jobs_df["tfidf_score"] = (
        tfidf_scores
    )

    jobs_df["semantic_score"] = (
        semantic_scores
    )

    jobs_df["skill_score"] = [
        result["skill_score"]
        for result in skill_results
    ]

    jobs_df["experience_score"] = (
        experience_scores
    )

    jobs_df["education_score"] = (
        education_scores
    )

    jobs_df["skill_details"] = (
        skill_results
    )

    jobs_df["final_score"] = (

        jobs_df["tfidf_score"]
        * TFIDF_WEIGHT

        + jobs_df["skill_score"]
        * SKILL_WEIGHT

        + jobs_df["semantic_score"]
        * SEMANTIC_WEIGHT

        + jobs_df["experience_score"]
        * EXPERIENCE_WEIGHT

        + jobs_df["education_score"]
        * EDUCATION_WEIGHT
    )

    jobs_df = jobs_df.sort_values(
        by="final_score",
        ascending=False
    ).reset_index(
        drop=True
    )

    return (
        jobs_df,
        resume_data
    )


if __name__ == "__main__":

    resume_path = (
        "data/raw/test_resume.docx"
    )

    results, resume_data = (
        generate_hybrid_ranking(
            resume_path
        )
    )

    print("\n")
    print("=" * 80)
    print(
        "AI RESUME-JOB HYBRID MATCHING"
    )
    print("=" * 80)

    print(
        f"\nCandidate: "
        f"{resume_data['name']}"
    )

    print("\nJOB RANKING")
    print("-" * 80)

    for index, row in (
        results.iterrows()
    ):

        details = (
            row["skill_details"]
        )

        print(
            f"\n{index + 1}. "
            f"{row['title']} — "
            f"{row['company']}"
        )

        print(
            f"   FINAL MATCH: "
            f"{row['final_score']:.2f}%"
        )

        print(
            f"   TF-IDF: "
            f"{row['tfidf_score']:.2f}%"
        )

        print(
            f"   Skills: "
            f"{row['skill_score']:.2f}%"
        )

        print(
            f"   Transformer: "
            f"{row['semantic_score']:.2f}%"
        )

        print(
            f"   Experience: "
            f"{row['experience_score']:.2f}%"
        )

        print(
            f"   Education: "
            f"{row['education_score']:.2f}%"
        )

        print(
            "   Missing Required: "
            + (
                ", ".join(
                    details[
                        "missing_required"
                    ]
                )
                if details[
                    "missing_required"
                ]
                else "None"
            )
        )

    print("\n")
    print("=" * 80)
    print("DONE")
    print("=" * 80)