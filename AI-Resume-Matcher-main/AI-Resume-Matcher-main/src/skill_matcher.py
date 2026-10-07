import pandas as pd

from config import PROCESSED_DATA_DIR
from resume_analyzer import analyze_resume


def parse_skills(skill_text):
    """
    Convert a comma-separated skill string into
    a normalized set of skills.
    """

    if not skill_text:
        return set()

    skills = skill_text.split(",")

    return {
        skill.strip().lower()
        for skill in skills
        if skill.strip()
    }


def calculate_skill_match(candidate_skills, job):
    """
    Calculate required and preferred skill matches
    for one job.
    """

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

    matched_required = candidate_skills.intersection(
        required_skills
    )

    missing_required = required_skills.difference(
        candidate_skills
    )

    matched_preferred = candidate_skills.intersection(
        preferred_skills
    )

    missing_preferred = preferred_skills.difference(
        candidate_skills
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

    overall_skill_score = (
        required_score * 0.75
        + preferred_score * 0.25
    )

    return {
        "required_score": required_score,
        "preferred_score": preferred_score,
        "overall_skill_score": overall_skill_score,
        "matched_required": sorted(matched_required),
        "missing_required": sorted(missing_required),
        "matched_preferred": sorted(matched_preferred),
        "missing_preferred": sorted(missing_preferred),
    }


def analyze_all_jobs(resume_path):
    """
    Analyze the candidate's skills against every job.
    """

    resume_data = analyze_resume(resume_path)

    candidate_skills = set(resume_data["skills"])

    jobs_path = PROCESSED_DATA_DIR / "jobs.csv"

    jobs_df = pd.read_csv(jobs_path)

    results = []

    for _, job in jobs_df.iterrows():

        skill_result = calculate_skill_match(
            candidate_skills,
            job
        )

        results.append({
            "job_id": job["job_id"],
            "title": job["title"],
            "company": job["company"],
            **skill_result
        })

    return pd.DataFrame(results)


if __name__ == "__main__":

    resume_path = "data/raw/test_resume.docx"

    results = analyze_all_jobs(resume_path)

    results = results.sort_values(
        by="overall_skill_score",
        ascending=False
    )

    print("\n" + "=" * 70)
    print("SKILL MATCHING ANALYSIS")
    print("=" * 70)

    for _, row in results.iterrows():

        print(
            f"\n{row['title']} — {row['company']}"
        )

        print(
            f"Required Skill Match: "
            f"{row['required_score']:.2f}%"
        )

        print(
            f"Preferred Skill Match: "
            f"{row['preferred_score']:.2f}%"
        )

        print(
            f"Overall Skill Score: "
            f"{row['overall_skill_score']:.2f}%"
        )

        print(
            f"Matched Required: "
            f"{', '.join(row['matched_required']) or 'None'}"
        )

        print(
            f"Missing Required: "
            f"{', '.join(row['missing_required']) or 'None'}"
        )

    print("\n" + "=" * 70)