import pandas as pd

from config import PROCESSED_DATA_DIR
from resume_analyzer import analyze_resume


def parse_skills(skill_text):
    """Convert comma-separated skills into a normalized set."""

    if not skill_text:
        return set()

    return {
        skill.strip().lower()
        for skill in str(skill_text).split(",")
        if skill.strip()
    }


def analyze_skill_gap(candidate_skills, job):
    """
    Compare candidate skills with the skills
    required by a particular job.
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

    matched_required = candidate_skills & required_skills
    missing_required = required_skills - candidate_skills

    matched_preferred = candidate_skills & preferred_skills
    missing_preferred = preferred_skills - candidate_skills

    if required_skills:
        required_match_percentage = (
            len(matched_required)
            / len(required_skills)
        ) * 100
    else:
        required_match_percentage = 0

    if preferred_skills:
        preferred_match_percentage = (
            len(matched_preferred)
            / len(preferred_skills)
        ) * 100
    else:
        preferred_match_percentage = 0

    return {
        "matched_required": sorted(matched_required),
        "missing_required": sorted(missing_required),
        "matched_preferred": sorted(matched_preferred),
        "missing_preferred": sorted(missing_preferred),
        "required_match_percentage": required_match_percentage,
        "preferred_match_percentage": preferred_match_percentage,
    }


def analyze_all_jobs(resume_path):
    """Analyze skill gaps against every job."""

    resume_data = analyze_resume(resume_path)

    candidate_skills = set(resume_data["skills"])

    jobs_path = PROCESSED_DATA_DIR / "jobs.csv"

    jobs_df = pd.read_csv(jobs_path)

    results = []

    for _, job in jobs_df.iterrows():

        gap = analyze_skill_gap(
            candidate_skills,
            job
        )

        results.append({
            "job_id": job["job_id"],
            "title": job["title"],
            "company": job["company"],
            **gap,
        })

    return results, resume_data


if __name__ == "__main__":

    resume_path = "data/raw/test_resume.docx"

    results, resume_data = analyze_all_jobs(
        resume_path
    )

    print("\n" + "=" * 75)
    print("SKILL GAP ANALYSIS")
    print("=" * 75)

    print(f"\nCandidate: {resume_data['name']}")

    print(
        "\nCandidate Skills:\n  "
        + ", ".join(resume_data["skills"])
    )

    for result in results:

        print("\n" + "-" * 75)

        print(
            f"{result['title']} — "
            f"{result['company']}"
        )

        print(
            f"\nRequired Skill Match: "
            f"{result['required_match_percentage']:.2f}%"
        )

        print(
            "Matched Required Skills: "
            + (
                ", ".join(result["matched_required"])
                if result["matched_required"]
                else "None"
            )
        )

        print(
            "Missing Required Skills: "
            + (
                ", ".join(result["missing_required"])
                if result["missing_required"]
                else "None"
            )
        )

        print(
            f"\nPreferred Skill Match: "
            f"{result['preferred_match_percentage']:.2f}%"
        )

        print(
            "Matched Preferred Skills: "
            + (
                ", ".join(result["matched_preferred"])
                if result["matched_preferred"]
                else "None"
            )
        )

    print("\n" + "=" * 75)