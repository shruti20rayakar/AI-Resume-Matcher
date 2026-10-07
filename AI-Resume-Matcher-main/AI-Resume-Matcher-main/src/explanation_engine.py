import pandas as pd

from pathlib import Path

from hybrid_matcher import generate_hybrid_ranking


STRONG_MATCH_THRESHOLD = 75
GOOD_MATCH_THRESHOLD = 60
MODERATE_MATCH_THRESHOLD = 45


def get_match_verdict(score):
    """Convert final score into a human-readable verdict."""

    if score >= STRONG_MATCH_THRESHOLD:
        return "Strong Match"

    if score >= GOOD_MATCH_THRESHOLD:
        return "Good Match"

    if score >= MODERATE_MATCH_THRESHOLD:
        return "Moderate Match"

    return "Weak Match"



def generate_skill_explanation(skill_details):
    """Generate explanations based on skill compatibility."""

    matched_required = skill_details[
        "matched_required"
    ]

    missing_required = skill_details[
        "missing_required"
    ]

    matched_preferred = skill_details[
        "matched_preferred"
    ]

    explanations = []

    if matched_required:

        explanations.append(
            f"Strong skill alignment: "
            f"{len(matched_required)} required skill(s) matched."
        )

    if missing_required:

        explanations.append(
            f"{len(missing_required)} required skill(s) "
            f"are missing."
        )

    if matched_preferred:

        explanations.append(
            f"{len(matched_preferred)} preferred skill(s) "
            f"also matched."
        )

    return explanations



def generate_score_explanation(row):
    """Explain the contribution of different scoring signals."""

    explanations = []

    semantic_score = row[
        "semantic_score"
    ]

    skill_score = row[
        "skill_score"
    ]

    experience_score = row[
        "experience_score"
    ]

    education_score = row[
        "education_score"
    ]

    tfidf_score = row[
        "tfidf_score"
    ]

    if semantic_score >= 70:

        explanations.append(
            "High semantic similarity between the resume "
            "and job description."
        )

    elif semantic_score >= 50:

        explanations.append(
            "Moderate semantic similarity between the resume "
            "and job description."
        )

    else:

        explanations.append(
            "Low semantic similarity with the job description."
        )

    if skill_score >= 80:

        explanations.append(
            "Very strong alignment with the required and "
            "preferred skills."
        )

    elif skill_score >= 50:

        explanations.append(
            "Reasonable alignment with the job's skills."
        )

    else:

        explanations.append(
            "Limited alignment with the required skills."
        )

    if experience_score >= 80:

        explanations.append(
            "The candidate's experience is highly relevant "
            "to this role."
        )

    elif experience_score >= 50:

        explanations.append(
            "The candidate has moderately relevant experience."
        )

    else:

        explanations.append(
            "The candidate has limited directly relevant experience."
        )

    if education_score >= 80:

        explanations.append(
            "Educational background strongly aligns with "
            "the role."
        )

    elif education_score >= 50:

        explanations.append(
            "Educational background has reasonable relevance."
        )

    else:

        explanations.append(
            "Educational relevance to the role is limited."
        )

    if tfidf_score >= 50:

        explanations.append(
            "The resume contains substantial textual overlap "
            "with the job description."
        )

    return explanations



def generate_explanation(row):
    """
    Generate a complete human-readable explanation
    for one job recommendation.
    """

    score = row[
        "final_score"
    ]

    skill_details = row[
        "skill_details"
    ]

    explanation = {

        "verdict":
            get_match_verdict(
                score
            ),

        "score":
            round(
                score,
                2
            ),

        "score_explanation":
            generate_score_explanation(
                row
            ),

        "skill_explanation":
            generate_skill_explanation(
                skill_details
            ),

        "matched_required":
            skill_details[
                "matched_required"
            ],

        "missing_required":
            skill_details[
                "missing_required"
            ],

        "matched_preferred":
            skill_details[
                "matched_preferred"
            ],
    }

    return explanation



def display_job_explanation(
    row,
    rank
):
    """Display a detailed explanation for a job."""

    explanation = generate_explanation(
        row
    )

    print("\n")
    print("=" * 80)

    print(
        f"{rank}. "
        f"{row['title']} — "
        f"{row['company']}"
    )

    print("=" * 80)

    print(
        f"\nOverall Match: "
        f"{explanation['score']:.2f}%"
    )

    print(
        f"Verdict: "
        f"{explanation['verdict']}"
    )

    

    print("\nSCORE BREAKDOWN")
    print("-" * 50)

    print(
        f"TF-IDF       : "
        f"{row['tfidf_score']:.2f}%"
    )

    print(
        f"Skills       : "
        f"{row['skill_score']:.2f}%"
    )

    print(
        f"Transformer  : "
        f"{row['semantic_score']:.2f}%"
    )

    print(
        f"Experience   : "
        f"{row['experience_score']:.2f}%"
    )

    print(
        f"Education    : "
        f"{row['education_score']:.2f}%"
    )

    

    print("\nWHY THIS MATCH?")
    print("-" * 50)

    for reason in (
        explanation[
            "score_explanation"
        ]
    ):

        print(
            f"✓ {reason}"
        )

    
    print("\nMATCHED REQUIRED SKILLS")
    print("-" * 50)

    if explanation[
        "matched_required"
    ]:

        for skill in (
            explanation[
                "matched_required"
            ]
        ):

            print(
                f"✓ {skill}"
            )

    else:

        print("None")

    

    print("\nMISSING REQUIRED SKILLS")
    print("-" * 50)

    if explanation[
        "missing_required"
    ]:

        for skill in (
            explanation[
                "missing_required"
            ]
        ):

            print(
                f"⚠ {skill}"
            )

    else:

        print(
            "None — all required skills matched."
        )

    

    print("\nMATCHED PREFERRED SKILLS")
    print("-" * 50)

    if explanation[
        "matched_preferred"
    ]:

        for skill in (
            explanation[
                "matched_preferred"
            ]
        ):

            print(
                f"✓ {skill}"
            )

    else:

        print("None")




if __name__ == "__main__":

    resume_path = (
        "data/raw/test_resume.docx"
    )

    print("\n")
    print("=" * 80)
    print(
        "EXPLAINABLE AI RESUME-JOB MATCHING"
    )
    print("=" * 80)

    results, resume_data = (
        generate_hybrid_ranking(
            resume_path
        )
    )

    print(
        f"\nCandidate: "
        f"{resume_data['name']}"
    )

    # Show top 5 recommendations
    top_jobs = results.head(5)

    for rank, (_, row) in enumerate(
        top_jobs.iterrows(),
        start=1
    ):

        display_job_explanation(
            row,
            rank
        )

    print("\n")
    print("=" * 80)
    print(
        "EXPLANATION GENERATION COMPLETE"
    )
    print("=" * 80)