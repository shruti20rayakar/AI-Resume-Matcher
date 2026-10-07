import re
from pathlib import Path

from resume_parser import extract_resume_text


KNOWN_SKILLS = [
    "python",
    "java",
    "c++",
    "sql",
    "numpy",
    "pandas",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "machine learning",
    "deep learning",
    "natural language processing",
    "nlp",
    "transformers",
    "bert",
    "computer vision",
    "opencv",
    "statistics",
    "matplotlib",
    "seaborn",
    "git",
    "docker",
    "aws",
    "azure",
    "flask",
    "fastapi",
]


SECTION_NAMES = {
    "education": ["education", "academic background", "academic qualification"],
    "skills": ["skills", "technical skills", "technical expertise"],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "internship",
    ],
    "projects": ["projects", "academic projects", "personal projects"],
    "certifications": ["certifications", "certificates"],
}


def normalize_text(text: str) -> str:
    """Clean unnecessary whitespace from resume text."""

    text = text.replace("\r", "\n")

    lines = []

    for line in text.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


def extract_name(text: str) -> str:
    """
    Estimate the candidate's name.

    Most resumes place the name near the beginning.
    """

    lines = text.split("\n")

    for line in lines[:8]:
        line = line.strip()

        if not line:
            continue

        
        if line.lower() in {
            "resume",
            "curriculum vitae",
            "cv",
            "education",
            "skills",
            "experience",
        }:
            continue

        
        if "@" in line or re.search(r"\d{5,}", line):
            continue

        
        if len(line.split()) <= 4 and len(line) <= 50:
            return line

    return "Unknown"


def extract_sections(text: str) -> dict:
    """
    Divide the resume into common sections.
    """

    lines = text.split("\n")

    sections = {
        "education": [],
        "skills": [],
        "experience": [],
        "projects": [],
        "certifications": [],
    }

    current_section = None

    for line in lines:
        clean_line = line.strip()

        if not clean_line:
            continue

        lower_line = clean_line.lower()

        detected_section = None

        for section, names in SECTION_NAMES.items():
            if lower_line in names:
                detected_section = section
                break

        if detected_section:
            current_section = detected_section
            continue

        if current_section:
            sections[current_section].append(clean_line)

    return {
        section: "\n".join(content)
        for section, content in sections.items()
    }


def extract_skills(text: str) -> list:
    """
    Extract known technical skills from the resume.
    """

    text_lower = text.lower()

    found_skills = []

    for skill in KNOWN_SKILLS:
       
        pattern = rf"(?<!\w){re.escape(skill)}(?!\w)"

        if re.search(pattern, text_lower):
            found_skills.append(skill)

    return sorted(found_skills)


def analyze_resume(file_path: str) -> dict:
    """
    Complete resume analysis pipeline.
    """

    raw_text = extract_resume_text(file_path)

    cleaned_text = normalize_text(raw_text)

    sections = extract_sections(cleaned_text)

    skills = extract_skills(cleaned_text)

    result = {
        "name": extract_name(cleaned_text),
        "skills": skills,
        "education": sections["education"],
        "experience": sections["experience"],
        "projects": sections["projects"],
        "certifications": sections["certifications"],
    }

    return result


if __name__ == "__main__":

    resume_path = Path("data/raw/test_resume.docx")

    resume_data = analyze_resume(str(resume_path))

    print("\n" + "=" * 60)
    print("RESUME ANALYSIS")
    print("=" * 60)

    print(f"\nName: {resume_data['name']}")

    print("\nSkills:")
    for skill in resume_data["skills"]:
        print(f"  - {skill}")

    print("\nEducation:")
    print(resume_data["education"])

    print("\nExperience:")
    print(resume_data["experience"])

    print("\nProjects:")
    print(resume_data["projects"])

    print("\nCertifications:")
    print(resume_data["certifications"])

    print("\n" + "=" * 60)