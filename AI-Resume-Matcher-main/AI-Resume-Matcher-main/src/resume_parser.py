from pathlib import Path

import pymupdf # PyMuPDF
from docx import Document


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Expected a PDF file.")

    text = []

    with pymupdf.open(path) as pdf:
        for page in pdf:
            page_text = page.get_text()

            if page_text:
                text.append(page_text)

    return "\n".join(text).strip()


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() != ".docx":
        raise ValueError("Expected a DOCX file.")

    document = Document(path)

    paragraphs = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            paragraphs.append(paragraph.text.strip())

    return "\n".join(paragraphs).strip()


def extract_resume_text(file_path: str) -> str:
    """Extract text automatically based on file type."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()

    if extension == ".pdf":
        return extract_text_from_pdf(file_path)

    if extension == ".docx":
        return extract_text_from_docx(file_path)

    raise ValueError("Unsupported file format. Please use PDF or DOCX.")


if __name__ == "__main__":
    from pathlib import Path

    resume_path = Path("data/raw/test_resume.docx")

    extracted_text = extract_resume_text(str(resume_path))

    print("\n" + "=" * 60)
    print("EXTRACTED RESUME TEXT")
    print("=" * 60)

    print(extracted_text)

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)