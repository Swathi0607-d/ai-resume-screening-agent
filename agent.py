from pathlib import Path


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    import pdfplumber
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)


def read_docx(path: Path) -> str:
    import docx
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text(path: Path) -> str:
    """Dispatch to the right reader based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return read_txt(path)
    elif suffix == ".pdf":
        return read_pdf(path)
    elif suffix == ".docx":
        return read_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


# --- Quick manual test, so you can see this actually works ---
if __name__ == "__main__":
    sample_resume = Path("sample_data/resumes/resume_priya_sharma.txt")
    text = extract_text(sample_resume)
    print("Extracted text from:", sample_resume)
    print("-" * 50)
    print(text)