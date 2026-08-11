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

# --------------------------------------------------------------------------
# Job description skill extraction
# --------------------------------------------------------------------------

def extract_skill_keywords(jd_text: str) -> list[str]:
    """
    Pull candidate 'skill-like' tokens out of the job description so we can
    later check which resumes mention them. This is a simple keyword list
    approach -- easy to explain and verify, though it only catches skills
    from a fixed list rather than discovering new ones automatically.
    """
    common_skills = [
        "python", "django", "flask", "fastapi", "rest api", "rest apis",
        "sql", "postgresql", "mysql", "sqlalchemy", "orm", "git", "docker",
        "kubernetes", "aws", "ec2", "s3", "lambda", "ci/cd", "pytest",
        "unit testing", "microservices", "agile", "scrum", "terraform",
        "react", "javascript", "html", "css",
    ]
    jd_lower = jd_text.lower()
    return [skill for skill in common_skills if skill in jd_lower]

# --------------------------------------------------------------------------
# NLP similarity score (TF-IDF + cosine similarity)
# --------------------------------------------------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity_scores(jd_text: str, resume_texts: list[str]) -> list[float]:
    """
    TF-IDF turns text into vectors of "how important is each word to this
    document". Cosine similarity then measures the angle between the JD
    vector and each resume vector -- a score from 0 (nothing in common) to
    1 (near-identical vocabulary). We scale it to 0-100 for readability.
    """
    documents = [jd_text] + resume_texts
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(documents)

    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(jd_vector, resume_vectors)[0]
    return [round(float(s) * 100, 2) for s in similarities]

# --- Quick manual test, so you can see this actually works ---
if __name__ == "__main__":
    sample_resume = Path("sample_data/resumes/resume_priya_sharma.txt")
    text = extract_text(sample_resume)

    jd_path = Path("sample_data/job_description.txt")
    jd_text = extract_text(jd_path)
    skills = extract_skill_keywords(jd_text)
    print("Skills found in job description:")
    print(skills)

    scores = compute_similarity_scores(jd_text, [text])
    print("\nSimilarity score for Priya Sharma's resume:", scores[0])