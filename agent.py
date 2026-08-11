import os
from dotenv import load_dotenv

load_dotenv()  # reads .env automatically, no need to type export ever again

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


# --------------------------------------------------------------------------
# Rank all resumes in a folder against the JD
# --------------------------------------------------------------------------

def rank_resumes(jd_path: str, resumes_dir: str) -> list[dict]:
    jd_path = Path(jd_path)
    resumes_dir = Path(resumes_dir)

    jd_text = extract_text(jd_path)

    resume_files = sorted([
        p for p in resumes_dir.iterdir()
        if p.suffix.lower() in (".txt", ".pdf", ".docx")
    ])
    resume_texts = [extract_text(p) for p in resume_files]
    scores = compute_similarity_scores(jd_text, resume_texts)

    results = [
        {"candidate_file": path.name, "score": score}
        for path, score in zip(resume_files, scores)
    ]
    results.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(results, start=1):
        r["rank"] = i
    return results


# --------------------------------------------------------------------------
# AI-generated reasoning (Google Gemini)
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an experienced technical recruiter assistant.
You will be given a job description and one candidate's resume text.
Write a short, honest assessment (3-4 sentences) of how well this
candidate fits the role. Mention 1-3 concrete strengths and 1-2 concrete
gaps, referencing specific skills or experience from the resume and JD.
Do not invent details that are not in the resume. Do not give a numeric
score -- only write the explanation.
"""


def get_llm_reasoning(jd_text: str, resume_text: str, model=None) -> str:
    """
    Calls the Gemini API to generate a human-readable explanation for this
    candidate's fit. If no model client is available (e.g. no API key set),
    we return a clearly-labeled placeholder so the pipeline still runs.
    """
    if model is None:
        return "[OFFLINE MODE - no GEMINI_API_KEY set]"

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"JOB DESCRIPTION:\n{jd_text}\n\n"
        f"CANDIDATE RESUME:\n{resume_text}\n\n"
        "Write the assessment now."
    )
    response = model.generate_content(prompt)
    return response.text.strip()


# --- Quick manual test, so you can see this actually works ---
if __name__ == "__main__":
    jd_path = Path("sample_data/job_description.txt")
    jd_text = extract_text(jd_path)

    resume_path = Path("sample_data/resumes/resume_priya_sharma.txt")
    resume_text = extract_text(resume_path)

    api_key = os.environ.get("GEMINI_API_KEY")
    model = None
    if api_key:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-flash-latest")
    else:
        print("No GEMINI_API_KEY set — reasoning will be a placeholder.\n")

    reasoning = get_llm_reasoning(jd_text, resume_text, model=model)
    print("AI reasoning for Priya Sharma:")
    print(reasoning)