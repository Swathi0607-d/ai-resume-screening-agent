import os
import re
import csv
import json
import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------------------------------
# Reading resumes in different formats
# --------------------------------------------------------------------------

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
# Candidate name guessing
# --------------------------------------------------------------------------

def guess_candidate_name(text: str, fallback: str) -> str:
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        word_count = len(line.split())
        looks_like_name = (
            1 <= word_count <= 4
            and not re.search(r"[\d@]", line)
            and len(line) < 60
        )
        if looks_like_name:
            return line
        break
    return fallback


# --------------------------------------------------------------------------
# Job description skill extraction
# --------------------------------------------------------------------------

def extract_skill_keywords(jd_text: str) -> list[str]:
    common_skills = [
        "python", "django", "flask", "fastapi", "rest api", "rest apis",
        "sql", "postgresql", "mysql", "sqlalchemy", "orm", "git", "docker",
        "kubernetes", "aws", "ec2", "s3", "lambda", "ci/cd", "pytest",
        "unit testing", "microservices", "agile", "scrum", "terraform",
        "react", "javascript", "html", "css",
    ]
    jd_lower = jd_text.lower()
    return [skill for skill in common_skills if skill in jd_lower]


def skill_match_breakdown(resume_text: str, skills: list[str]) -> tuple[list[str], list[str]]:
    resume_lower = resume_text.lower()
    matched = [s for s in skills if s in resume_lower]
    missing = [s for s in skills if s not in resume_lower]
    return matched, missing


# --------------------------------------------------------------------------
# NLP similarity score (TF-IDF + cosine similarity)
# --------------------------------------------------------------------------

def compute_similarity_scores(jd_text: str, resume_texts: list[str]) -> list[float]:
    documents = [jd_text] + resume_texts
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(documents)

    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(jd_vector, resume_vectors)[0]
    return [round(float(s) * 100, 2) for s in similarities]


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


def get_llm_reasoning(jd_text: str, resume_text: str, candidate_name: str, model=None) -> str:
    if model is None:
        return (
            "[OFFLINE MODE - no GEMINI_API_KEY set. This is a placeholder so "
            "the pipeline still runs end-to-end. Set GEMINI_API_KEY in your "
            ".env file to get real AI-generated reasoning.]"
        )

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"JOB DESCRIPTION:\n{jd_text}\n\n"
        f"CANDIDATE ({candidate_name}) RESUME:\n{resume_text}\n\n"
        "Write the assessment now."
    )
    response = model.generate_content(prompt)
    return response.text.strip()


# --------------------------------------------------------------------------
# The main pipeline
# --------------------------------------------------------------------------

def run(jd_path: str, resumes_dir: str, out_prefix: str, use_llm: bool = True):
    jd_path = Path(jd_path)
    resumes_dir = Path(resumes_dir)

    jd_text = extract_text(jd_path)
    skills = extract_skill_keywords(jd_text)

    resume_files = sorted([
        p for p in resumes_dir.iterdir()
        if p.suffix.lower() in (".txt", ".pdf", ".docx")
    ])
    if not resume_files:
        print(f"No resumes found in {resumes_dir}", file=sys.stderr)
        sys.exit(1)

    resume_texts = [extract_text(p) for p in resume_files]
    scores = compute_similarity_scores(jd_text, resume_texts)

    model = None
    if use_llm:
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-flash-latest")
        else:
            print(
                "WARNING: GEMINI_API_KEY not set. Running in offline mode -- "
                "scores are real, but reasoning text will be a placeholder.",
                file=sys.stderr,
            )

    results = []
    for path, text, score in zip(resume_files, resume_texts, scores):
        candidate_name = guess_candidate_name(text, fallback=path.stem)
        matched, missing = skill_match_breakdown(text, skills)
        reasoning = get_llm_reasoning(jd_text, text, candidate_name, model=model)

        results.append({
            "candidate_name": candidate_name,
            "file": path.name,
            "score": score,
            "matched_skills": matched,
            "missing_skills": missing,
            "reasoning": reasoning,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(results, start=1):
        r["rank"] = i

    ordered_results = [
        {
            "rank": r["rank"],
            "candidate_name": r["candidate_name"],
            "file": r["file"],
            "score": r["score"],
            "matched_skills": r["matched_skills"],
            "missing_skills": r["missing_skills"],
            "reasoning": r["reasoning"],
        }
        for r in results
    ]

    save_outputs(ordered_results, out_prefix)
    print_summary(ordered_results)


def save_outputs(results: list[dict], out_prefix: str):
    out_dir = Path(out_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = f"{out_prefix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    csv_path = f"{out_prefix}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "candidate_name", "file", "score",
            "matched_skills", "missing_skills", "reasoning",
        ])
        for r in results:
            writer.writerow([
                r["rank"], r["candidate_name"], r["file"], r["score"],
                "; ".join(r["matched_skills"]), "; ".join(r["missing_skills"]),
                r["reasoning"],
            ])

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")


def print_summary(results: list[dict]):
    print("\n" + "=" * 70)
    print("RANKED SHORTLIST")
    print("=" * 70)
    for r in results:
        print(f"#{r['rank']:>2}  {r['candidate_name']:<20} score={r['score']:>6}  "
              f"({r['file']})")
    print("=" * 70)


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Resume Screening Agent")
    parser.add_argument("--jd", required=True, help="Path to job description file (.txt/.pdf/.docx)")
    parser.add_argument("--resumes", required=True, help="Path to folder containing resumes")
    parser.add_argument("--out", default="output/ranked_results", help="Output file prefix (no extension)")
    parser.add_argument("--no-llm", action="store_true", help="Skip AI reasoning, scores only")
    args = parser.parse_args()

    run(
        jd_path=args.jd,
        resumes_dir=args.resumes,
        out_prefix=args.out,
        use_llm=not args.no_llm,
    )


if __name__ == "__main__":
    main()