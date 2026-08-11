import os
import re
import csv
import json
import time
import hashlib
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
# Experience (years) and education extraction
# --------------------------------------------------------------------------

def extract_years_of_experience(resume_text: str):
    """
    Looks for patterns like '4 years', '6+ years', '2-3 years' in the resume
    text and returns the highest number found -- a simple, explainable
    heuristic rather than a black-box guess.
    """
    matches = re.findall(r"(\d+)\+?\s*(?:-\s*\d+\s*)?years?", resume_text, re.IGNORECASE)
    if not matches:
        return None
    return max(int(m) for m in matches)


def extract_education(resume_text: str):
    """
    Looks for common degree abbreviations/keywords and returns the line
    they appear on, which usually contains the full degree + institution.
    """
    degree_keywords = [
        "b.tech", "btech", "b.e.", "be ", "m.tech", "mtech", "mba",
        "b.sc", "bsc", "m.sc", "msc", "diploma", "bachelor", "master", "phd",
    ]
    for line in resume_text.splitlines():
        line_lower = line.lower()
        if any(kw in line_lower for kw in degree_keywords):
            return line.strip()
    return None


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
# Response caching -- avoid paying/spending quota twice for the same
# (job description, resume) pair.
# --------------------------------------------------------------------------

CACHE_DIR = Path(".cache")
CACHE_FILE = CACHE_DIR / "reasoning_cache.json"


def _cache_key(jd_text: str, resume_text: str, matched: list[str] | None = None,
                missing: list[str] | None = None) -> str:
    h = hashlib.sha256()
    h.update(jd_text.encode("utf-8"))
    h.update(b"\x00")
    h.update(resume_text.encode("utf-8"))
    h.update(b"\x00")
    h.update(",".join(sorted(matched or [])).encode("utf-8"))
    h.update(b"\x00")
    h.update(",".join(sorted(missing or [])).encode("utf-8"))
    return h.hexdigest()


def load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_cache(cache: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# AI-generated reasoning (Groq)
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an experienced technical recruiter assistant.
You will be given a job description and one candidate's resume text.
Write a short, honest assessment (3-4 sentences) of how well this
candidate fits the role. Mention 1-3 concrete strengths and 1-2 concrete
gaps, referencing specific skills or experience from the resume and JD.
Do not invent details that are not in the resume. Do not give a numeric
score -- only write the explanation.
"""

GROQ_MODEL = "llama-3.3-70b-versatile"


class DailyQuotaExhausted(Exception):
    """Raised when the Groq free-tier daily request/token quota is used up."""
    pass


def get_llm_reasoning(jd_text: str, resume_text: str, candidate_name: str,
                       matched_skills: list[str] | None = None,
                       missing_skills: list[str] | None = None,
                       model=None, cache: dict | None = None,
                       max_retries: int = 3) -> str:
    matched_skills = matched_skills or []
    missing_skills = missing_skills or []

    if model is None:
        return (
            "[OFFLINE MODE - no GROQ_API_KEY set. This is a placeholder so "
            "the pipeline still runs end-to-end. Set GROQ_API_KEY in your "
            ".env file to get real AI-generated reasoning.]"
        )

    # 1. Check cache first -- this is the main way we avoid burning quota.
    key = _cache_key(jd_text, resume_text, matched_skills, missing_skills)
    if cache is not None and key in cache:
        return cache[key] + "  [cached]"

    matched_str = ", ".join(matched_skills) if matched_skills else "none"
    missing_str = ", ".join(missing_skills) if missing_skills else "none"

    user_prompt = (
        f"JOB DESCRIPTION:\n{jd_text}\n\n"
        f"CANDIDATE ({candidate_name}) RESUME:\n{resume_text}\n\n"
        f"CALCULATED SKILL MATCH (already determined by keyword matching -- "
        f"treat this as ground truth, do not contradict it):\n"
        f"- Skills the candidate HAS: {matched_str}\n"
        f"- Skills the candidate is MISSING: {missing_str}\n\n"
        "Write the assessment now. Your strengths and gaps must be "
        "consistent with the CALCULATED SKILL MATCH above -- do not claim "
        "the candidate has a skill listed as missing, and do not claim "
        "they lack a skill listed as matched. You may still reference "
        "other relevant experience from the resume that isn't in the "
        "skill list."
    )

    from groq import RateLimitError

    delay = 5
    for attempt in range(1, max_retries + 1):
        try:
            response = model.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
                max_tokens=300,
            )
            text = response.choices[0].message.content.strip()
            if cache is not None:
                cache[key] = text
                save_cache(cache)
            return text
        except RateLimitError as e:
            message = str(e)
            # Groq reports daily limits as "requests per day" / "TPD"
            # (tokens per day) in the error body -- that's a hard wall for
            # today, so stop the whole run instead of retrying pointlessly.
            if any(term in message.lower() for term in
                   ("per day", "tpd", "rpd", "daily")):
                raise DailyQuotaExhausted(
                    "Groq free-tier DAILY quota is used up. "
                    "Results so far have been saved -- re-run tomorrow, "
                    "or on the same command, to pick up remaining resumes "
                    "(cached ones won't cost you anything)."
                ) from e
            # Otherwise it's a short per-minute limit -- worth a short wait.
            if attempt == max_retries:
                raise
            print(f"  Rate limited, retrying in {delay}s "
                  f"(attempt {attempt}/{max_retries})...", file=sys.stderr)
            time.sleep(delay)
            delay *= 2

    # Should not be reached
    raise RuntimeError("Unexpected retry loop exit")


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
    cache = load_cache()
    if use_llm:
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            from groq import Groq
            model = Groq(api_key=api_key)
        else:
            print(
                "WARNING: GROQ_API_KEY not set. Running in offline mode -- "
                "scores are real, but reasoning text will be a placeholder.",
                file=sys.stderr,
            )

    results = []
    quota_hit = False
    for path, text, score in zip(resume_files, resume_texts, scores):
        candidate_name = guess_candidate_name(text, fallback=path.stem)
        matched, missing = skill_match_breakdown(text, skills)
        years_experience = extract_years_of_experience(text)
        education = extract_education(text)

        try:
            reasoning = get_llm_reasoning(
                jd_text, text, candidate_name,
                matched_skills=matched, missing_skills=missing,
                model=model, cache=cache,
            )
        except DailyQuotaExhausted as e:
            print(f"\n{e}\n", file=sys.stderr)
            reasoning = "[SKIPPED - daily Groq quota exhausted, re-run later]"
            quota_hit = True

        results.append({
            "candidate_name": candidate_name,
            "file": path.name,
            "score": score,
            "years_experience": years_experience,
            "education": education,
            "matched_skills": matched,
            "missing_skills": missing,
            "reasoning": reasoning,
        })

        if quota_hit:
            # Stop calling the API for the rest of this run, but still
            # score + save every remaining resume (score-only, no LLM text).
            for path2, text2, score2 in zip(
                resume_files[len(results):], resume_texts[len(results):],
                scores[len(results):]
            ):
                candidate_name2 = guess_candidate_name(text2, fallback=path2.stem)
                matched2, missing2 = skill_match_breakdown(text2, skills)
                results.append({
                    "candidate_name": candidate_name2,
                    "file": path2.name,
                    "score": score2,
                    "years_experience": extract_years_of_experience(text2),
                    "education": extract_education(text2),
                    "matched_skills": matched2,
                    "missing_skills": missing2,
                    "reasoning": "[SKIPPED - daily Groq quota exhausted, re-run later]",
                })
            break

    results.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(results, start=1):
        r["rank"] = i

    ordered_results = [
        {
            "rank": r["rank"],
            "candidate_name": r["candidate_name"],
            "file": r["file"],
            "score": r["score"],
            "years_experience": r["years_experience"],
            "education": r["education"],
            "matched_skills": r["matched_skills"],
            "missing_skills": r["missing_skills"],
            "reasoning": r["reasoning"],
        }
        for r in results
    ]

    save_outputs(ordered_results, out_prefix)
    print_summary(ordered_results)

    if quota_hit:
        print(
            "\nNote: daily quota ran out partway through. Resumes already "
            "processed are cached, so re-running the same command later "
            "will only spend quota on the ones still marked SKIPPED.",
            file=sys.stderr,
        )


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
            "rank", "candidate_name", "file", "score", "years_experience",
            "education", "matched_skills", "missing_skills", "reasoning",
        ])
        for r in results:
            writer.writerow([
                r["rank"], r["candidate_name"], r["file"], r["score"],
                r["years_experience"], r["education"],
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