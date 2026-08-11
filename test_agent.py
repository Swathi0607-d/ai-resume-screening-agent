"""
Automated tests for the Resume Screening Agent.

Run with:
    pytest test_agent.py -v
"""

from pathlib import Path

from agent import (
    extract_text,
    extract_skill_keywords,
    compute_similarity_scores,
    skill_match_breakdown,
    guess_candidate_name,
)


JD_PATH = Path("sample_data/job_description.txt")
RESUMES_DIR = Path("sample_data/resumes")
STRONG_MATCH_RESUME = RESUMES_DIR / "resume_priya_sharma.txt"
WEAK_MATCH_RESUME = RESUMES_DIR / "resume_sneha_nair.txt"


def test_sample_data_exists():
    assert JD_PATH.exists(), "Sample job description is missing"
    resume_files = list(RESUMES_DIR.glob("*.txt"))
    assert len(resume_files) >= 10, "Need at least 10 sample resumes"


def test_extract_text_reads_txt_file():
    text = extract_text(STRONG_MATCH_RESUME)
    assert isinstance(text, str)
    assert len(text) > 0
    assert "Priya" in text


def test_extract_skill_keywords_finds_known_skills():
    jd_text = extract_text(JD_PATH)
    skills = extract_skill_keywords(jd_text)
    assert "python" in skills
    assert "aws" in skills


def test_similarity_score_is_higher_for_strong_match():
    jd_text = extract_text(JD_PATH)
    strong_text = extract_text(STRONG_MATCH_RESUME)
    weak_text = extract_text(WEAK_MATCH_RESUME)

    scores = compute_similarity_scores(jd_text, [strong_text, weak_text])
    strong_score, weak_score = scores

    assert strong_score > weak_score, (
        f"Expected strong match ({strong_score}) to outscore "
        f"weak match ({weak_score})"
    )


def test_similarity_scores_are_in_valid_range():
    jd_text = extract_text(JD_PATH)
    resume_text = extract_text(STRONG_MATCH_RESUME)
    scores = compute_similarity_scores(jd_text, [resume_text])
    assert 0 <= scores[0] <= 100


def test_skill_match_breakdown_matched_and_missing_dont_overlap():
    jd_text = extract_text(JD_PATH)
    skills = extract_skill_keywords(jd_text)
    resume_text = extract_text(STRONG_MATCH_RESUME)

    matched, missing = skill_match_breakdown(resume_text, skills)
    assert set(matched).isdisjoint(set(missing))
    assert set(matched) | set(missing) == set(skills)


def test_guess_candidate_name_returns_first_line():
    text = "Priya Sharma\nBackend Software Engineer\n..."
    name = guess_candidate_name(text, fallback="unknown")
    assert name == "Priya Sharma"


def test_guess_candidate_name_falls_back_when_first_line_is_odd():
    text = "1234 this is clearly not a name @@@\nSecond line"
    name = guess_candidate_name(text, fallback="fallback_name")
    assert name == "fallback_name"