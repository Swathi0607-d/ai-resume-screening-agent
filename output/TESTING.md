# Testing Notes

This document records the tests run against the Resume Screening Agent
before final submission.

## Test 1 — Single resume, offline mode (no API key)
Command:py -3.12 agent.py --jd sample_data/job_description.txt --resumes sample_data/resumes --no-llm

Result: Ran successfully. TF-IDF scores computed correctly for all 10
resumes without calling the AI model. Reasoning field showed the
"[OFFLINE MODE]" placeholder as expected. Confirms the agent works even
without an API key configured.

## Test 2 — Full run, 10 resumes, with AI reasoning
Command:py -3.12 agent.py --jd sample_data/job_description.txt --resumes sample_data/resumes --out output/ranked_results

Result: All 10 resumes scored, ranked, and given AI-generated reasoning
via the Gemini API. Output saved correctly to
`output/ranked_results.json` and `output/ranked_results.csv`.
Ranking order matched expectations: strong Python/backend candidates
(Priya Sharma, Rohan Iyer, Vikram Singh) ranked highest; unrelated
profiles (HR, Marketing, Design) ranked lowest.

## Test 3 — Edge case: completely unrelated resume
Resume: `resume_sneha_nair.txt` (HR Executive, no technical background)
Result: Correctly scored lowest (1.54) with zero matched skills. AI
reasoning correctly identified the candidate as "not a fit" and explained
why, without inventing any false technical skills.

## Test 4 — Edge case: overqualified candidate
Resume: `resume_rohan_iyer.txt` (6+ years senior engineer vs. JD asking
for 2+ years)
Result: Scored highly (rank #2) as expected since all required skills
are present. AI reasoning correctly flagged the overqualification as a
potential retention/salary risk, showing the reasoning step adds real
value beyond the raw similarity score.

## Test 5 — Skill match breakdown accuracy
Manually checked `matched_skills` / `missing_skills` output for Priya
Sharma against her resume text. All matched skills (Python, Django,
Flask, PostgreSQL, AWS, Docker, pytest, etc.) were verified present in
her resume text. The only skill listed as missing was "orm" -- correct,
since her resume doesn't use that exact word even though she uses
SQLAlchemy and Django ORM in practice (a known limitation, see README
tradeoffs).

## Test 6 — File format handling
Confirmed `.txt` resumes parse correctly (all 10 sample resumes are
`.txt`). PDF and DOCX parsing functions (`read_pdf`, `read_docx`) are
implemented using `pdfplumber` and `python-docx` but not exercised in
the sample dataset -- noted as a limitation in the README.

## Summary
All 10 sample resumes processed successfully in a single run, satisfying
the "handle 10+ resumes" requirement. The scoring is deterministic and
reproducible (same TF-IDF score every run), while the AI reasoning adds
qualitative context a pure numeric score can't capture.