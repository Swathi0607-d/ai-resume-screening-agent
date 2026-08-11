# AI Resume Screening Agent

An AI-powered agent that takes a job description and a folder of resumes, and produces a ranked, scored shortlist of candidates with an explanation for each score.

> **One-sentence description:** This agent takes a job description and a set of resumes, and produces an ordered shortlist of candidates with a relevance score and a plain-English explanation of fit for each one.

---

## How it works (pipeline)

```text
Job Description + Resumes (.txt/.pdf/.docx)
            |
            v
1. Extract text from every file
            |
            v
2. Score each resume against the JD using TF-IDF + cosine similarity
   (a standard, deterministic NLP technique -- not the AI model)
            |
            v
3. Ask Groq (LLM) to explain *why* each candidate scored the way
   they did -- matched skills, gaps, notable strengths -- constrained
   to stay consistent with the calculated skill match
            |
            v
4. Rank candidates by score, save to CSV + JSON
```

**Why split scoring and reasoning like this?** The numeric score comes from TF-IDF/cosine similarity -- deterministic, so it produces the exact same score every time and can't "hallucinate" a number. The LLM is used only for what it's actually good at: reading in context and writing a clear explanation. See [Tradeoffs](#tradeoffs--design-notes) for why we didn't just ask the LLM to output the score directly.

---

## Setup (step by step)

### 1. Install Python 3.12

This project requires Python 3.10-3.12 (some dependencies don't yet support very new Python releases like 3.14). Check your version:

```text
python --version
```

If needed, download Python 3.12 from [Python.org](https://www.python.org/downloads/release/python-3120/).

### 2. Clone this repo

```text
git clone https://github.com/Swathi0607-d/ai-resume-screening-agent.git
cd ai-resume-screening-agent
```

### 3. Install dependencies

```text
pip install -r requirements.txt
```

### 4. Get a Groq API key

1. Go to the [Groq Console](https://console.groq.com/keys).
2. Sign in and click **Create API Key**.
3. Copy the generated API key.

### 5. Configure your key

Create a file named `.env` in the project root (copy `.env.example` and rename it) and add your key:

```text
GROQ_API_KEY=your-real-key-here
```

**Never commit your real `.env` file** -- it is already excluded via `.gitignore`.

---

## Running the agent

```text
python agent.py --jd sample_data/job_description.txt --resumes sample_data/resumes --out output/ranked_results
```

This will:

* Read `sample_data/job_description.txt`
* Read all 10 resumes in `sample_data/resumes/`
* Print a ranked shortlist to the terminal
* Save full results to `output/ranked_results.json` and `output/ranked_results.csv`

### CLI options

| Flag        | Description                                          | Default                 |
| ----------- | ---------------------------------------------------- | ----------------------- |
| `--jd`      | Path to job description file (.txt/.pdf/.docx)       | required                |
| `--resumes` | Path to folder of resumes (.txt/.pdf/.docx)          | required                |
| `--out`     | Output file prefix (no extension)                    | `output/ranked_results` |
| `--no-llm`  | Skip AI reasoning, scores only (faster, no API cost) | off                     |

### Using your own job description and resumes

Point `--jd` and `--resumes` at your own files/folder -- any mix of `.txt`, `.pdf`, and `.docx` is supported.

### Running without an API key

```text
python agent.py --jd sample_data/job_description.txt --resumes sample_data/resumes --no-llm
```

Skips the AI reasoning call and outputs only numeric scores and matched/missing skills.

---

## Sample output

Running the agent on the 10 included sample resumes against the included "Backend Python Developer" job description produces this ranking (see `output/ranked_results.json` / `.csv` for the full data with reasoning):

| Rank | Candidate    | Score | Why (short)                                                                 |
| ---: | ------------ | ----: | --------------------------------------------------------------------------- |
|    1 | Priya Sharma |  21.4 | 4 yrs Django/Flask, AWS, Docker, pytest -- near-complete skill overlap      |
|    2 | Rohan Iyer   | 19.04 | 6+ yrs senior backend -- fully qualified, flagged as possibly overqualified |
|    3 | Vikram Singh | 17.33 | Strong backend skills, though resume leans full-stack (React)               |
|    4 | Farhan Ali   | 16.26 | 3 yrs Django/PostgreSQL/Docker -- solid match                               |
|    5 | Sahil Kapoor | 13.48 | Junior/intern level, Flask + MySQL basics                                   |
|    6 | Arjun Mehta  |  9.09 | Some Python/Flask/MySQL but mostly scripting, not backend-focused           |
|    7 | Meera Das    |  6.55 | Data analyst -- Python/SQL present but wrong specialization                 |
|    8 | Kavya Reddy  |  2.57 | Graphic designer -- no technical overlap                                    |
|    9 | Ananya Gupta |  1.58 | Marketing role -- no relevant overlap                                       |
|   10 | Sneha Nair   |  1.54 | HR role -- no relevant overlap                                              |

This ordering matches what a human reviewer would expect: strong Python backend engineers at the top, and unrelated fields (design, marketing, HR) correctly pushed to the bottom.

---

## Project structure

```text
ai-resume-screening-agent/
├── agent.py                     # main agent script
├── requirements.txt             # Python dependencies
├── .env.example                 # template for API key config
├── TESTING.md                   # test cases and results
├── sample_data/
│   ├── job_description.txt
│   └── resumes/                 # 10 sample resumes (.txt)
├── output/
│   ├── ranked_results.json
│   └── ranked_results.csv
└── README.md
```

---

## Scoring method

**Relevance score:** TF-IDF vectorization of the JD and each resume, followed by cosine similarity between the JD vector and each resume vector, scaled to 0-100. This rewards resumes sharing specific, JD-relevant vocabulary (e.g. "Django", "PostgreSQL", "AWS") rather than generic words, since TF-IDF down-weights terms that appear everywhere.

**Skill match breakdown:** A fixed list of ~28 common backend-engineering skill keywords is checked against both the JD and each resume, giving a transparent "matched vs missing" list a human can sanity-check against the raw score.

**AI-generated reasoning:** Groq (`llama-3.3-70b-versatile`) reads the JD and resume together and writes a short, specific explanation of fit. The prompt passes in the already-calculated matched/missing skill lists as ground truth and explicitly instructs the model not to contradict them -- this keeps the written explanation consistent with the structured data rather than letting the LLM re-derive (and potentially misjudge) skill matches on its own. The prompt also forbids inventing a numeric score or fabricating details not present in the resume.

---

## Tradeoffs & design notes

* **Why not let the LLM assign the score directly?** LLM scoring is inconsistent across many documents in a batch -- the same resume can score differently depending on prompt phrasing or ordering. TF-IDF is deterministic: same input, same score every time, which matters for a hiring shortlist.
* **Why TF-IDF and not embeddings?** Given the time constraints, TF-IDF needed no extra API calls or model downloads and is fast, free, and easy to explain. With more time I'd use embedding-based similarity (e.g. `sentence-transformers`) to better handle synonyms (e.g. "Postgres" vs "PostgreSQL").
* **Name extraction is a heuristic.** Assumes the candidate's name is the first line of the resume -- true for most resumes but can fail on ones with a header/logo before the name.
* **Skill keyword list is hardcoded** (~28 skills in `agent.py`). With more time, this would be generated dynamically per-JD via the LLM so the agent adapts to any job description without code changes.
* **PDF/DOCX parsing implemented but not exercised in the sample data** (all 10 sample resumes are `.txt`). The `read_pdf` and `read_docx` functions are implemented but are not covered by the current sample test dataset.
* **No OCR for scanned/image-based PDFs.** `pdfplumber` extracts text from real (selectable-text) PDFs only.
* **What I'd add with more time:** parallelize the LLM calls (currently sequential, one per resume), a simple web UI instead of CLI args, and a config file for the skill keyword list instead of hardcoding it.

---

## Tech stack

* **Language:** Python 3.12
* **AI model:** Groq (`llama-3.3-70b-versatile`) -- for reasoning text
* **NLP similarity:** scikit-learn's `TfidfVectorizer` + `cosine_similarity`
* **File parsing:** `pdfplumber` (PDF), `python-docx` (DOCX), built-in (TXT)
* **Output:** CSV and JSON

## Testing

See [TESTING.md](TESTING.md) for the full list of test cases run before submission.
