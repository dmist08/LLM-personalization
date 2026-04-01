# Cold-Start StyleVector — Antigravity Agent Prompts
## Complete Implementation Guide (Claude via Google Antigravity)

**How to use:**
- Use **Planning mode** for all complex, multi-file prompts (Phases 2–10)
- Use **Fast mode** for isolated single-file tasks and quick verifications
- Work through prompts IN ORDER — each depends on the previous
- After each prompt, paste the agent's output back and verify before continuing
- Reference `@COLD_START_STYLEVECTOR_V3.md` in any prompt for full project context

---

## STEP 0 — Antigravity Rules

Paste into: **Antigravity → Settings → Customizations → Rules**

```
You are a senior ML engineer working on Cold-Start StyleVector, a personalized
headline generation research project.

HARD RULES (never break these):

1. Before running any terminal command, show the exact command and what it does.
   For destructive commands (delete, overwrite, pip install --upgrade all packages),
   always wait for explicit user confirmation. For read-only commands (python -c, 
   nvidia-smi, ls), run directly.

2. Never open the browser to look things up unless the user explicitly says 
   "you can search for X". Ask the user if you need to verify something.

3. Before writing any code, state your understanding of the task and your
   implementation plan in 3-5 bullet points. Proceed unless something is
   genuinely ambiguous — don't ask for confirmation on obvious things.

4. Always generate complete, runnable code. No placeholders, no "TODO: implement 
   this", no skeleton functions left empty. Every function must have a body.

5. Handle all edge cases explicitly: empty files, missing fields, single-author 
   datasets, authors with <5 articles, malformed dates, missing agnostic headlines.

6. Python 3.10+ syntax only. All paths via pathlib.Path. Logging via Python's 
   logging module — never print(). Timestamps in all log messages.

7. No Gemini API anywhere in this project. Style-agnostic headlines use base 
   LLaMA-3.1-8B-Instruct only. This is a hard rule.

8. RAG baseline uses BM25 per-author only. Never mix articles across authors.
   Retrieval corpus = that author's train split ONLY. Never val or test.

9. After finishing each module, always provide:
   (a) what the script outputs
   (b) the exact command to run it
   (c) what to check in the output to confirm it worked

10. If a result looks wrong (NaN loss, ROUGE-L = 0, empty output files, 
    unexpected shapes), STOP and diagnose before continuing. Never silently 
    continue past a broken state.

PROJECT CONTEXT:
- Project root: D:/HDD/Project/DL/
- Indian dataset: data/raw/indian_news/ as JSONL files
- JSONL fields: author, author_name, author_id, source, url, headline, body,
  date, word_count, scraped_at
- HT: 6,601 articles, 25 authors | TOI: 3,318 articles, 18 authors
- LaMP-4: data/raw/LaMP_4/ with train_questions.json, dev_questions.json,
  test_questions.json, train_outputs.json, dev_outputs.json, test_outputs.json
- Rich authors: ≥50 articles | Sparse: 5–20 articles (cold-start test set)
- Model: LLaMA-3.1-8B-Instruct (base + fine-tuned via QLoRA)
- Compute: Lightning AI L4 GPU (24GB VRAM)
- Conda env: cold_start_sv (Python 3.10)

ML-SPECIFIC RULES (apply to all training/inference code):
- NEVER fit any scaler, encoder, or PCA on validation or test data
- NEVER modify files in data/raw/ — they are immutable
- ALWAYS save best model checkpoint by val metric, never last epoch
- ALWAYS clip gradients: torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
- ALWAYS use torch.no_grad() during validation and inference
- Use mixed precision (torch.cuda.amp.autocast) for all GPU forward passes
- Log MD5 hash of input data files at the start of every training run
- Set seeds: random, numpy, torch, transformers.set_seed — always at script start
```

---

## PHASE 0 — Environment Verification

**Mode:** Fast  
**Expected time:** 15 minutes

```
Task: Verify the project environment is fully ready on Lightning AI.

Run these verification commands and show me the full output of each:

  conda activate cold_start_sv
  python --version                        # Must show 3.10.x
  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  python -c "import transformers; print(transformers.__version__)"
  python -c "import peft; print(peft.__version__)"
  python -c "import bitsandbytes; print(bitsandbytes.__version__)"
  python -c "import rank_bm25; print('rank_bm25 OK')"
  python -c "import rouge_score; print('rouge_score OK')"
  python -c "import nltk; print('nltk OK')"
  nvidia-smi

If any import fails, show the exact error and suggest the pip install command.
Do NOT run the install automatically — show me the command first.

After all checks pass, create these directories if they don't exist:
  src/pipeline/
  src/baselines/
  data/processed/indian/
  data/processed/lamp4/
  data/interim/
  author_vectors/
  outputs/baselines/
  outputs/evaluation/
  logs/

Create src/__init__.py and src/pipeline/__init__.py if missing.
```

---

## PROMPT 1 — `src/config.py`

**Mode:** Fast  
**Expected time:** 20 minutes

```
Task: Create src/config.py

This is the single source of truth for all hyperparameters and paths.
No magic numbers anywhere else in the codebase — everything references this file.

Requirements:

1. Use Python dataclasses with defaults. No argparse, no YAML loading.

2. Create these dataclasses:

   @dataclass
   class PathConfig:
       project_root: Path = Path("D:/HDD/Project/DL")
       raw_dir: Path = project_root / "data/raw/indian_news"
       lamp4_dir: Path = project_root / "data/raw/LaMP_4"
       processed_dir: Path = project_root / "data/processed"
       indian_processed_dir: Path = processed_dir / "indian"
       lamp4_processed_dir: Path = processed_dir / "lamp4"
       interim_dir: Path = project_root / "data/interim"
       vectors_dir: Path = project_root / "author_vectors"
       outputs_dir: Path = project_root / "outputs"
       models_dir: Path = project_root / "models"
       logs_dir: Path = project_root / "logs"

   @dataclass
   class DataConfig:
       min_body_words: int = 100
       max_body_words: int = 5000
       toi_min_year: int = 2015
       train_ratio: float = 0.70
       val_ratio: float = 0.15
       test_ratio: float = 0.15
       rich_author_min_articles: int = 50
       sparse_author_min_articles: int = 5
       sparse_author_max_articles: int = 20

   @dataclass
   class ModelConfig:
       base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
       lora_rank: int = 16
       lora_alpha: int = 32
       lora_dropout: float = 0.1
       lora_target_modules: list = field(default_factory=lambda: [
           "q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj"
       ])
       quantization_bits: int = 4
       extraction_layer_range: tuple = (15, 28)
       extraction_layers: list = field(default_factory=lambda: [15, 18, 21, 24, 27])
       pca_components: int = 50
       kmeans_k_range: tuple = (5, 20)
       alpha_range: list = field(default_factory=lambda: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
       intervention_alpha: float = 0.7     # default steering strength at inference
       max_new_tokens_headline: int = 30
       max_new_tokens_agnostic: int = 40

   @dataclass
   class TrainingConfig:
       batch_size: int = 4
       grad_accumulation: int = 8          # effective batch = 32
       learning_rate: float = 2e-4
       num_epochs: int = 2
       save_steps: int = 500
       warmup_ratio: float = 0.03
       max_seq_length: int = 1024
       max_train_samples: int = 25000      # cap to avoid OOM on full LaMP-4
       hf_repo_id: str = "dharmik-mistry/cold-start-stylevector"
       seed: int = 42

   @dataclass
   class RAGConfig:
       k_retrieved: int = 2
       max_article_words_in_context: int = 150
       max_new_tokens: int = 30

   @dataclass
   class EvalConfig:
       metrics: list = field(default_factory=lambda: ["rouge_l", "meteor", "bert_score"])
       bert_score_model: str = "distilbert-base-uncased"
       paper_rouge_l: float = 0.0411      # paper's reported number — for sanity check
       paper_meteor: float = 0.0809

3. Top-level Config that composes all:
   @dataclass
   class Config:
       paths: PathConfig = field(default_factory=PathConfig)
       data: DataConfig = field(default_factory=DataConfig)
       model: ModelConfig = field(default_factory=ModelConfig)
       training: TrainingConfig = field(default_factory=TrainingConfig)
       rag: RAGConfig = field(default_factory=RAGConfig)
       eval: EvalConfig = field(default_factory=EvalConfig)

4. get_config() function:
   - Returns Config()
   - Calls mkdir(parents=True, exist_ok=True) on every path in PathConfig
   - Returns the config

5. if __name__ == "__main__": block that prints the full config as JSON.
   Use dataclasses.asdict() and handle Path serialization (convert to str).

Output: Complete src/config.py
After generating, tell me the exact command to verify it works.
```

---

## PROMPT 2 — `src/utils.py`

**Mode:** Fast  
**Expected time:** 20 minutes

```
Task: Create src/utils.py

Shared utilities used across all pipeline stages.

Implement exactly these functions (complete, no stubs):

1. setup_logging(name: str, log_dir: Path, level=logging.INFO) -> logging.Logger
   - Creates a logger that writes to both console AND a timestamped file in log_dir
   - File name format: {name}_{YYYYMMDD_HHMMSS}.log
   - Format: "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
   - Returns the configured logger

2. set_seed(seed: int = 42) -> None
   - Sets seed for: random, numpy, torch, os.environ["PYTHONHASHSEED"]
   - Also calls transformers.set_seed(seed) if transformers is available
   - Sets torch.backends.cudnn.deterministic = True
   - Sets torch.backends.cudnn.benchmark = False

3. load_jsonl(path: Path) -> list[dict]
   - Streams JSONL line by line (do NOT load all into memory at once for large files)
   - Skips blank lines silently
   - Logs and skips malformed JSON lines (don't crash)
   - Returns list of dicts

4. save_jsonl(records: list[dict], path: Path) -> None
   - Creates parent directories if needed
   - Writes one JSON record per line
   - Uses ensure_ascii=False for non-ASCII characters (important for Indian names)

5. get_device() -> str
   - Returns "cuda" if torch.cuda.is_available()
   - Returns "mps" if torch.backends.mps.is_available() (Apple Silicon)
   - Returns "cpu" otherwise
   - Logs which device was selected and available VRAM if CUDA

6. compute_file_hash(filepath: Path) -> str
   - Returns MD5 hex digest of the file
   - Used for data versioning — log this at start of every training run

7. format_article_for_prompt(article_text: str, max_words: int = 800) -> str
   - Truncates article to max_words words (split on whitespace)
   - Strips extra whitespace and newlines
   - Returns clean string

8. parse_date_safe(date_str: str) -> datetime | None
   - Tries these formats in order: "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S",
     "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S+%f"
   - If all fail: uses dateutil.parser.parse with a try/except as final fallback
   - Returns datetime object or None if completely unparseable
   - Never raises

9. slugify(name: str) -> str
   - "Ananya Das" -> "ananya-das"
   - "N Ananthanarayanan" -> "n-ananthanarayanan"
   - Lowercase, replace spaces and non-alphanumeric chars with hyphens
   - Collapse multiple hyphens into one, strip leading/trailing hyphens

10. estimate_runtime(n_items: int, seconds_per_item: float) -> str
    - Returns human-readable string: "~2h 34m" or "~45m" or "~3m 20s"
    - Used for progress logging in long-running jobs

No external dependencies beyond: pathlib, logging, json, datetime, re, random,
numpy, torch, transformers (optional), python-dateutil

Output: Complete src/utils.py
```

---

## PROMPT 3 — `src/pipeline/validate_indian_data.py`

**Mode:** Planning  
**Expected time:** 1 hour

```
Task: Create src/pipeline/validate_indian_data.py

Cleans and validates the scraped HT + TOI data before any ML work.
This is the FIRST step in the pipeline. Garbage in = garbage out.

Input files (stream them — do NOT load all into memory):
  data/raw/indian_news/hindustan_times_articles.jsonl
  data/raw/indian_news/toi_articles.jsonl

Input record schema (actual fields from the scraper):
  {
    "author": "Ananya Das",
    "author_name": "Ananya Das",
    "author_id": "ananya-das-101616133886672",
    "source": "HT" or "Times of India",
    "url": "https://...",
    "headline": "...",
    "body": "...",
    "date": "2026-03-29",
    "word_count": 655,
    "scraped_at": "2026-03-30"
  }

Output schema (unified, standardized):
  {
    "author_id": "ananya-das",         ← slugify(author_name), consistent
    "author_name": "Ananya Das",
    "source": "HT",                    ← normalize: "HT" or "TOI"
    "url": "https://...",
    "headline": "...",
    "article_text": "...",             ← rename from "body"
    "date": "2026-03-29",             ← always YYYY-MM-DD
    "word_count": 655
  }

VALIDATION RULES — reject article if any rule fails (log rejection reason):

1. author_name: non-empty, not a desk account
   Desk patterns (case-insensitive): "desk", "tnn", "correspondent",
   "bureau", "agency", "pti", "ani", "ians", "staff", "reporter", "web team"

2. headline: non-empty, len >= 10 chars, len <= 200 chars

3. article_text: word_count >= config.data.min_body_words (100)
   AND word_count <= config.data.max_body_words (5000)

4. date: parseable → extract as YYYY-MM-DD using parse_date_safe

5. url: non-empty, starts with "http"

6. TOI only: year(date) >= config.data.toi_min_year (2015)

7. LISTICLE FILTER — reject headline if:
   - Starts with digit(s) + one of: "best", "top", "things", "ways", "tips", "reasons"
     (e.g., "5 best ways to..." or "10 tips for...")
   - Contains (case-insensitive): "horoscope", "zodiac", "web story", "in pics",
     "gallery", "slideshow", "watch video", "ipl schedule", "quiz:", "in photos"

DEDUPLICATION: remove duplicate URLs (keep first occurrence across both files)

NORMALIZATION STEPS:
- source: "Times of India" → "TOI", keep "HT" as-is
- author_id: slugify(author_name), NOT the raw author_id from the scraper
  (the scraped author_id has long numeric suffixes — we want clean slugs)
- word_count: recompute from article_text if field is missing or 0
- Field priority: prefer "author_name" over "author" if both exist and differ
- Body field: check "body" first, then "article_text" as fallback

Output file:
  data/processed/indian_news_clean.jsonl

Print quality report at the end:
  ┌──────────────────────────────────────────────────────┐
  │           DATA VALIDATION REPORT                     │
  ├──────────────────────────────────────────────────────┤
  │  Input records (HT):                  6,601          │
  │  Input records (TOI):                 3,318          │
  │  Total input:                         9,919          │
  │                                                      │
  │  Valid records output:                X,XXX          │
  │                                                      │
  │  Rejected — desk accounts:              XXX          │
  │  Rejected — word count (too short):     XXX          │
  │  Rejected — word count (too long):       XX          │
  │  Rejected — unparseable date:            XX          │
  │  Rejected — TOI pre-2015:               XXX          │
  │  Rejected — listicle/low quality:       XXX          │
  │  Rejected — duplicate URL:               XX          │
  │  Rejected — invalid headline:            XX          │
  │                                                      │
  │  Per-source breakdown:                               │
  │    HT:  X,XXX articles | XX authors                 │
  │    TOI: X,XXX articles | XX authors                 │
  │                                                      │
  │  Per-author article counts:                          │
  │  ┌──────────────────────────┬────────┬──────┐       │
  │  │ Author                   │ Source │ Count│       │
  │  ├──────────────────────────┼────────┼──────┤       │
  │  │ Ananya Das               │ HT     │  XXX │       │
  │  │ ...                      │ ...    │  ... │       │
  │  └──────────────────────────┴────────┴──────┘       │
  └──────────────────────────────────────────────────────┘

Import Config from src/config.py and utilities from src/utils.py.

Verification check:
  After running, confirm:
  1. Output file exists at data/processed/indian_news_clean.jsonl
  2. Record count is in expected range (expect 7,000–9,500 after filtering)
  3. No record has an empty article_text or headline
  4. All author_ids are clean slugs (no numbers, no underscores)

Output: Complete src/pipeline/validate_indian_data.py
Tell me: (a) exact run command, (b) what to check in the report
```

---

## PROMPT 4 — `src/pipeline/split_dataset.py`

**Mode:** Planning  
**Expected time:** 1 hour

```
Task: Create src/pipeline/split_dataset.py

Creates leakage-free chronological train/val/test splits per author.

CRITICAL RULE — this is a research project. The test set must NEVER
appear in training or retrieval. Sort by date FIRST, then split.
Most recent articles → test. Oldest → train. Never shuffle before splitting.

Input:
  data/processed/indian_news_clean.jsonl (from validate_indian_data.py)

PER-AUTHOR SPLITTING ALGORITHM:
  1. Group articles by author_id
  2. Sort each group by date ascending (oldest first)
  3. n = len(articles)
  4. Edge case — n < 5:
       Put ALL articles in train. Skip this author for val/test.
       Log warning: "Author {name} has only {n} articles — all placed in train"
  5. Edge case — 5 <= n < 10:
       train = articles[:int(n*0.70)]
       test = articles[int(n*0.70):]  (skip val)
  6. Standard — n >= 10:
       train_end = int(n * 0.70)
       val_end = int(n * 0.85)
       train = articles[:train_end]
       val = articles[train_end:val_end]
       test = articles[val_end:]

ADD "author_class" FIELD to every record before saving:
  "rich"   → total_articles >= 50
  "mid"    → 21 <= total_articles <= 49
  "sparse" → 5 <= total_articles <= 20
  "tiny"   → total_articles < 5

OUTPUT DIRECTORY STRUCTURE:
  data/processed/indian/{author_id}/train.jsonl
  data/processed/indian/{author_id}/val.jsonl
  data/processed/indian/{author_id}/test.jsonl

Also create unified cross-author files:
  data/processed/indian/all_train.jsonl
  data/processed/indian/all_val.jsonl
  data/processed/indian/all_test.jsonl

Author metadata file:
  data/processed/indian/author_metadata.json
  Format: {
    "ananya-das": {
      "name": "Ananya Das",
      "source": "HT",
      "total": 618,
      "train": 432,
      "val": 93,
      "test": 93,
      "class": "rich"
    },
    ...
  }

Print summary table:
  Author split summary:
  ┌──────────────────────────┬────────┬───────┬──────┬──────┬─────────┐
  │ Author                   │ Total  │ Train │ Val  │ Test │ Class   │
  ├──────────────────────────┼────────┼───────┼──────┼──────┼─────────┤
  │ Ananya Das               │   618  │  432  │   93 │   93 │ rich    │
  │ ...                      │        │       │      │      │         │
  └──────────────────────────┴────────┴───────┴──────┴──────┴─────────┘

  Class distribution:
    Rich  (≥50):   X authors, X,XXX total articles
    Mid   (21-49): X authors,   XXX total articles
    Sparse (5-20): X authors,   XXX total articles
    Tiny  (<5):    X authors,    XX total articles

Verification checks after running:
  1. No article appears in both train and test for the same author
  2. Test articles are always chronologically AFTER train articles (per author)
  3. author_metadata.json exists and has entries for all authors
  4. Per-author JSONL files exist in data/processed/indian/

Output: Complete src/pipeline/split_dataset.py
```

---

## PROMPT 5 — `src/pipeline/prepare_lamp4.py`

**Mode:** Planning  
**Expected time:** 1–2 hours

```
Task: Create src/pipeline/prepare_lamp4.py

Converts LaMP-4 raw data into the project's unified schema.

LAMP-4 RAW FILE STRUCTURE (actual files, DO NOT assume HuggingFace format):
  data/raw/LaMP_4/train_questions.json  — list of question objects
  data/raw/LaMP_4/dev_questions.json
  data/raw/LaMP_4/test_questions.json
  data/raw/LaMP_4/train_outputs.json   — list of {id: str, output: str} objects
  data/raw/LaMP_4/dev_outputs.json
  (test_outputs.json may NOT exist — test set has no labels by design)

QUESTION OBJECT STRUCTURE:
  {
    "id": "user_42_0",
    "input": "Generate a headline for the following article: {article text here}",
    "profile": [
      {"id": "user_42_0", "title": "headline text", "text": "article body"},
      ...
    ]
  }

INPUT PARSING:
  Strip the instruction prefix from "input" to get raw article text.
  Check for ALL these prefix variants (the dataset has inconsistencies):
    - "Generate a headline for the following article: "
    - "Please write a headline for the following article:\n"
    - "Write a headline for the following article: "
    - Any other prefix ending with ": " before the article body
  If no known prefix is found, use the full input string (log a warning).

USER ID EXTRACTION:
  "user_42_0" → user_id = "user_42"  (everything before the last underscore+number)

OUTPUT SCHEMA (per record):
  {
    "user_id": "user_42",
    "article_text": "...",         ← stripped from input
    "headline": "...",             ← from outputs.json (None for test split)
    "profile": [
      {"article_text": "...", "headline": "..."},
      ...
    ],
    "split": "train",
    "profile_size": 42,
    "lamp4_id": "user_42_0"
  }

PROFILE CONSTRUCTION:
  For each question, the profile contains the user's OTHER questions.
  Exclude the current question from its own profile (to prevent leakage).
  Order = original order in the questions file (proxy for temporal order).
  Only include profile entries where BOTH article_text and headline are non-empty.

FILTERS:
  - Skip records where stripped article_text < 20 words
  - Skip records where headline is empty (except test split)
  - Skip profile entries where title or text is empty

USER CLASSIFICATION (add "user_class" field):
  "lamp4_rich":        profile_size >= 50
  "lamp4_mid":         20 <= profile_size < 50
  "lamp4_sparse_sim":  5 <= profile_size < 20  (for cold-start simulation)
  "lamp4_tiny":        profile_size < 5

COLD-START SIMULATION — critical for evaluation:
  For lamp4_rich users, create additional records with truncated profiles.
  These simulate the cold-start condition using known ground-truth users.
  Save to data/processed/lamp4/cold_start_simulation/:
    profile_5.jsonl   — profile truncated to first 5 entries
    profile_10.jsonl  — profile truncated to first 10 entries
    profile_15.jsonl  — profile truncated to first 15 entries
    profile_20.jsonl  — profile truncated to first 20 entries
  Only create these for lamp4_rich users (those with ≥50 profile docs).
  Full profile remains in train.jsonl for the same user.

OUTPUT FILES:
  data/processed/lamp4/train.jsonl
  data/processed/lamp4/val.jsonl        ← from dev_questions.json
  data/processed/lamp4/test.jsonl       ← no headlines (None)
  data/processed/lamp4/user_metadata.json
  data/processed/lamp4/cold_start_simulation/profile_5.jsonl
  data/processed/lamp4/cold_start_simulation/profile_10.jsonl
  data/processed/lamp4/cold_start_simulation/profile_15.jsonl
  data/processed/lamp4/cold_start_simulation/profile_20.jsonl

Print summary:
  LaMP-4 processing summary:
  ┌────────────────────────────────────────────┐
  │  Total train users:          X,XXX         │
  │  Total val users:            X,XXX         │
  │  Total test users:           X,XXX         │
  │                                            │
  │  User classes (train):                     │
  │    lamp4_rich        (≥50):  XXX           │
  │    lamp4_mid       (20-49):  XXX           │
  │    lamp4_sparse_sim (5-19):  XXX           │
  │    lamp4_tiny         (<5):  XXX           │
  │                                            │
  │  Cold-start simulation sets:               │
  │    profile_5:   X,XXX records             │
  │    profile_10:  X,XXX records             │
  │    profile_15:  X,XXX records             │
  │    profile_20:  X,XXX records             │
  │                                            │
  │  Profile size distribution (train):        │
  │    min: X, median: XXX, p95: X,XXX        │
  └────────────────────────────────────────────┘

Output: Complete src/pipeline/prepare_lamp4.py
```

---

## PROMPT 6 — `src/pipeline/rag_baseline.py`

**Mode:** Planning  
**Expected time:** 2 hours

```
Task: Create src/pipeline/rag_baseline.py

BM25 RAG baseline using BASE LLaMA-3.1-8B-Instruct (no fine-tuning).
This is Baseline 2 in the evaluation table.
Baseline 1 (no personalization) is generated as a byproduct of the same script.

Required install (show user, don't run):
  pip install rank-bm25

ARCHITECTURE RULES:
- BM25 index built PER AUTHOR. Never mix articles across authors.
- Retrieval corpus = that author's TRAIN split ONLY. Never val or test.
- k=2 retrieved examples (matching the paper exactly).

class AuthorBM25Index:
    def __init__(self, author_id: str, train_articles: list[dict]):
        # train_articles: list of {article_text, headline} dicts
        # Tokenize: lowercase, split on whitespace, no stemming
        # Build BM25Okapi index
        # Store original articles for retrieval
    
    def retrieve(self, query_article: str, k: int = 2) -> list[dict]:
        # Returns top-k dicts with {article_text, headline}
        # Edge case: len(train_articles) < k → return all available (no crash)
        # Edge case: len(train_articles) == 0 → return [] (sparse fallback)

class RAGPromptBuilder:
    def __init__(self, max_article_words: int = 150):
        ...
    
    def build_prompt(self, new_article: str, retrieved_examples: list[dict]) -> str:
        # If retrieved_examples is empty (author has no train history):
        #   Fallback to: "Write a concise news headline for:\n\n{article}\n\nHeadline:"
        # Otherwise use EXACTLY this template:
        # ---
        # Here are past headlines written by this journalist:
        #
        # Article: {truncated_article_1}
        # Headline: {headline_1}
        #
        # Article: {truncated_article_2}
        # Headline: {headline_2}
        #
        # Now write a headline for the following article:
        # Article: {new_article_truncated}
        #
        # Headline:
        # ---
        # NO text after "Headline:" — the model generates from there
        
    def build_base_prompt(self, new_article: str) -> str:
        # Baseline 1 prompt (no context):
        # "Write a concise news headline for the following article:\n\n{article}\n\nHeadline:"

class RAGBaseline:
    def __init__(self, model_name: str):
        # Load tokenizer and model
        # Use load_in_8bit=True for memory efficiency (~10GB for 8B model)
        # Set model to eval() mode
        # Log GPU memory after loading
    
    def _load_author_indices(self, train_dir: Path) -> dict[str, AuthorBM25Index]:
        # Load all per-author train JSONL files from data/processed/indian/{author_id}/train.jsonl
        # Build BM25 index for each author
        # Cache in self._indices dict
        # Log: "Built BM25 indices for X authors"
    
    def generate_headline(
        self,
        prompt: str,
        max_new_tokens: int = 30
    ) -> str:
        # Greedy decoding (do_sample=False, temperature=1.0)
        # torch.no_grad() + autocast()
        # Extract ONLY the newly generated tokens (strip the prompt)
        # Clean output: strip quotes, strip "Headline:", strip newlines, strip leading spaces
    
    def run_evaluation(
        self,
        test_dir: Path,
        output_path: Path,
        author_ids: list[str] | None = None
    ) -> None:
        # For each author in test set:
        #   For each test article:
        #     - Generate Baseline 1 output (no context prompt)
        #     - Generate RAG output (BM25 retrieved prompt)
        #     - Save both in one record
        #
        # Output record format:
        # {
        #   "author_id": "...",
        #   "author_name": "...",
        #   "author_class": "rich|sparse|mid|tiny",
        #   "source": "HT|TOI",
        #   "article_text": "...",
        #   "ground_truth": "...",
        #   "base_output": "...",      ← Baseline 1
        #   "rag_output": "...",       ← Baseline 2
        #   "num_retrieved": 2         ← actual examples retrieved (may be <2)
        # }
        #
        # Save to: outputs/baselines/rag_and_base_outputs.jsonl
        # Do NOT compute metrics here — just save outputs.
        #
        # Progress: "[5/43 authors] Ananya Das: 93 test articles done | ~12 min remaining"
        # Resume: check if output_path exists, skip already-processed authors

Standalone runner:
  if __name__ == "__main__":
    args: --model-path, --test-dir, --train-dir, --output-path,
          --authors (optional comma-separated list for partial runs)
    Show user: exact run command with all paths filled in for this project

Verification after running:
  1. outputs/baselines/rag_and_base_outputs.jsonl exists
  2. Every record has both base_output and rag_output as non-empty strings
  3. No output looks like the raw prompt (model echoed prompt — this is a bug)
  4. Rough sanity: headline lengths should be 5–20 words, not full articles

Output: Complete src/pipeline/rag_baseline.py
```

---

## PROMPT 7 — `src/pipeline/agnostic_gen.py`

**Mode:** Planning  
**Expected time:** 1 hour coding + 8–12 hours GPU runtime

```
Task: Create src/pipeline/agnostic_gen.py

Generates style-agnostic (neutral, generic) headlines for TRAIN split articles
using BASE LLaMA-3.1-8B-Instruct. These are the "negative" samples for
contrastive activation extraction in Phase 4.

WHY THIS STEP EXISTS:
For each article we need two headlines:
  1. Journalist's real headline (in dataset) → "positive" (authentic style)
  2. Style-agnostic headline (from this script) → "negative" (content only, no style)
The difference in LLM activations = the style signal. This is StyleVector's core idea.

USE EXACTLY THIS PROMPT (do not change wording):
  "Write a neutral, factual news headline for the following article.
   Be concise and objective.\n\n{article}\n\nHeadline:"

This is DIFFERENT from the RAG baseline prompt — it deliberately avoids any
style reference, simulating what a wire service would write.

class AgnosticHeadlineGenerator:
    def __init__(self, model_name: str, batch_size: int = 8):
        # Load base model in 8-bit
        # batch_size for batched inference (much faster than one-by-one)
    
    def generate_batch(self, articles: list[str]) -> list[str]:
        # Batch tokenize all articles, left-pad for batch generation
        # max_new_tokens=40, greedy decoding, no sampling
        # Decode only generated tokens (exclude prompt)
        # Clean each output: strip "Headline:", quotes, newlines
        # Return list of strings in same order as input
    
    def process_dataset(
        self,
        input_jsonl: Path,
        output_csv: Path,
        article_field: str = "article_text",
        id_field: str = "url",
        resume: bool = True
    ) -> None:
        # RESUME LOGIC (critical — this job takes 8-12 hours):
        #   If output_csv exists: load already-processed IDs into a set
        #   Skip records whose ID is already in that set
        #   Open output_csv in append mode (do NOT overwrite)
        #
        # Output CSV columns: id, agnostic_headline
        #
        # Process in batches of self.batch_size
        # Flush CSV buffer every 200 articles (don't lose progress on crash)
        #
        # Progress logging every 100 articles:
        #   "Processed 1000/6601 articles (15.2%) | ~2h 34m remaining"
        #   (use estimate_runtime from utils.py)
        #
        # Log GPU memory usage every 500 articles
        # torch.cuda.empty_cache() every 500 articles

Standalone runner:
  if __name__ == "__main__":
  --dataset: choices = ["indian", "lamp4", "both"]
  --indian-input:  data/processed/indian/all_train.jsonl
  --lamp4-input:   data/processed/lamp4/train.jsonl
  --output-dir:    data/interim/
  --batch-size:    default 8
  --resume:        default True

  Output files:
    data/interim/indian_agnostic_headlines.csv    (id=url)
    data/interim/lamp4_agnostic_headlines.csv     (id=lamp4_id)

CRITICAL: Only process TRAIN split articles.
Never process val or test (leakage prevention).

RUNTIME ESTIMATE for user (add as a comment):
  L4 GPU: ~1.5-2s per article at batch_size=8
  Indian train (~7,000 articles): ~3-4 hours
  LaMP-4 train (~25,000 articles): ~10-14 hours
  Recommendation: run --dataset indian first to validate, then run lamp4 overnight.

Verification after running:
  1. Output CSV exists with expected number of rows
  2. No empty agnostic_headline values
  3. Spot check: agnostic headlines should look generic/wire-service style,
     NOT like the journalist's real headlines
  4. If 10%+ of outputs are empty strings: the prompt stripping has a bug

Output: Complete src/pipeline/agnostic_gen.py
```

---

## PROMPT 8 — `notebooks/02_qlora_finetune.ipynb`

**Mode:** Planning  
**Expected time:** 1 hour coding + 8–12 hours GPU runtime

```
Task: Create notebooks/02_qlora_finetune.ipynb

Fine-tunes LLaMA-3.1-8B-Instruct on LaMP-4 with author-conditioned prompts.
Designed for Lightning AI L4 (24GB VRAM). Session may disconnect — build in recovery.

Generate as a Python script with # %% cell markers.
I will convert to .ipynb or run as a script with jupytext.

## Cell 1: Pip installs (show commands, don't auto-run)
  # Run these before starting the notebook:
  # pip install transformers peft bitsandbytes accelerate datasets trl wandb

## Cell 2: Imports and seed
  All imports. Call set_seed(42) from src/utils.py.

## Cell 3: Config
  MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
  HF_REPO_ID = "dharmik-mistry/cold-start-stylevector"
  OUTPUT_DIR = Path("./checkpoints/qlora")
  TRAIN_FILE = Path("data/processed/lamp4/train.jsonl")
  VAL_FILE = Path("data/processed/lamp4/val.jsonl")
  MAX_SEQ_LENGTH = 1024
  BATCH_SIZE = 4
  GRAD_ACCUM = 8          # effective batch = 32
  LR = 2e-4
  NUM_EPOCHS = 2
  SAVE_STEPS = 500
  WARMUP_RATIO = 0.03
  LORA_RANK = 16
  LORA_ALPHA = 32
  LORA_DROPOUT = 0.1
  MAX_TRAIN_SAMPLES = 25000
  LORA_TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

## Cell 4: Dataset formatting
  For each record, build the FULL training text in this EXACT format:
  
  f"Write a headline in the style of {record['user_id']}:\n\n{record['article_text'][:800]}\n\nHeadline: {record['headline']}{tokenizer.eos_token}"
  
  Use SFTTrainer from trl — it handles prompt masking automatically.
  Limit to MAX_TRAIN_SAMPLES. If more, sample randomly (set seed=42 before sampling).

## Cell 5: Load model in 4-bit
  BitsAndBytesConfig:
    load_in_4bit=True
    bnb_4bit_quant_type="nf4"
    bnb_4bit_compute_dtype=torch.bfloat16
    bnb_4bit_use_double_quant=True
  
  After loading:
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()  ← saves ~3GB VRAM
    model = get_peft_model(model, LoraConfig(...))
    Print trainable parameters (should be ~0.3-0.5% of total)

## Cell 6: Training with SFTTrainer
  TrainingArguments:
    output_dir=str(OUTPUT_DIR)
    num_train_epochs=NUM_EPOCHS
    per_device_train_batch_size=BATCH_SIZE
    gradient_accumulation_steps=GRAD_ACCUM
    learning_rate=LR
    warmup_ratio=WARMUP_RATIO
    lr_scheduler_type="cosine"
    save_steps=SAVE_STEPS
    save_total_limit=3
    evaluation_strategy="steps"
    eval_steps=SAVE_STEPS
    logging_steps=50
    bf16=True
    dataloader_num_workers=0    ← avoid Windows multiprocessing issues
    report_to="none"            ← disable wandb for now (can enable later)
    load_best_model_at_end=True
    metric_for_best_model="eval_loss"
  
  Add after each checkpoint save:
    model.push_to_hub(HF_REPO_ID + "-checkpoint")
    tokenizer.push_to_hub(HF_REPO_ID + "-checkpoint")
  
  NOTE FOR USER (in comment):
    # If session crashes, resume with:
    # trainer = Trainer(..., resume_from_checkpoint=str(OUTPUT_DIR))
    # All checkpoints also saved to HF Hub — recoverable even if local storage is lost

## Cell 7: Save and push final model
  trainer.save_model(str(OUTPUT_DIR / "final"))
  model = model.merge_and_unload()   ← merge LoRA weights into base model
  model.save_pretrained(str(OUTPUT_DIR / "merged"))
  model.push_to_hub(HF_REPO_ID)
  tokenizer.push_to_hub(HF_REPO_ID)

## Cell 8: Quick validation
  Load merged model. Run on 5 random test examples.
  Print table:
    User ID | Ground Truth | Model Output | Match?
  
  If outputs look like garbage (repeated tokens, empty strings, prompt echo):
    STOP. The merge or training may have failed. Do not proceed to Phase 4.

RUNTIME COMMENT TO INCLUDE:
  # Expected training time on L4:
  # 25,000 samples × 2 epochs / (batch 4 × grad_accum 8) = ~800 steps
  # ~40-45s per step on L4 → ~9-10 hours total
  # Lightning AI L4 sessions: up to 24h — should complete in one session
  # Checkpoint every 500 steps → max 500 steps lost if crash

Output: Complete Python script with # %% cell markers.
```

---

## PROMPT 9 — `src/pipeline/extract_style_vectors.py`

**Mode:** Planning  
**Expected time:** 2 hours coding + 8–12 hours GPU runtime

```
Task: Create src/pipeline/extract_style_vectors.py

Extracts style vectors for every author using the fine-tuned LLaMA-3.1-8B-Instruct.
This is the core StyleVector extraction — Phase 4 of the project.

CONCEPTUAL EXPLANATION (embed this in comments at the top of the file):
  For each author u with N articles:
    For each article i:
      pos_text = article_i + "\n\nHeadline: " + real_headline_i
      neg_text = article_i + "\n\nHeadline: " + agnostic_headline_i
      pos_activation = hidden_state_at_layer_l(pos_text, last_token)
      neg_activation = hidden_state_at_layer_l(neg_text, last_token)
      diff_i = pos_activation - neg_activation
    style_vector_u = mean(diff_i for all i)   ← shape: [4096]
  
  The "last token" is position -1 on the sequence dimension.
  We extract the OUTPUT of transformer block l (post-attention, post-FF).
  Layer l is swept over [15, 18, 21, 24, 27] — best layer selected on val set.

class ActivationExtractor:
    def __init__(self, model, tokenizer, layer_indices: list[int]):
        self.model = model
        self.tokenizer = tokenizer
        self._hook_handles = {}
        self._layer_outputs = {}    # layer_idx -> np.ndarray [hidden_dim]
    
    def _register_hooks(self, layer_indices: list[int]) -> None:
        # Hook target: model.model.layers[i]
        # FIRST: verify this attribute exists:
        #   assert hasattr(model, 'model') and hasattr(model.model, 'layers')
        # Hook captures OUTPUT (not input) of the layer:
        #   def hook(module, input, output):
        #     hidden = output[0] if isinstance(output, tuple) else output
        #     self._layer_outputs[layer_idx] = hidden[0, -1, :].cpu().float().numpy()
        # Store handle: self._hook_handles[i] = layer.register_forward_hook(hook)
    
    def remove_hooks(self) -> None:
        for handle in self._hook_handles.values():
            handle.remove()
        self._hook_handles.clear()
        self._layer_outputs.clear()
    
    def extract_activations(
        self,
        text: str,
        layer_indices: list[int],
        max_length: int = 512
    ) -> dict[int, np.ndarray]:
        # Tokenize (truncate to max_length)
        # Single forward pass with torch.no_grad() + autocast()
        # Returns {layer_idx: np.ndarray of shape [4096]}
        # Clears self._layer_outputs after extraction

class StyleVectorExtractor:
    def __init__(self, model_path: str):
        # Load fine-tuned merged model in 8-bit
        # Instantiate ActivationExtractor
        # Log: model loaded, VRAM usage
    
    def extract_author_vector(
        self,
        train_articles: list[dict],
        agnostic_headlines: dict[str, str],    # url -> agnostic_headline
        layer_idx: int,
        author_id: str
    ) -> np.ndarray | None:
        # For each article in train_articles:
        #   Look up agnostic headline by url
        #   If missing: log warning, skip
        #   Build pos_text and neg_text (truncate article to 400 words)
        #   Extract pos and neg activations
        #   diff = pos - neg
        # If fewer than 5 valid diffs: log warning, return None
        # Return np.mean(diffs, axis=0) — shape [4096]
        # Clear GPU cache after each author: torch.cuda.empty_cache()
    
    def extract_all_authors(
        self,
        train_dir: Path,
        agnostic_csv: Path,
        layer_indices: list[int],
        output_dir: Path,
        resume: bool = True
    ) -> None:
        # For each author, for each layer:
        #   Save to: {output_dir}/layer_{l}/{author_id}.npy
        # Resume: skip if output file already exists
        # Save manifest JSON: {output_dir}/manifest.json
        #   {author_id: {"layer_15": "path", "layer_18": "path", ...}}
        # Progress: log author name, article count, estimated time
    
    def layer_sweep_on_val(
        self,
        val_jsonl: Path,
        agnostic_csv: Path,
        style_vector_dir: Path,
        layer_indices: list[int],
        inference_fn: callable
    ) -> dict[int, float]:
        # For each layer: run StyleVector inference on first 200 val records
        # Compute ROUGE-L for each layer
        # Log results table:
        #   Layer 15: ROUGE-L = 0.038
        #   Layer 18: ROUGE-L = 0.041
        #   Layer 21: ROUGE-L = 0.043  ← best
        # Return {layer: rouge_l}, best layer logged clearly
        # Save plot: outputs/layer_sweep.png

Also process LaMP-4 rich users (lamp4_rich class) using the same extraction.
Use lamp4 train JSONL and lamp4 agnostic CSV.
Save to: author_vectors/lamp4/layer_{l}/{user_id}.npy

Standalone runner:
  --model-path, --train-dir, --agnostic-csv, --output-dir
  --layers (comma-separated, default: 15,18,21,24,27)
  --dataset (indian|lamp4|both)
  --run-layer-sweep (flag to also run val sweep after extraction)

Verification:
  1. Spot-check one author: load their vector, confirm shape is (4096,)
  2. Compute cosine similarity between two authors from same cluster —
     should be higher than between two random authors
  3. Layer sweep result should show non-flat curve (some layers better than others)

Output: Complete src/pipeline/extract_style_vectors.py
```

---

## PROMPT 10 — `src/pipeline/cold_start.py`

**Mode:** Planning  
**Expected time:** 2 hours

```
Task: Create src/pipeline/cold_start.py

The novel cold-start interpolation contribution. Runs on CPU. No GPU needed.

Inputs:
  - author_vectors/lamp4/layer_{l}/*.npy  — rich-author vectors (full profiles)
  - author_vectors/indian/layer_{l}/*.npy — Indian journalist vectors (sparse)
  - data/processed/indian/author_metadata.json

class ColdStartInterpolator:
    def __init__(self, layer_idx: int, vector_dir: Path, author_metadata_path: Path):
        # Load author metadata
        # Store layer_idx
        # self.pca, self.kmeans, self.centroids_4096d = None (set during fit())
    
    def _load_vectors(
        self,
        dataset: str = "lamp4",    # "lamp4" or "indian"
        author_class: str = "lamp4_rich"
    ) -> tuple[np.ndarray, list[str]]:
        # Load all vectors for the given class
        # Returns (matrix [N, 4096], list of author_ids)
        # Skip missing files with a warning (don't crash)
    
    def fit(self, k_range: tuple[int, int] = (5, 20)) -> dict:
        # Step 1: Load lamp4_rich vectors → matrix [N_rich, 4096]
        # Step 2: PCA(n_components=50) — fit ONLY on lamp4_rich vectors
        #   Log: total variance explained by 50 components
        # Step 3: KMeans sweep k=5 to 20
        #   For each k: fit KMeans on 50D vectors, compute silhouette_score
        #   If silhouette_score < 0.1 for all k:
        #     log WARNING: "Poor cluster structure detected — check PCA output"
        #   Select best k (highest silhouette)
        # Step 4: Store:
        #   self.pca (fitted)
        #   self.kmeans (fitted on best k)
        #   self.centroids_50d = self.kmeans.cluster_centers_
        #   self.centroids_4096d = self.pca.inverse_transform(self.centroids_50d)
        # Step 5: Compute cluster assignments for all lamp4_rich authors
        # Step 6: Save fit results:
        #   author_vectors/cold_start_fit.json
        #   {best_k, silhouette_scores, cluster_assignments: {author_id: cluster_id}}
        # Return fit results dict
    
    def interpolate(
        self,
        sparse_author_id: str,
        alpha: float = 0.5,
        dataset: str = "indian"
    ) -> np.ndarray:
        # Load sparse author's raw (noisy) vector
        # Project to 50D
        # Find nearest centroid using COSINE SIMILARITY (not Euclidean distance)
        #   cos_sim = (v · c) / (||v|| * ||c||)  for each centroid c
        #   nearest = centroid with HIGHEST cosine similarity
        # Interpolate in 50D:
        #   v_interp_50d = alpha * sparse_50d + (1 - alpha) * nearest_centroid_50d
        # Project back to 4096D:
        #   v_interp_4096d = self.pca.inverse_transform(v_interp_50d)
        # L2-normalize the result (unit norm)
        # Return np.ndarray [4096]
        
        # NOTE on cosine vs Euclidean:
        # Euclidean distance is dominated by vector magnitude, not direction.
        # In activation spaces, direction encodes style — magnitude is less meaningful.
        # Cosine similarity is the correct metric here.
    
    def interpolate_all_sparse(
        self,
        alpha_values: list[float],
        output_dir: Path
    ) -> None:
        # For each sparse Indian author:
        #   For each alpha:
        #     v = self.interpolate(author_id, alpha)
        #     Save to: {output_dir}/cold_start/alpha_{alpha}/{author_id}.npy
        # Also save cluster assignment info:
        #   author_vectors/cold_start/cluster_assignments.json
        #   {author_id: {
        #       cluster_id: int,
        #       nearest_centroid_authors: [top-3 lamp4_rich authors in same cluster]
        #   }}
        #   nearest_centroid_authors is important for qualitative analysis in the paper
    
    def alpha_sweep_on_val(
        self,
        val_jsonl: Path,
        agnostic_csv: Path,
        layer_idx: int,
        inference_fn: callable,
        alpha_values: list[float],
        n_val_samples: int = 100
    ) -> dict[float, float]:
        # For each alpha: run cold-start inference on first n_val_samples val records
        # Compute ROUGE-L
        # Log results:
        #   alpha=0.2: ROUGE-L = 0.038
        #   alpha=0.5: ROUGE-L = 0.043  ← best
        #   alpha=0.8: ROUGE-L = 0.041
        # Save plot: outputs/alpha_sweep.png
        # Return {alpha: rouge_l}, best alpha logged clearly

Print cluster analysis after fit():
  Cluster analysis:
  ┌─────────┬────────────┬───────────────────────────────────────────────┐
  │ Cluster │ N Authors  │ Sample members                                │
  ├─────────┼────────────┼───────────────────────────────────────────────┤
  │    0    │ 12 rich    │ user_42, user_156, user_891, ...              │
  │    1    │  9 rich    │ user_23, user_677, user_334, ...              │
  │   ...   │ ...        │ ...                                           │
  └─────────┴────────────┴───────────────────────────────────────────────┘
  
  Silhouette scores: k=5: 0.12 | k=8: 0.18 | k=10: 0.21* | k=15: 0.19 | k=20: 0.17

Also run t-SNE on the 50D vectors and save colored scatter plot:
  outputs/style_vector_tsne.png (colored by cluster label)
  This is a required figure in the paper.

Standalone runner:
  --vector-dir, --metadata, --layer, --output-dir
  --alpha-values (comma-separated, default: 0.2,0.3,0.4,0.5,0.6,0.7,0.8)
  --run-alpha-sweep (flag)

Output: Complete src/pipeline/cold_start.py
```

---

## PROMPT 11 — `src/pipeline/evaluate.py`

**Mode:** Planning  
**Expected time:** 2 hours

```
Task: Create src/pipeline/evaluate.py

Computes all evaluation metrics and produces the final results table.

Required installs (show user, don't run):
  pip install rouge-score bert-score
  python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt_tab')"

METHODS BEING EVALUATED:
  1. base          — no personalization (from rag_and_base_outputs.jsonl)
  2. rag            — BM25 RAG (from rag_and_base_outputs.jsonl)
  3. stylevector    — vanilla style vector steering (from extract_style_vectors outputs)
  4. cold_start     — cold-start interpolation (from cold_start outputs)
  For methods 3 and 4: run inference first, save outputs, then evaluate here.

AUTHOR GROUPS:
  - all: all test authors
  - rich: author_class == "rich"
  - sparse: author_class == "sparse"

METRICS:
  1. ROUGE-L F1 — rouge_score library, RougeScorer(['rougeL'])
  2. METEOR — nltk.translate.meteor_score.single_meteor_score
  3. BERTScore F1 — bert_score library, model="distilbert-base-uncased"
     (distilbert is fast enough on CPU for evaluation)

class Evaluator:
    def compute_metrics(
        self,
        predictions: list[str],
        references: list[str]
    ) -> dict:
        # Returns: {rouge_l: float, meteor: float, bert_score_f1: float}
        # All are mean across the prediction list
        # Skip empty predictions (log warning, count them)
    
    def evaluate_method(
        self,
        records: list[dict],
        pred_field: str,       # field name in records dict
        author_metadata: dict
    ) -> dict:
        # Returns:
        # {
        #   "all":    {rouge_l, meteor, bert_score_f1, n_samples},
        #   "rich":   {rouge_l, meteor, bert_score_f1, n_samples},
        #   "sparse": {rouge_l, meteor, bert_score_f1, n_samples}
        # }
    
    def generate_result_table(
        self,
        results: dict,          # method_name -> evaluate_method output
        output_dir: Path
    ) -> None:
        # Produce ASCII table to console:
        # ┌──────────────────────────┬────────────────────┬────────────────────┐
        # │ Method                   │ Rich Authors       │ Sparse Authors     │
        # │                          │ RL    MET    BS    │ RL    MET    BS    │
        # ├──────────────────────────┼────────────────────┼────────────────────┤
        # │ No personalization       │ .XXX  .XXX   .XXX  │ .XXX  .XXX   .XXX │
        # │ RAG (BM25)               │ .XXX  .XXX   .XXX  │ .XXX  .XXX   .XXX │
        # │ StyleVector (vanilla)    │ .XXX  .XXX   .XXX  │ .XXX  .XXX   .XXX │
        # │ Cold-Start StyleVector   │ .XXX  .XXX   .XXX  │ .XXX  .XXX   .XXX │
        # └──────────────────────────┴────────────────────┴────────────────────┘
        #
        # Save to:
        #   outputs/evaluation/result_table.txt   (ASCII)
        #   outputs/evaluation/result_table.json  (machine-readable)
        #   outputs/evaluation/result_table.tex   (LaTeX — for IEEE paper)
        #   outputs/evaluation/per_author_results.jsonl (per-author granular)
        
        # SANITY CHECK (print prominently):
        # Compare our StyleVector ROUGE-L on LaMP-4 against paper's Table 2
        # Paper: ROUGE-L = 0.0411
        # If our number is < 0.035 or > 0.050: print a big WARNING
        # "WARNING: Our StyleVector ROUGE-L deviates >15% from paper. Investigate before reporting."
        
        # KEY RESULT CHECK:
        # Does Cold-Start StyleVector (sparse authors) beat No Personalization?
        # Print clearly: "PASS ✓" or "FAIL ✗ — cold-start doesn't beat baseline"
        # If FAIL: do NOT hide it. The evaluation is what it is.

Standalone runner:
  --rag-outputs: outputs/baselines/rag_and_base_outputs.jsonl
  --sv-outputs:  outputs/stylevector_outputs.jsonl
  --cs-outputs:  outputs/cold_start_outputs.jsonl
  --metadata:    data/processed/indian/author_metadata.json
  --output-dir:  outputs/evaluation/

Output: Complete src/pipeline/evaluate.py
```

---

## PROMPT 12 — Deployment

**Mode:** Planning  
**Expected time:** 3–5 hours

```
Task: Build the deployment for the course demo requirement.

Two stages:
  Stage A: Pre-computed outputs served from HuggingFace Spaces + Vercel (permanent)
  Stage B: Live inference demo on Colab T4 (presentation only — label it clearly)

Focus entirely on Stage A.

FILE 1 — backend/app.py (FastAPI):
  - Startup: load pre-computed predictions JSON into memory at startup
    (do NOT load LLaMA in the API — free HF Spaces has no GPU)
  - GET /health → {"status": "ok", "version": "1.0", "n_authors": int}
  - GET /authors → list of {author_id, author_name, source, n_articles, author_class}
  - GET /methods → list of {method_id, display_name, description}
  - POST /predict:
      Input: {author_id: str, article_text: str, method: str}
      Output: {headline: str, method: str, author_id: str, latency_ms: float}
      Logic: look up pre-computed headline for (author_id, method)
      Fallback: if article_text not in cache, return nearest match by
        TF-IDF similarity (sklearn cosine_similarity) — never crash
  - POST /predict_batch → list input → list output (max 10 per batch)
  - Add CORS middleware (required for Vercel frontend to call HF Spaces API)

FILE 2 — backend/schemas.py (Pydantic):
  - PredictRequest, PredictResponse, BatchRequest, BatchResponse, AuthorInfo, MethodInfo
  - Input validation: article_text min 50 chars, method must be in known list

FILE 3 — backend/Dockerfile:
  - python:3.10-slim base
  - Non-root user (security)
  - Copy only backend/ + outputs/cached_predictions.json
  - Health check: GET /health every 30s
  - Expose port 7860 (HuggingFace Spaces default)

FILE 4 — scripts/prepare_deployment_cache.py:
  - Collects all prediction output JSONLs from outputs/
  - Builds: outputs/cached_predictions.json
    {author_id: {method: [{article_id, article_text, predicted_headline, ground_truth}]}}
  - This is what the API loads at startup

FILE 5 — frontend/ (React + TailwindCSS):
  - Single page. Key components:
    - Journalist selector dropdown (from GET /authors)
    - Method selector with descriptions (from GET /methods)
    - Article textarea (paste article body)
    - "Generate Headline" button → POST /predict
    - Side-by-side comparison: all 4 methods for the same article simultaneously
    - Metrics display: ROUGE-L for this journalist (from cached stats)
  - API_URL configured via .env.local → VITE_API_URL

FILE 6 — .github/workflows/deploy.yml (CI/CD):
  - On push to main: deploy frontend to Vercel, deploy backend Docker to HF Spaces
  - Simple workflow, no tests required

After implementing:
  1. Test API locally: uvicorn backend.app:app --reload --port 7860
  2. curl http://localhost:7860/health — show me the response
  3. curl -X POST http://localhost:7860/predict with a real author/article pair
  4. Build Docker: docker build -t cold-start-sv-api backend/
  5. Run Docker: docker run -p 7860:7860 cold-start-sv-api
  6. Confirm frontend runs: cd frontend && npm install && npm run dev

Output: All 6 files complete.
```

---

## PROMPT 13 — Report Structure & README

**Mode:** Fast  
**Expected time:** 30 minutes

```
Task: Generate report_outline.md and README.md. Do NOT write full IEEE LaTeX.
Just the structure I will write from.

FILE 1 — report_outline.md:
  Title: "Cold-Start StyleVector: Cluster-Centroid Interpolation for
  Personalized News Headline Generation with Sparse Author History"
  
  Abstract (4–5 sentences, write it in full):
    - Problem: StyleVector degrades for authors with < 20 articles
    - Method: cluster-centroid interpolation with α weighting
    - Data: 9,919 Indian news articles, 43 journalists (TOI + HT)
    - Result: [placeholder — fill with actual numbers after evaluation]
  
  Section outline with one bullet per subsection saying what goes in it.
  
  Figures list (file path → caption for each):
    - outputs/style_vector_tsne.png
    - outputs/layer_sweep.png
    - outputs/alpha_sweep.png
    - System architecture diagram (I will create manually in draw.io)
    - Cold-start performance vs. N curves
  
  Ablation study table design:
    - Effect of N (3/5/10/20 history articles) on Cold-Start StyleVector
    - Effect of α (0.2–0.8) on Cold-Start StyleVector
    - Effect of K (5–20 clusters) on cluster quality

FILE 2 — README.md:
  - 2-paragraph project description
  - Badges: Python version, license, arXiv link to StyleVector paper
  - Installation (conda + pip commands)
  - Pipeline walkthrough: numbered list of scripts in order with exact commands
  - Results summary table (placeholder — update after evaluation)
  - Live demo link (placeholder: HF Spaces URL TBD)
  - Citation block for StyleVector paper (BibTeX)
  - Contact: student name + course info

Output: Both files.
```

---

## APPENDIX — Antigravity Workflows

Save each as a file in `.agent/workflows/`. Trigger with `/workflow_name` in chat.

### `/run_eval`
```markdown
---
name: run_eval
description: Re-run full evaluation and regenerate results table
---
Run the evaluation pipeline:
1. python -m src.pipeline.evaluate (with standard args)
2. Show me the full results table
3. Print the key pass/fail check: does cold-start beat baseline on sparse authors?
4. Compare to previous run if result_table.json already exists — show what changed
```

### `/check_data`
```markdown
---
name: check_data
description: Validate data pipeline integrity
---
Run these checks:
1. Confirm data/raw/ files exist and show MD5 hashes (do NOT modify them)
2. Check data/processed/ files exist with expected row counts:
   indian_news_clean.jsonl: ~7,000-9,500 records
   lamp4/train.jsonl: ~12,527 records
   lamp4/val.jsonl: ~1,925 records
3. Check data/interim/ CSVs exist (agnostic headlines)
4. Flag any missing file or row count outside expected range
```

### `/gpu_check`
```markdown
---
name: gpu_check
description: Check GPU status before launching any training job
---
Run: nvidia-smi
Report: GPU model, total VRAM, used VRAM, running processes
If used VRAM > 20GB: warn me and do NOT launch any training job until I confirm
```

### `/resume_check`
```markdown
---
name: resume_check
description: Check which long-running jobs are complete and what needs to resume
---
Check these files and report status (exists / missing / partial):
1. data/interim/indian_agnostic_headlines.csv — expected ~7,000 rows
2. data/interim/lamp4_agnostic_headlines.csv — expected ~25,000 rows
3. author_vectors/indian/layer_21/ — expected ~43 .npy files
4. author_vectors/lamp4/layer_21/ — expected 200+ .npy files
5. checkpoints/qlora/final/ — QLoRA training complete?
For each partial file: show current row/file count vs. expected
```
