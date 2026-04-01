"""
src/config.py — Single source of truth for all hyperparameters and paths.
=========================================================================
USAGE:
    from src.config import Config
    cfg = Config()
    print(cfg.MODEL_NAME, cfg.LORA_RANK, cfg.DATA_DIR)

All paths are relative to PROJECT_ROOT (auto-detected).
Override any field at runtime: cfg.LORA_RANK = 8
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    """Project-wide configuration. Modify defaults here, not in scripts."""

    # ── Paths ────────────────────────────────────────────────────────────
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    @property
    def DATA_DIR(self) -> Path:
        return self.PROJECT_ROOT / "data"

    @property
    def RAW_DIR(self) -> Path:
        return self.DATA_DIR / "raw"

    @property
    def PROCESSED_DIR(self) -> Path:
        return self.DATA_DIR / "processed"

    @property
    def SPLITS_DIR(self) -> Path:
        return self.DATA_DIR / "splits"

    @property
    def LAMP4_RAW_DIR(self) -> Path:
        return self.RAW_DIR / "LaMP_4"

    @property
    def INDIAN_RAW_DIR(self) -> Path:
        return self.RAW_DIR / "indian_news"

    @property
    def MODELS_DIR(self) -> Path:
        return self.PROJECT_ROOT / "models"

    @property
    def OUTPUTS_DIR(self) -> Path:
        return self.PROJECT_ROOT / "outputs"

    @property
    def LOGS_DIR(self) -> Path:
        return self.PROJECT_ROOT / "logs"

    @property
    def CONFIGS_DIR(self) -> Path:
        return self.PROJECT_ROOT / "configs"

    # ── Model ────────────────────────────────────────────────────────────
    MODEL_NAME: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    MODEL_DTYPE: str = "bfloat16"       # bfloat16 on L4, float16 on T4
    MAX_SEQ_LEN: int = 2048             # enough for article + headline
    DEVICE: str = "auto"                # auto-detected in utils.py

    # ── QLoRA ────────────────────────────────────────────────────────────
    LORA_RANK: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05
    LORA_TARGET_MODULES: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]
    )
    QUANTIZATION_BITS: int = 4          # 4-bit NormalFloat (NF4)

    # ── Training ─────────────────────────────────────────────────────────
    LEARNING_RATE: float = 2e-4
    BATCH_SIZE: int = 4
    GRADIENT_ACCUMULATION_STEPS: int = 8
    EPOCHS: int = 3
    WARMUP_RATIO: float = 0.03
    WEIGHT_DECAY: float = 0.01
    MAX_GRAD_NORM: float = 0.3
    LR_SCHEDULER: str = "cosine"
    SAVE_STEPS: int = 100
    EVAL_STEPS: int = 100
    LOGGING_STEPS: int = 10

    # ── Style Vector Extraction ──────────────────────────────────────────
    EXTRACTION_LAYERS: List[int] = field(
        default_factory=lambda: list(range(16, 29))  # layers 16-28 (middle-to-late)
    )
    BEST_LAYER: int = 20                # default; tuned on LaMP-4 dev
    STEERING_ALPHA: float = 0.7         # default; tuned on LaMP-4 dev
    EXTRACTION_FUNCTION: str = "mean_difference"  # paper recommends this

    # ── Cold-Start Clustering ────────────────────────────────────────────
    PCA_COMPONENTS: int = 50
    KMEANS_K_RANGE: List[int] = field(
        default_factory=lambda: list(range(5, 21))   # K = 5 to 20
    )
    COLD_START_ALPHA_RANGE: List[float] = field(
        default_factory=lambda: [0.2, 0.4, 0.6, 0.8]
    )
    COLD_START_N_ARTICLES: List[int] = field(
        default_factory=lambda: [3, 5, 10]           # simulate sparse history
    )

    # ── RAG Baseline ─────────────────────────────────────────────────────
    RAG_K: int = 2                       # retrieve top-2 similar articles
    RAG_RETRIEVER: str = "bm25"          # BM25 per-author index, no cross-author

    # ── Data Filtering ───────────────────────────────────────────────────
    MIN_ARTICLE_WORDS: int = 100
    MAX_ARTICLE_WORDS: int = 5000
    MIN_HEADLINE_WORDS: int = 4
    MAX_HEADLINE_WORDS: int = 30
    MIN_ARTICLES_PER_AUTHOR: int = 5     # drop authors with fewer
    TOI_MIN_DATE: str = "2015-01-01"     # filter out pre-2015 TOI articles

    # ── Evaluation ───────────────────────────────────────────────────────
    EVAL_METRICS: List[str] = field(
        default_factory=lambda: ["rouge_l", "meteor"]
    )

    # ── Generation ───────────────────────────────────────────────────────
    GEN_MAX_NEW_TOKENS: int = 64
    GEN_TEMPERATURE: float = 0.7
    GEN_TOP_P: float = 0.9
    GEN_DO_SAMPLE: bool = True

    # ── Experiment Tracking ──────────────────────────────────────────────
    WANDB_PROJECT: str = "cold-start-stylevector"
    WANDB_ENTITY: str = ""               # set in .env or override at runtime

    # ── Reproducibility ──────────────────────────────────────────────────
    SEED: int = 42

    # ── Split Ratios (chronological per-author) ──────────────────────────
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    def ensure_dirs(self) -> None:
        """Create all required directories."""
        for d in [
            self.PROCESSED_DIR, self.SPLITS_DIR, self.MODELS_DIR,
            self.OUTPUTS_DIR, self.LOGS_DIR, self.CONFIGS_DIR,
            self.PROCESSED_DIR / "lamp4",
            self.PROCESSED_DIR / "indian",
            self.OUTPUTS_DIR / "baselines",
            self.OUTPUTS_DIR / "results",
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Serialize to dict for W&B / JSON logging."""
        result = {}
        for k, v in self.__dataclass_fields__.items():
            val = getattr(self, k)
            if isinstance(val, Path):
                result[k] = str(val)
            elif isinstance(val, list):
                result[k] = val
            else:
                result[k] = val
        return result
