"""
src/config.py — Single source of truth for all hyperparameters and paths.
=========================================================================
Nested dataclasses: PathConfig, DataConfig, ModelConfig, TrainingConfig,
RAGConfig, EvalConfig, composed into a top-level Config.

USAGE:
    from src.config import get_config
    cfg = get_config()
    print(cfg.paths.raw_dir, cfg.model.base_model)

The get_config() call auto-creates all directories.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
import json
import dataclasses


@dataclass
class PathConfig:
    project_root: Path = Path(__file__).resolve().parent.parent

    @property
    def raw_dir(self) -> Path:
        return self.project_root / "data" / "raw" / "indian_news"

    @property
    def lamp4_dir(self) -> Path:
        return self.project_root / "data" / "raw" / "LaMP_4"

    @property
    def processed_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def indian_processed_dir(self) -> Path:
        return self.processed_dir / "indian"

    @property
    def lamp4_processed_dir(self) -> Path:
        return self.processed_dir / "lamp4"

    @property
    def interim_dir(self) -> Path:
        return self.project_root / "data" / "interim"

    @property
    def vectors_dir(self) -> Path:
        return self.project_root / "author_vectors"

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"

    @property
    def splits_dir(self) -> Path:
        return self.project_root / "data" / "splits"

    # Canonical Indian split paths (underscore author_ids, article_body field)
    @property
    def indian_train_jsonl(self) -> Path:
        return self.splits_dir / "indian_train.jsonl"

    @property
    def indian_val_jsonl(self) -> Path:
        return self.splits_dir / "indian_val.jsonl"

    @property
    def indian_test_jsonl(self) -> Path:
        return self.splits_dir / "indian_test.jsonl"


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
    min_headline_chars: int = 10
    max_headline_chars: int = 200
    min_articles_per_author: int = 5


@dataclass
class ModelConfig:
    base_model: str = "models/Llama-3.1-8B-Instruct"
    # LoRA config — bf16, attention-only, no quantization
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ])
    # StyleVector config
    extraction_layer_range: Tuple[int, int] = (15, 28)
    extraction_layers: List[int] = field(default_factory=lambda: [15, 18, 21, 24, 27])
    pca_components: int = 50
    kmeans_k_range: Tuple[int, int] = (5, 20)
    alpha_range: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    intervention_alpha: float = 0.7
    max_new_tokens_headline: int = 30
    max_new_tokens_agnostic: int = 40
    # LaMP-4 cluster pool caps (V4.2 spec)
    lamp4_max_users: int = 500           # max rich users for SV extraction
    lamp4_max_profile_articles: int = 100  # max articles per user
    # Set after Phase 2C and Phase 3 respectively — None means "not yet determined"
    best_layer: int = 15                  # locked after Phase 2C layer sweep (2026-04-15)
    best_alpha: float = 0.6             # locked after Phase 3 alpha sweep (2026-04-15)


@dataclass
class TrainingConfig:
    batch_size: int = 4
    grad_accumulation: int = 8
    learning_rate: float = 2e-4
    num_epochs: int = 7
    early_stopping_patience: int = 2  # Stop if val loss rises 2 consecutive epochs
    save_steps: int = 500
    warmup_steps: int = 100
    max_seq_length: int = 1024
    max_train_samples: int = 25000
    hf_repo_id: str = "dharmik-mistry/cold-start-stylevector"
    seed: int = 42


@dataclass
class RAGConfig:
    k_retrieved: int = 2
    max_article_words_in_context: int = 150
    max_new_tokens: int = 30


@dataclass
class EvalConfig:
    metrics: List[str] = field(default_factory=lambda: ["rouge_l", "meteor"])
    paper_rouge_l: float = 0.0411
    paper_meteor: float = 0.0809


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def get_config() -> Config:
    """Return Config and create all directories."""
    cfg = Config()
    dirs = [
        cfg.paths.processed_dir,
        cfg.paths.indian_processed_dir,
        cfg.paths.lamp4_processed_dir,
        cfg.paths.interim_dir,
        cfg.paths.vectors_dir,
        cfg.paths.outputs_dir,
        cfg.paths.outputs_dir / "baselines",
        cfg.paths.outputs_dir / "evaluation",
        cfg.paths.outputs_dir / "stylevector",
        cfg.paths.outputs_dir / "cold_start",
        cfg.paths.models_dir,
        cfg.paths.logs_dir,
        cfg.paths.logs_dir / "gpu_tracking",
        cfg.paths.splits_dir,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return cfg


def _config_to_dict(obj) -> dict:
    """Recursively serialize config to dict, handling Path objects."""
    if dataclasses.is_dataclass(obj):
        result = {}
        for f in dataclasses.fields(obj):
            val = getattr(obj, f.name)
            result[f.name] = _config_to_dict(val)
        return result
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, (list, tuple)):
        return [_config_to_dict(v) for v in obj]
    return obj


if __name__ == "__main__":
    cfg = get_config()
    print(json.dumps(_config_to_dict(cfg), indent=2))
