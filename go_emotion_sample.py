"""
GoEmotions Anchor Builder
Pulls exemplar sentences per ACM emotion from GoEmotions, filters outliers
via a two-pass semantic coherence check, and writes config/anchors.json.

Two-pass approach:
  Pass 1 — encode a large candidate pool (~200), compute a rough centroid.
  Pass 2 — score every candidate against the rough centroid, keep top-N
            by cosine similarity, discarding semantically incoherent outliers.

Usage:
    pip install datasets requirements.txt
    python go_emotion_sample.py
"""

import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Set

import torch
import torch.nn.functional as F
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from acm import build_acm_from_configs, AffectiveCoherenceMonitor

logging.basicConfig(level=logging.INFO, format="[ACM] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GoEmotions → ACM mapping
# ---------------------------------------------------------------------------
GO_EMOTION_LABELS: List[str] = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

ACM_TO_GO: Dict[str, List[str]] = {
    "joy":        ["joy", "amusement", "excitement", "gratitude",
                   "love", "optimism", "relief", "pride", "admiration", "caring"],
    "sadness":    ["sadness", "grief", "disappointment", "remorse", "embarrassment"],
    "curiosity":  ["curiosity", "realization"],
    "creativity": ["curiosity", "excitement"],
    "fear":       ["fear", "nervousness"],
    "anger":      ["anger", "annoyance", "disapproval"],
    "surprise":   ["surprise", "confusion", "realization"],
    "disgust":    ["disgust", "disapproval"],
}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FINAL_SAMPLES:    int   = 50    # exemplars written to anchors.json per emotion
CANDIDATE_POOL:   int   = 200   # Pass 1 pool size before semantic filtering
RANDOM_SEED:      int   = 42
OUTPUT_PATH:      Path  = Path("config/anchors.json")
MIN_TEXT_LEN:     int   = 20    # characters
MAX_TEXT_LEN:     int   = 200   # characters
MODEL_NAME:       str   = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_label_index(dataset) -> Dict[str, List[str]]:
    """Bucket texts by every GoEmotions label they carry."""
    label_to_texts: Dict[str, List[str]] = {lbl: [] for lbl in GO_EMOTION_LABELS}
    for row in dataset:
        text = row["text"].strip()
        if not (MIN_TEXT_LEN <= len(text) <= MAX_TEXT_LEN):
            continue
        for lbl_id in row["labels"]:
            label_to_texts[GO_EMOTION_LABELS[lbl_id]].append(text)
    for lbl, texts in label_to_texts.items():
        logger.info("GoEmotions '%s': %d candidates", lbl, len(texts))
    return label_to_texts


def pool_candidates(
    go_labels: List[str],
    label_index: Dict[str, List[str]],
    n: int,
    rng: random.Random,
    used: Set[str],
) -> List[str]:
    """Collect up to n unique, unused candidates from the mapped GoEmotions labels."""
    pool: List[str] = []
    for lbl in go_labels:
        for text in label_index.get(lbl, []):
            if text not in used:
                pool.append(text)
    pool = list(dict.fromkeys(pool))   # deduplicate preserving order
    rng.shuffle(pool)
    return pool[:n]


def rough_centroid(texts: List[str], model: SentenceTransformer) -> torch.Tensor:
    """
    Pass 1 — encode all candidates, return the L2-normalised mean embedding
    as a rough centroid for semantic filtering.
    """
    emb = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    emb_n = F.normalize(emb, p=2, dim=1)
    centroid = emb_n.mean(dim=0)
    return F.normalize(centroid, p=2, dim=0)


def semantic_filter(
    texts: List[str],
    model: SentenceTransformer,
    n: int,
    acm_emotion: str,
) -> List[str]:
    """
    Two-pass semantic coherence filter.

    Pass 1: compute a rough centroid over the full candidate pool.
    Pass 2: score every candidate by cosine similarity to that centroid,
            keep the top-n. Outliers (low similarity) are discarded.

    Logs the similarity range so annotation quality is transparent.
    """
    if not texts:
        return []

    # Pass 1 — rough centroid
    centroid = rough_centroid(texts, model)

    # Pass 2 — score each candidate
    emb = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    emb_n = F.normalize(emb, p=2, dim=1)
    scores = torch.mv(emb_n, centroid)          # cosine similarity (n_candidates,)

    # Sort descending by semantic coherence
    ranked_indices = scores.argsort(descending=True).tolist()
    selected = [texts[i] for i in ranked_indices[:n]]

    # Logging: similarity range of kept vs discarded
    kept_scores    = scores[ranked_indices[:n]]
    dropped_scores = scores[ranked_indices[n:]]
    logger.info(
        "'%s' — kept sim: [%.3f – %.3f] | dropped sim: [%.3f – %.3f] | "
        "kept %d / %d candidates",
        acm_emotion,
        kept_scores.min().item(),    kept_scores.max().item(),
        dropped_scores.min().item()  if len(dropped_scores) else float("nan"),
        dropped_scores.max().item()  if len(dropped_scores) else float("nan"),
        len(selected), len(texts),
    )
    return selected


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build_anchors(
    final_samples: int  = FINAL_SAMPLES,
    candidate_pool: int = CANDIDATE_POOL,
    output_path: Path   = OUTPUT_PATH,
    seed: int           = RANDOM_SEED,
) -> Dict[str, List[str]]:
    logger.info("Loading GoEmotions dataset (train split) …")
    dataset = load_dataset(
        "google-research-datasets/go_emotions", "simplified", split="train"
    )
    logger.info("Dataset loaded: %d rows", len(dataset))

    model = SentenceTransformer(MODEL_NAME)
    label_index = build_label_index(dataset)

    rng  = random.Random(seed)
    used: Set[str] = set()
    anchors: Dict[str, List[str]] = {}

    for acm_emotion, go_labels in ACM_TO_GO.items():
        # Gather a larger candidate pool for Pass 1
        candidates = pool_candidates(go_labels, label_index, candidate_pool, rng, used)

        if not candidates:
            logger.warning("'%s': no candidates found.", acm_emotion)
            anchors[acm_emotion] = []
            continue

        # Two-pass semantic filter → final exemplars
        selected = semantic_filter(candidates, model, final_samples, acm_emotion)
        anchors[acm_emotion] = selected
        used.update(selected)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump({"emotional_anchors": anchors}, fh, indent=2, ensure_ascii=False)
    logger.info("Anchors written to %s", output_path)

    return anchors


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    anchors = build_anchors()

    print("\n=== Anchor sample (5 per emotion) ===")
    for emotion, texts in anchors.items():
        print(f"\n[{emotion.upper()}]")
        for t in texts[:5]:
            print(f"  • {t}")

    logger.info("Rebuilding ACM from updated anchors …")
    acm: AffectiveCoherenceMonitor = build_acm_from_configs("config/*.json")
    logger.info("ACM ready with GoEmotions-backed centroids.")