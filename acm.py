"""
acm.py
Affective Coherence Monitoring (ACM) — Full Pipeline
All equations from the paper are implemented and annotated by section/eq number.
"""

import json
import glob
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from collections import deque

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[ACM] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMOTIONS: List[str] = [
    "joy", "sadness", "curiosity", "creativity",
    "fear", "anger", "surprise", "disgust",
]

# §4 Step 1 — per-emotion keyword importance multipliers
KEYWORD_IMPORTANCE: Dict[str, float] = {
    "joy":        0.95,
    "sadness":    0.90,
    "curiosity":  0.85,
    "creativity": 0.88,
    "fear":       0.92,
    "anger":      0.93,
    "surprise":   0.87,
    "disgust":    0.89,
}

KEYWORD_HIT_BASE: float = 0.35   # §3.1.1 — score per matched keyword before importance scaling

# Eq. 1 — fusion weights
W_KW:  float = 0.5
W_SEM: float = 0.5

# Eq. 3 — centroid softmax sharpening temperature
CENTROID_SOFTMAX_TEMP: float = 3.0

# Eq. 4 — valence weights (positive: joy, curiosity, creativity, surprise; negative: sadness, fear, anger, disgust)
VALENCE_W: Dict[str, float] = {
    "joy":        1.0,
    "curiosity":  0.5,
    "creativity": 0.4,
    "surprise":   0.3,
    "sadness":   -1.0,
    "fear":      -0.8,
    "anger":     -0.7,
    "disgust":   -0.9,
}

# Eq. 5 — arousal weights
AROUSAL_W: Dict[str, float] = {
    "joy":        0.70,
    "curiosity":  0.80,
    "creativity": 0.60,
    "sadness":    0.50,
    "fear":       0.90,
    "anger":      0.95,
    "surprise":   0.85,
    "disgust":    0.60,
}

# Eqs. 8–9 — activation sum weights
SPOS_W: Dict[str, float] = {"joy": 1.0, "surprise": 0.4, "curiosity": 0.6}
SNEG_W: Dict[str, float] = {"sadness": 1.0, "fear": 0.9, "anger": 0.95, "disgust": 0.9}

# Eqs. 12–13 — arousal disparity weights
AHIGH_W: Dict[str, float] = {"fear": 1.0, "anger": 1.0, "surprise": 1.0, "joy": 0.7}
ALOW_W:  Dict[str, float] = {"sadness": 1.0, "disgust": 0.5}

# §3.4 — homeostasis
HALF_LIFE: float = 4.0  # t_1/2 in seconds (Eq. 18)

# Table 1 — emotion-specific decay factors d_i
DECAY_FACTORS: Dict[str, float] = {
    "joy":        1.00,
    "sadness":    1.20,
    "curiosity":  0.90,
    "creativity": 0.95,
    "fear":       1.10,
    "anger":      1.15,
    "surprise":   0.80,
    "disgust":    1.05,
}

# Eq. 19 — state integration blend weights
BLEND_CURRENT: float = 0.4
BLEND_NEW:     float = 0.6

# §3.5 — habituation
HABITUATION_WINDOW: float = 30.0  # seconds
HABITUATION_THRESH: float = 5.0   # T_h (Eq. 20)

MAX_ACM_RETRIES: int = 2  # §4 Step 9


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def load_configs(pattern: str = "config/*.json") -> Dict:
    merged: Dict = {}
    for path in glob.glob(pattern):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for k, v in data.items():
                if k in merged:
                    logger.warning("Duplicate key '%s' in %s — overwriting.", k, path)
                merged[k] = v
            logger.info("Loaded %s", path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load %s: %s", path, exc)
    return merged


# ---------------------------------------------------------------------------
# Centroid generator  (§3.1.2 – §3.1.3)
# ---------------------------------------------------------------------------
class EmotionalCentroidGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info("Loading SentenceTransformer: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.dim: int = self.model.get_sentence_embedding_dimension()
        logger.info("Model ready. Embedding dim: %d", self.dim)

    def compute_centroid(self, sentences: List[str]) -> torch.Tensor:
        """
        Eq. 3 — weighted mean of exemplar embeddings, L2-normalised.
        Centrality score s̄_i = mean pairwise cosine similarity of L2-normalised
        embeddings.  Softmax with temperature 3.0 sharpens toward central exemplars.
        Weighted sum is computed on RAW (un-normalised) embeddings as per the paper,
        then the result is L2-normalised.
        """
        if not sentences:
            return torch.zeros(self.dim)

        emb = self.model.encode(
            sentences, convert_to_tensor=True, show_progress_bar=False
        )                                                   # (n, d) raw
        emb_n = F.normalize(emb, p=2, dim=1)               # (n, d) unit-norm for similarity

        # s̄_i = mean pairwise cosine similarity (centrality)
        sim = torch.mm(emb_n, emb_n.T)                     # (n, n)
        s_bar = sim.mean(dim=1)                             # (n,)

        # w_i = softmax(s̄_i * 3.0)
        weights = torch.softmax(s_bar * CENTROID_SOFTMAX_TEMP, dim=0)  # (n,)

        # c̃_e = weighted sum over RAW embeddings / sum(w_i)  [sum(w_i)==1 after softmax]
        c_tilde = (emb * weights.unsqueeze(1)).sum(dim=0)  # (d,)

        # c_e = c̃_e / ||c̃_e||_2
        return F.normalize(c_tilde, p=2, dim=0)

    def encode_text(self, text: str) -> torch.Tensor:
        """Return L2-normalised embedding for a single input string."""
        vec = self.model.encode(text, convert_to_tensor=True, show_progress_bar=False)
        return F.normalize(vec, p=2, dim=0)

    def build_centroids(self, anchors: Dict[str, List[str]]) -> Dict[str, torch.Tensor]:
        centroids: Dict[str, torch.Tensor] = {}
        for emotion, sentences in anchors.items():
            if emotion not in EMOTIONS:
                logger.debug("Skipping unknown anchor key: %s", emotion)
                continue
            centroids[emotion] = self.compute_centroid(sentences)
            logger.info("Centroid built for '%s' (%d exemplars)", emotion, len(sentences))
        for e in EMOTIONS:
            if e not in centroids:
                logger.warning("No exemplars for '%s'; using zero centroid.", e)
                centroids[e] = torch.zeros(self.dim)
        return centroids


# ---------------------------------------------------------------------------
# Keyword scorer  (§3.1.1)
# ---------------------------------------------------------------------------
class KeywordScorer:
    def __init__(self, emotion_words: Dict[str, List[str]]):
        # Keys in lexica.json are canonical ACM emotion names; skip anything else (e.g. "general")
        self.lookup: Dict[str, str] = {}
        for emotion, words in emotion_words.items():
            if emotion not in EMOTIONS:
                continue
            for w in words:
                self.lookup[w.lower()] = emotion

    def score(self, text: str) -> Dict[str, float]:
        """
        s_kw,e = sum of (KEYWORD_HIT_BASE * importance_e) for each matched token.
        Clamped to [0, 1].
        """
        scores: Dict[str, float] = {e: 0.0 for e in EMOTIONS}
        for token in text.lower().split():
            token = token.strip(".,!?;:\"'()")
            if token in self.lookup:
                emotion = self.lookup[token]
                scores[emotion] += KEYWORD_HIT_BASE * KEYWORD_IMPORTANCE[emotion]
        return {e: min(v, 1.0) for e, v in scores.items()}


# ---------------------------------------------------------------------------
# ACM conversational state
# ---------------------------------------------------------------------------
@dataclass
class ACMState:
    # E_t — current emotional state vector (Eq. 17, 19)
    emotional_state: Dict[str, float] = field(
        default_factory=lambda: {e: 0.0 for e in EMOTIONS}
    )
    last_update_time: float = field(default_factory=time.monotonic)
    # habituation sliding window: deque of (timestamp, emotion) tuples
    activation_log: deque = field(default_factory=deque)


# ---------------------------------------------------------------------------
# Core ACM engine
# ---------------------------------------------------------------------------
class AffectiveCoherenceMonitor:

    def __init__(
        self,
        centroids: Dict[str, torch.Tensor],
        keyword_scorer: KeywordScorer,
        encoder: EmotionalCentroidGenerator,
    ):
        self.centroids = centroids
        self.kw_scorer = keyword_scorer
        self.encoder = encoder
        self.state = ACMState()

    # ------------------------------------------------------------------
    # Eq. 1 — Emotion fusion
    # ------------------------------------------------------------------
    def _fuse(
        self, skw: Dict[str, float], ssem: Dict[str, float]
    ) -> Dict[str, float]:
        return {e: W_KW * skw[e] + W_SEM * ssem[e] for e in EMOTIONS}

    # ------------------------------------------------------------------
    # Eq. 2 — Semantic similarity scores
    # ------------------------------------------------------------------
    def _semantic_scores(self, text: str) -> Dict[str, float]:
        """
        s_sem,e = (v_text · c_e + 1.0) / 2.0
        Both v_text and c_e are L2-normalised, so the dot product is cosine
        similarity in [-1, 1], mapped to [0, 1].
        """
        v = self.encoder.encode_text(text)
        scores: Dict[str, float] = {}
        for e in EMOTIONS:
            cos_sim = torch.dot(v, self.centroids[e]).item()
            scores[e] = (cos_sim + 1.0) / 2.0
        return scores

    # ------------------------------------------------------------------
    # Eqs. 6–7 — Cross-emotion modulation
    # ------------------------------------------------------------------
    @staticmethod
    def _modulate(s: Dict[str, float]) -> Dict[str, float]:
        m = dict(s)

        # Eq. 6 — curiosity modulation
        if s["joy"]       > 0.4: m["curiosity"] += 0.40 * s["joy"]
        if s["sadness"]   > 0.4: m["curiosity"] -= 0.50 * s["sadness"]
        if s["fear"]      > 0.4: m["curiosity"] -= 0.20 * s["fear"]
        if s["anger"]     > 0.4: m["curiosity"] -= 0.30 * s["anger"]
        if s["surprise"]  > 0.5: m["curiosity"] += 0.30 * s["surprise"]
        if s["disgust"]   > 0.4: m["curiosity"] -= 0.25 * s["disgust"]

        # Eq. 7 — creativity modulation
        if s["joy"]        > 0.4: m["creativity"] += 0.45 * s["joy"]
        if s["sadness"]    > 0.4: m["creativity"] -= 0.40 * s["sadness"]
        if s["fear"]       > 0.4: m["creativity"] -= 0.30 * s["fear"]
        if s["disgust"]    > 0.4: m["creativity"] -= 0.20 * s["disgust"]
        if s["curiosity"]  > 0.6: m["creativity"] += 0.20

        return {e: max(0.0, min(1.0, v)) for e, v in m.items()}

    # ------------------------------------------------------------------
    # Eqs. 4–5 — Valence and Arousal
    # ------------------------------------------------------------------
    @staticmethod
    def _valence_arousal(s: Dict[str, float]) -> Tuple[float, float]:
        # Eq. 4
        V = (
            s["joy"]        * VALENCE_W["joy"]
            + s["curiosity"]  * VALENCE_W["curiosity"]
            + s["creativity"] * VALENCE_W["creativity"]
            + s["surprise"]   * VALENCE_W["surprise"]
            - s["sadness"]    * abs(VALENCE_W["sadness"])
            - s["fear"]       * abs(VALENCE_W["fear"])
            - s["anger"]      * abs(VALENCE_W["anger"])
            - s["disgust"]    * abs(VALENCE_W["disgust"])
        )
        # Eq. 5
        A = (
            s["joy"]        * AROUSAL_W["joy"]
            + s["curiosity"]  * AROUSAL_W["curiosity"]
            + s["creativity"] * AROUSAL_W["creativity"]
            + s["sadness"]    * AROUSAL_W["sadness"]
            + s["fear"]       * AROUSAL_W["fear"]
            + s["anger"]      * AROUSAL_W["anger"]
            + s["surprise"]   * AROUSAL_W["surprise"]
            + s["disgust"]    * AROUSAL_W["disgust"]
        )
        return max(-1.0, min(1.0, V)), max(0.0, min(1.0, A))

    # ------------------------------------------------------------------
    # Eqs. 8–16 — Context mismatch detection
    # ------------------------------------------------------------------
    @staticmethod
    def _mismatch(s: Dict[str, float]) -> Dict:
        # Eqs. 8–10 — activation sums
        Spos   = sum(SPOS_W.get(e, 0.0) * s[e] for e in EMOTIONS)   # Eq. 8
        Sneg   = sum(SNEG_W.get(e, 0.0) * s[e] for e in EMOTIONS)   # Eq. 9
        Stotal = Spos + Sneg                                          # Eq. 10

        # Eqs. 12–13 — arousal sub-sums
        Ahigh = sum(AHIGH_W.get(e, 0.0) * s[e] for e in EMOTIONS)   # Eq. 12
        Alow  = sum(ALOW_W.get(e,  0.0) * s[e] for e in EMOTIONS)   # Eq. 13

        # Eq. 11 — valence conflict
        VC = (min(Spos, Sneg) / (Stotal / 2.0)) if Stotal > 0.0 else 0.0

        # Eq. 14 — arousal disparity
        AD = abs(Ahigh - Alow) / max(Ahigh + Alow, 0.1)

        # Eq. 15 — mismatch score
        if Stotal == 0.0:
            MS = 0.0
        else:
            MS = max(-1.0, min(1.0, ((Spos - Sneg) / Stotal) * (1.0 + 0.5 * VC)))

        # Eq. 16 — confidence metric
        C = min(Stotal * VC * 1.5, 1.0)

        return {
            "Spos":   Spos,
            "Sneg":   Sneg,
            "Stotal": Stotal,
            "Ahigh":  Ahigh,
            "Alow":   Alow,
            "VC":     VC,
            "AD":     AD,
            "MS":     MS,
            "C":      C,
        }

    # ------------------------------------------------------------------
    # §3.6 — Affective state type classification
    # ------------------------------------------------------------------
    @staticmethod
    def _state_type(MS: float, AD: float) -> str:
        if AD   >  0.6:  return "emotional_suppression"
        if MS   < -0.15: return "frustrated_sarcasm"
        if MS   >  0.15: return "playful_sarcasm"
        return "congruent"

    # ------------------------------------------------------------------
    # Eq. 22 — Alert level classification
    # ------------------------------------------------------------------
    @staticmethod
    def _alert_level(MS: float, VC: float, st: str) -> str:
        if MS >= 0.60 or st == "emotional_suppression":
            return "Alert"
        if MS >= 0.35 or VC >= 0.40:
            return "Watch"
        return "Nominal"

    # ------------------------------------------------------------------
    # Eq. 23 — Temperature modulation
    # ------------------------------------------------------------------
    @staticmethod
    def _temperature(alert: str, st: str) -> float:
        if alert == "Alert":
            return 0.3
        if alert == "Watch" and st == "frustrated_sarcasm":
            return 0.4
        if alert == "Watch":
            return 0.5
        return 0.7

    # ------------------------------------------------------------------
    # Eqs. 17–19 — Homeostatic decay
    # ------------------------------------------------------------------
    def _apply_decay(self, pulse: Dict[str, float]) -> None:
        """
        Eq. 17: E_{t+Δt} = E_t · e^{-λ·Δt}
        Eq. 18: λ_i = d_i · ln2 / t_{1/2}
        Eq. 19: E_new = clamp[0,1](0.4 · E_decayed + 0.6 · P_new)
        """
        now = time.monotonic()
        dt = now - self.state.last_update_time
        self.state.last_update_time = now

        for e in EMOTIONS:
            lam = DECAY_FACTORS[e] * math.log(2) / HALF_LIFE      # Eq. 18
            decayed = self.state.emotional_state[e] * math.exp(-lam * dt)  # Eq. 17
            blended = BLEND_CURRENT * decayed + BLEND_NEW * pulse[e]       # Eq. 19
            self.state.emotional_state[e] = max(0.0, min(1.0, blended))

    # ------------------------------------------------------------------
    # Eqs. 20–21 — Habituation dynamics
    # ------------------------------------------------------------------
    def _apply_habituation(self, s: Dict[str, float]) -> Dict[str, float]:
        """
        Eq. 20: H_score = (N_e · |s_e − E_current,e|) / T_h
        Eq. 21: piecewise attenuation based on H_score magnitude.
        Note: E_current,e is read BEFORE _apply_decay mutates the state,
        so this method must be called before _apply_decay in the pipeline.
        """
        now = time.monotonic()
        cutoff = now - HABITUATION_WINDOW

        while self.state.activation_log and self.state.activation_log[0][0] < cutoff:
            self.state.activation_log.popleft()

        result: Dict[str, float] = {}
        for e in EMOTIONS:
            # N_e — activations in the 30-second window
            Ne = sum(1 for (_, em) in self.state.activation_log if em == e)

            # Eq. 20 — habituation score
            Hscore = (Ne * abs(s[e] - self.state.emotional_state[e])) / HABITUATION_THRESH

            # Eq. 21 — piecewise attenuation
            if Hscore > 1.0:
                attenuated = min(s[e] / (1.0 + Hscore), 0.5 * s[e])
            elif Hscore > 0.5:
                attenuated = s[e] * (1.0 - 0.3 * Hscore)
            else:
                attenuated = s[e]

            result[e] = max(0.0, min(1.0, attenuated))

            if s[e] > 0.0:
                self.state.activation_log.append((now, e))

        return result

    # ------------------------------------------------------------------
    # Public API — full pipeline (§4 Steps 1–9)
    # ------------------------------------------------------------------
    def process(self, text: str) -> Dict:
        # Step 1 — keyword + semantic scoring (Eqs. 1–2)
        skw  = self.kw_scorer.score(text)
        ssem = self._semantic_scores(text)

        # Step 2 — fusion (Eq. 1) + cross-modulation (Eqs. 6–7)
        fused     = self._fuse(skw, ssem)
        modulated = self._modulate(fused)

        # Step 3 — dimensional computation (Eqs. 4–5)
        valence, arousal = self._valence_arousal(modulated)

        # Step 4 — mismatch detection (Eqs. 8–16)
        mismatch = self._mismatch(modulated)
        MS, VC, AD = mismatch["MS"], mismatch["VC"], mismatch["AD"]

        # Step 6 — habituation BEFORE decay so E_current is the pre-decay state (Eqs. 20–21)
        habituated = self._apply_habituation(modulated)

        # Step 5 — homeostatic decay + integration (Eqs. 17–19)
        self._apply_decay(habituated)

        # Step 7 — alert classification (Eq. 22)
        st    = self._state_type(MS, AD)
        alert = self._alert_level(MS, VC, st)

        # Step 8 — temperature modulation (Eq. 23)
        T_gen = self._temperature(alert, st)

        return {
            "text": text,
            "scores": {
                "keyword":    skw,
                "semantic":   ssem,
                "fused":      fused,
                "modulated":  modulated,
                "habituated": habituated,
            },
            "dimensions":      {"valence": valence, "arousal": arousal},
            "mismatch":        mismatch,    # includes Spos, Sneg, Stotal, Ahigh, Alow, VC, AD, MS, C
            "state_type":      st,
            "alert_level":     alert,
            "temperature":     T_gen,
            "emotional_state": dict(self.state.emotional_state),
        }

    def reset_state(self) -> None:
        self.state = ACMState()


# ---------------------------------------------------------------------------
# Batch processor
# ---------------------------------------------------------------------------
class ACMBatchProcessor:
    def __init__(self, acm: AffectiveCoherenceMonitor):
        self.acm = acm

    def process_batch(self, texts: List[str], reset_between: bool = True) -> List[Dict]:
        results = []
        for text in texts:
            if reset_between:
                self.acm.reset_state()
            results.append(self.acm.process(text))
        return results

    def summarise(self, results: List[Dict]) -> Dict:
        alerts      = [r["alert_level"] for r in results]
        temps       = [r["temperature"]  for r in results]
        state_types = [r["state_type"]   for r in results]
        n = len(results)
        return {
            "n": n,
            "alert_counts": {
                lvl: alerts.count(lvl) for lvl in ("Alert", "Watch", "Nominal")
            },
            "state_type_counts": {
                st: state_types.count(st)
                for st in ("congruent", "frustrated_sarcasm",
                           "playful_sarcasm", "emotional_suppression")
            },
            "mean_temperature": sum(temps) / n if n else 0.0,
            "mean_valence":     sum(r["dimensions"]["valence"] for r in results) / n,
            "mean_arousal":     sum(r["dimensions"]["arousal"]  for r in results) / n,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_acm_from_configs(config_pattern: str = "config/*.json") -> AffectiveCoherenceMonitor:
    cfg = load_configs(config_pattern)
    anchors       = cfg.get("emotional_anchors", {})
    emotion_words = cfg.get("emotion_words", {})
    if not anchors:
        logger.warning("No 'emotional_anchors' found in configs.")
    if not emotion_words:
        logger.warning("No 'emotion_words' found in configs.")
    generator = EmotionalCentroidGenerator()
    centroids = generator.build_centroids(anchors)
    kw_scorer = KeywordScorer(emotion_words)
    return AffectiveCoherenceMonitor(centroids, kw_scorer, generator)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pprint

    acm = build_acm_from_configs("config/*.json")
    processor = ACMBatchProcessor(acm)
#Source: GoEmotions dataset
    sample_texts = [
        "My favourite food is anything I didn't have to cook myself.",
        "Yes I heard abt the f bombs! That has to be why. Thanks for your reply:) until then hubby and I will anxiously wait 😝",
        "Damn youtube and outrage drama is super lucrative for reddit",
        "I'm curious about what we haven't discovered yet.",
        "I feel alone and empty inside, but everything is fine I guess.",
    ]

    results = processor.process_batch(sample_texts, reset_between=False)
    for r in results:
        # Dominant emotion = highest score in the habituated scores
        habituated = r["scores"]["habituated"]
        dominant_emotion = max(habituated, key=habituated.get)
        dominant_score   = habituated[dominant_emotion]

        print(f"\n--- '{r['text'][:60]}' ---")
        print(f"  Classified as: {dominant_emotion.upper()} (score: {dominant_score:.3f})")
        print(f"  Alert:   {r['alert_level']:8s} | State: {r['state_type']}")
        print(f"  Valence: {r['dimensions']['valence']:+.3f}    | Arousal: {r['dimensions']['arousal']:.3f}")
        print(f"  T_gen:   {r['temperature']}          | MS: {r['mismatch']['MS']:+.3f} | VC: {r['mismatch']['VC']:.3f} | C: {r['mismatch']['C']:.3f}")
        print(f"  Ahigh:   {r['mismatch']['Ahigh']:.3f}        | Alow: {r['mismatch']['Alow']:.3f} | AD: {r['mismatch']['AD']:.3f}")
        print(f"  All scores: { {e: f'{v:.3f}' for e, v in sorted(habituated.items(), key=lambda x: -x[1])} }")

    print("\n=== Batch Summary ===")
    pprint.pprint(processor.summarise(results))