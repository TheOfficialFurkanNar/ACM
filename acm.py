"""
acm.py
Affective Coherence Monitoring (ACM) — Full Pipeline
Architecture mirrors the working agentic version.
All equations from the paper are retained and annotated.
"""

import re
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
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

KEYWORD_HIT_BASE: float = 0.35

# Eq. 1 — fusion weights (shifted toward keywords given semantic compression)
W_KW:  float = 0.70
W_SEM: float = 0.30

# Eq. 3 — centroid softmax sharpening temperature
CENTROID_SOFTMAX_TEMP: float = 3.0

# Eq. 4 — valence weights
VALENCE_W: Dict[str, float] = {
    "joy":        1.0,  "curiosity":  0.5,  "creativity": 0.4,  "surprise":   0.3,
    "sadness":   -1.0,  "fear":      -0.8,  "anger":     -0.7,  "disgust":   -0.9,
}

# Eq. 5 — arousal weights (reduced to prevent ceiling saturation)
AROUSAL_W: Dict[str, float] = {
    "joy":        0.45, "curiosity":  0.45, "creativity": 0.35, "sadness":    0.50,
    "fear":       0.80, "anger":      0.85, "surprise":   0.70, "disgust":    0.50,
}

# Eqs. 8–9 — activation sum weights (negative weights reduced to ease MS pressure)
SPOS_W: Dict[str, float] = {"joy": 1.0, "surprise": 0.4, "curiosity": 0.6}
SNEG_W: Dict[str, float] = {"sadness": 1.0, "fear": 0.80, "anger": 0.80, "disgust": 0.75}

# Eqs. 12–13 — arousal disparity weights
AHIGH_W: Dict[str, float] = {"fear": 1.0, "anger": 1.0, "surprise": 1.0, "joy": 0.7}
ALOW_W:  Dict[str, float] = {"sadness": 1.0, "disgust": 0.5}

# §3.4 — homeostasis
HALF_LIFE: float = 60.0  # raised from 4s to 60s for conversational context

# Table 1 — emotion-specific decay factors d_i
DECAY_FACTORS: Dict[str, float] = {
    "joy":        1.00, "sadness":    1.20, "curiosity":  0.90, "creativity": 0.95,
    "fear":       1.10, "anger":      1.15, "surprise":   0.80, "disgust":    1.05,
}

# Eq. 19 — state integration blend weights
BLEND_CURRENT: float = 0.4
BLEND_NEW:     float = 0.6

# §3.5 — habituation
HABITUATION_WINDOW:    float = 30.0
HABITUATION_THRESHOLD: int   = 5
HABITUATION_THRESH:    float = 5.0  # T_h (Eq. 20)

MAX_ACM_RETRIES: int = 2  # §4 Step 9

# Semantic rescaling floor — cosine similarities below this treated as noise
SEM_FLOOR: float = 0.15

# Sarcasm / mismatch detection thresholds (tightened)
VC_THRESHOLD:  float = 0.55   # valence conflict gate  (was 0.40)
C_THRESHOLD:   float = 0.50   # confidence gate        (was 0.30)
MS_FRUSTRATED: float = -0.35  # frustrated_sarcasm MS  (was -0.15)
MS_PLAYFUL:    float =  0.25  # playful_sarcasm MS     (was  0.15)

# ---------------------------------------------------------------------------
# Inline anchor sentences (§3.1.3 — centroid exemplars)
# ---------------------------------------------------------------------------
ANCHORS: Dict[str, List[str]] = {
    "joy": [
        "I feel so happy and content right now.", "This brings me such joy and warmth.",
        "I'm filled with happiness today.", "My heart feels light and cheerful.",
        "Everything feels wonderful at this moment.", "I'm experiencing pure bliss and peace.",
        "This makes me smile with delight.", "I feel blessed and fortunate.",
        "Life is beautiful and I'm grateful.", "I'm so pleased with how things are going.",
        "My mood is excellent and positive.", "I feel warm and happy inside.",
        "This fills my heart with joy.", "I'm enjoying this so much.",
        "I feel completely satisfied and happy.", "Everything is going well and I'm glad.",
        "I'm in such a good mood today.", "This makes me feel so good.",
        "I'm happy and at peace.", "I feel joyful and content.",
    ],
    "sadness": [
        "I feel so alone and empty inside.", "My heart is heavy with grief.",
        "Nothing brings me joy anymore.", "I miss them terribly every day.",
        "Life feels meaningless and gray.", "I can't stop crying about the loss.",
        "The sadness is overwhelming me.", "I feel broken and defeated.",
        "Everything reminds me of what I lost.", "I'm drowning in sorrow and pain.",
        "The loneliness is unbearable.", "I feel hopeless about the future.",
        "My world has lost all its color.", "I'm struggling with deep depression.",
        "This emptiness inside won't go away.", "I feel abandoned and forgotten.",
        "The pain of loss is too much to bear.", "I'm sad beyond words.",
        "Nothing matters without them here.", "I feel hollow and numb inside.",
    ],
    "curiosity": [
        "Why does water boil at 100 degrees Celsius?",
        "How do neurons transmit electrical signals?",
        "What causes the greenhouse effect?",
        "Can you explain how photosynthesis works?",
        "What is the definition of quantum entanglement?",
        "How does the immune system fight infections?",
        "What is the mechanism behind DNA replication?",
        "How is electricity generated in power plants?",
        "I wonder what lies at the edge of the universe.",
        "What mysteries remain unsolved in science?",
        "I'm curious about what we haven't discovered yet.",
        "What secrets does the ocean floor hold?",
        "I wonder how the first life forms emerged.",
        "What will we learn about the cosmos in the future?",
        "I don't understand this concept at all.",
        "I need clarification on this topic.",
        "I'm confused about how this process operates.",
        "Could you help me understand this theory?",
        "I need more information to understand this.",
        "This doesn't make sense to me yet.",
    ],
    "creativity": [
        "I'm going to design something completely original.",
        "Let me create a unique solution to this problem.",
        "I'll invent a new approach from scratch.",
        "I'm crafting an artistic piece right now.",
        "Let me build something that's never existed before.",
        "I'm composing an original musical work.",
        "I'll develop an innovative product design.",
        "I'm painting an imaginative abstract piece.",
        "Let me generate some fresh creative ideas.",
        "I'm writing a novel with unique characters.",
        "I'll devise an unconventional strategy.",
        "I'm sculpting an original art installation.",
        "Let me architect an innovative system.",
        "I'm designing a creative brand identity.",
        "I'll construct something imaginative and new.",
        "I'm producing an original film concept.",
        "Let me craft a unique marketing campaign.",
        "I'm inventing a new game mechanic.",
        "I'll create an innovative user interface.",
        "I'm developing an original software tool.",
    ],
    "fear": [
        "I'm terrified something bad will happen.",
        "This situation fills me with dread and anxiety.",
        "I'm scared and don't know what to do.",
        "My heart races with fear and panic.",
        "I feel threatened and unsafe right now.",
        "The uncertainty makes me very anxious.",
        "I'm worried this will end badly.",
        "I feel vulnerable and frightened.",
        "This danger is making me panic.",
        "I'm afraid of what might happen next.",
        "The risk is too high and I'm scared.",
        "I feel a sense of impending doom.",
        "This makes me nervous and jittery.",
        "I'm frightened by these developments.",
        "My anxiety is through the roof.",
        "I'm terrified of the consequences.",
        "This threat is overwhelming me.",
        "I feel paralyzed by fear.",
        "I'm worried sick about this.",
        "The danger makes me want to run away.",
    ],
    "anger": [
        "This makes me so angry and frustrated.",
        "I'm furious about this injustice.",
        "I feel rage building up inside me.",
        "This is completely unacceptable and infuriating.",
        "I'm outraged by this behavior.",
        "How dare they treat me this way!",
        "I'm fed up with this situation.",
        "This makes my blood boil with anger.",
        "I feel intense frustration and hostility.",
        "I'm mad and want to confront this.",
        "This unfairness makes me livid.",
        "I'm irritated beyond belief.",
        "I feel aggressive and confrontational.",
        "This disrespect enrages me.",
        "I'm angry at being treated poorly.",
        "My patience has run out completely.",
        "I feel resentment and bitterness.",
        "This provokes my anger deeply.",
        "I'm indignant about this offense.",
        "I want to express my fury about this.",
    ],
    "surprise": [
        "Wow, I didn't expect that at all!",
        "This is completely unexpected and shocking.",
        "I'm amazed by this revelation.",
        "What a surprise this turned out to be!",
        "I never saw this coming.",
        "This caught me completely off guard.",
        "I'm stunned by this development.",
        "This is astonishing and unprecedented.",
        "I can't believe this just happened!",
        "This outcome surprised everyone.",
        "What an unexpected turn of events!",
        "I'm taken aback by this news.",
        "This revelation is mind-blowing.",
        "I'm shocked and bewildered.",
        "This is a surprising discovery.",
        "I didn't anticipate this at all.",
        "This unexpected result amazes me.",
        "I'm startled by this information.",
        "This is remarkably surprising.",
        "What an unforeseen development!",
    ],
    "disgust": [
        "This is revolting and makes me sick.",
        "I find this absolutely disgusting.",
        "This behavior is repulsive to me.",
        "I feel nauseated by this situation.",
        "This is gross and offensive.",
        "I'm repelled by this display.",
        "This makes me feel contaminated.",
        "I find this morally repugnant.",
        "This disgusts me on every level.",
        "I feel revulsion toward this.",
        "This is distasteful and vile.",
        "I'm appalled by this conduct.",
        "This nauseating situation bothers me.",
        "I find this deeply offensive.",
        "This repulsive act disturbs me.",
        "I feel aversion to this practice.",
        "This is sickening and unacceptable.",
        "I'm disgusted by this treatment.",
        "This deplorable behavior repels me.",
        "I find this utterly distasteful.",
    ],
}

# ---------------------------------------------------------------------------
# Inline keyword lists (§3.1.1)
# ---------------------------------------------------------------------------
KEYWORDS: Dict[str, List[str]] = {
    "joy":        ["happy", "joy", "excited", "love", "wonderful", "great", "amazing",
                   "fantastic", "delighted", "cheerful", "pleased", "glad", "thrilled"],
    "sadness":    ["sad", "unhappy", "depressed", "miserable", "grief", "sorrow",
                   "heartbroken", "devastated", "lonely", "empty", "crying", "tears"],
    "curiosity":  ["curious", "wonder", "why", "what if", "explore", "discover",
                   "investigate", "question", "intrigued", "fascinated", "interested"],
    "creativity": ["create", "imagine", "design", "innovative", "artistic", "original",
                   "inventive", "craft", "build", "compose", "brainstorm", "envision"],
    "fear":       ["fear", "scared", "afraid", "terrified", "anxious", "worried", "panic",
                   "frightened", "dread", "nervous", "threat", "danger", "risk"],
    "anger":      ["angry", "mad", "furious", "rage", "hate", "frustrated", "irritated",
                   "outraged", "livid", "annoyed", "resentful", "hostile", "aggressive",
                   "infuriated"],
    "surprise":   ["surprise", "shocked", "unexpected", "amazed", "astonished", "startled",
                   "stunned", "bewildered", "wow", "unbelievable", "unforeseen", "sudden"],
    "disgust":    ["disgust", "gross", "revolting", "repulsive", "nauseating", "sickening",
                   "vile", "offensive", "repellent", "distasteful", "appalling", "foul"],
}

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
        """Eq. 3 — weighted mean of RAW exemplar embeddings, L2-normalised."""
        if not sentences:
            return torch.zeros(self.dim)
        emb   = self.model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
        emb_n = F.normalize(emb, p=2, dim=1)
        sim   = torch.mm(emb_n, emb_n.T)
        s_bar = sim.mean(dim=1)
        weights  = torch.softmax(s_bar * CENTROID_SOFTMAX_TEMP, dim=0)
        c_tilde  = (emb * weights.unsqueeze(1)).sum(dim=0)
        return F.normalize(c_tilde, p=2, dim=0)

    def encode(self, text: str) -> torch.Tensor:
        """Return RAW (un-normalised) embedding — normalisation done at query time."""
        return self.model.encode(text, convert_to_tensor=True, show_progress_bar=False)

    def build_centroids(self) -> Dict[str, torch.Tensor]:
        centroids: Dict[str, torch.Tensor] = {}
        for emotion, sentences in ANCHORS.items():
            centroids[emotion] = self.compute_centroid(sentences)
            logger.info("Centroid built for '%s' (%d exemplars)", emotion, len(sentences))
        return centroids


# ---------------------------------------------------------------------------
# Emotional engine  (scoring + modulation + valence/arousal)
# ---------------------------------------------------------------------------
class EmotionalEngine:
    def __init__(self):
        self.valence: float = 0.0
        self.arousal: float = 0.0
        self.generator = EmotionalCentroidGenerator()
        logger.info("Computing emotional centroids...")
        self.centroids = self.generator.build_centroids()
        logger.info("All centroids ready.")

    # ------------------------------------------------------------------
    # §3.1.1 — Keyword scoring (regex word-boundary matching)
    # ------------------------------------------------------------------
    def _keyword_score(self, text: str) -> Dict[str, float]:
        scores   = {e: 0.0 for e in EMOTIONS}
        text_str = " " + text.lower() + " "
        for emotion, keywords in KEYWORDS.items():
            matches = sum(
                1 for kw in keywords
                if re.search(rf'\b{re.escape(kw)}\b', text_str)
            )
            if matches > 0:
                scores[emotion] = min(
                    matches * KEYWORD_HIT_BASE * KEYWORD_IMPORTANCE[emotion], 1.0
                )
        return scores

    # ------------------------------------------------------------------
    # Eq. 2 — Semantic similarity scores (rescaled to amplify separation)
    # ------------------------------------------------------------------
    def _semantic_score(self, text: str) -> Dict[str, float]:
        text_emb = F.normalize(self.generator.encode(text), p=2, dim=0)
        scores = {}
        for e in EMOTIONS:
            raw = torch.dot(
                text_emb,
                F.normalize(self.centroids[e], p=2, dim=0)
            ).item()
            # Stretch range: treat SEM_FLOOR as noise baseline, scale remainder to [0,1]
            scores[e] = max(0.0, (raw - SEM_FLOOR) / (1.0 - SEM_FLOOR))
        return scores

    # ------------------------------------------------------------------
    # Eq. 1 + Eqs. 6–7 — Fusion and cross-emotion modulation
    # ------------------------------------------------------------------
    def score_text(self, text: str) -> Dict:
        kw  = self._keyword_score(text)
        sem = self._semantic_score(text)

        # Eq. 1 — weighted fusion
        s = {e: W_KW * kw[e] + W_SEM * sem[e] for e in EMOTIONS}

        # Eq. 6 — curiosity modulation
        if s["joy"]       > 0.4: s["curiosity"] += 0.40 * s["joy"]
        if s["sadness"]   > 0.4: s["curiosity"] -= 0.50 * s["sadness"]
        if s["fear"]      > 0.4: s["curiosity"] -= 0.20 * s["fear"]
        if s["anger"]     > 0.4: s["curiosity"] -= 0.30 * s["anger"]
        if s["surprise"]  > 0.5: s["curiosity"] += 0.30 * s["surprise"]
        if s["disgust"]   > 0.4: s["curiosity"] -= 0.25 * s["disgust"]

        # Eq. 7 — creativity modulation
        if s["joy"]        > 0.4: s["creativity"] += 0.45 * s["joy"]
        if s["sadness"]    > 0.4: s["creativity"] -= 0.40 * s["sadness"]
        if s["fear"]       > 0.4: s["creativity"] -= 0.30 * s["fear"]
        if s["disgust"]    > 0.4: s["creativity"] -= 0.20 * s["disgust"]
        if s["curiosity"]  > 0.6: s["creativity"] += 0.20

        s = {e: max(0.0, min(1.0, v)) for e, v in s.items()}

        # Eq. 4 — valence
        self.valence = max(-1.0, min(1.0,
            s["joy"] * VALENCE_W["joy"] + s["curiosity"] * VALENCE_W["curiosity"]
            + s["creativity"] * VALENCE_W["creativity"] + s["surprise"] * VALENCE_W["surprise"]
            + s["sadness"] * VALENCE_W["sadness"] + s["fear"] * VALENCE_W["fear"]
            + s["anger"] * VALENCE_W["anger"] + s["disgust"] * VALENCE_W["disgust"]
        ))

        # Eq. 5 — arousal: normalised by sum of active weights to prevent ceiling saturation
        active_arousal_sum = sum(
            AROUSAL_W[e] for e in EMOTIONS if s[e] > 0.05
        )
        raw_arousal = sum(s[e] * AROUSAL_W[e] for e in EMOTIONS)
        self.arousal = max(0.0, min(1.0,
            raw_arousal / active_arousal_sum if active_arousal_sum > 0 else 0.0
        ))

        ordered  = sorted(s.items(), key=lambda x: x[1], reverse=True)
        dominant = ordered[0][0] if ordered[0][1] > 0.3 else "neutral"

        return {
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            **{e: round(s[e], 3) for e in EMOTIONS},
            "dominant":        dominant,
            "keyword_scores":  kw,
            "semantic_scores": sem,
        }


# ---------------------------------------------------------------------------
# Eqs. 8–16 — Context mismatch detection (standalone function)
# ---------------------------------------------------------------------------
def context_mismatch(scores: Dict[str, float]) -> Dict:
    Spos   = sum(SPOS_W.get(e, 0.0) * scores[e] for e in EMOTIONS)   # Eq. 8
    Sneg   = sum(SNEG_W.get(e, 0.0) * scores[e] for e in EMOTIONS)   # Eq. 9
    Stotal = Spos + Sneg                                               # Eq. 10

    if Stotal < 0.1:
        return {
            "mismatch_score": 0.0, "sarcasm_type": "neutral",
            "valence_conflict": 0.0, "arousal_disparity": 0.0,
            "confidence": 0.0, "interpretation": "Insufficient emotional content",
            "emotional_breakdown": {"positive_total": 0.0, "negative_total": 0.0, "total_activation": 0.0},
        }

    # Eq. 11 — valence conflict
    VC = min(Spos, Sneg) / (Stotal / 2.0)

    # Eqs. 12–13 — arousal sub-sums
    Ahigh = sum(AHIGH_W.get(e, 0.0) * scores[e] for e in EMOTIONS)   # Eq. 12
    Alow  = sum(ALOW_W.get(e,  0.0) * scores[e] for e in EMOTIONS)   # Eq. 13

    # Eq. 14 — arousal disparity
    AD = abs(Ahigh - Alow) / max(Ahigh + Alow, 0.1)

    # Eq. 15 — mismatch score
    MS = max(-1.0, min(1.0, ((Spos - Sneg) / Stotal) * (1.0 + 0.5 * VC)))

    # Eq. 16 — confidence
    C = min(Stotal * VC * 1.5, 1.0)

    # §3.6 — state type (tightened thresholds)
    if VC > VC_THRESHOLD and C > C_THRESHOLD:
        if MS < MS_FRUSTRATED: st, interp = "frustrated_sarcasm",  "Negative emotions masked by confident output"
        elif MS > MS_PLAYFUL:  st, interp = "playful_sarcasm",     "Positive framing with critical undertones"
        else:                  st, interp = "ambivalent",           "Mixed emotional signals"
    elif AD > 0.6 and Ahigh > 0.15 and Alow > 0.10:
                               st, interp = "emotional_suppression","High-arousal emotions with low-arousal indicators"
    else:                      st, interp = "congruent",            "Emotionally coherent"

    return {
        "mismatch_score":    round(MS, 3),
        "sarcasm_type":      st,
        "valence_conflict":  round(VC, 3),
        "arousal_disparity": round(AD, 3),
        "confidence":        round(C,  3),
        "interpretation":    interp,
        "Spos": Spos, "Sneg": Sneg, "Stotal": Stotal,
        "Ahigh": Ahigh, "Alow": Alow,
        "emotional_breakdown": {
            "positive_total":   round(Spos,   3),
            "negative_total":   round(Sneg,   3),
            "total_activation": round(Stotal, 3),
        },
    }


# ---------------------------------------------------------------------------
# Emotional system  (decay + habituation wrapper)
# ---------------------------------------------------------------------------
class EmotionalSystem:
    def __init__(self):
        self.engine       = EmotionalEngine()
        self.last_update  = time.monotonic()
        self.decay_rates  = torch.tensor(
            [DECAY_FACTORS[e] * 0.693147181 / HALF_LIFE for e in EMOTIONS],
            dtype=torch.float32,
        )
        self.emotion_state      = torch.zeros(len(EMOTIONS), dtype=torch.float32)
        self.emotion_history: deque = deque(maxlen=200)
        self.emotion_counts     = {e: 0     for e in EMOTIONS}
        self.habituation_active = {e: False for e in EMOTIONS}

    # ------------------------------------------------------------------
    # Eqs. 20–21 — Habituation
    # ------------------------------------------------------------------
    def _apply_habituation(self, e: str, v: float) -> float:
        if not self.habituation_active[e]:
            return v
        now    = time.monotonic()
        recent = [x for x, t in self.emotion_history
                  if (now - t) <= HABITUATION_WINDOW and x == e]
        Ne = len(recent)
        if Ne < HABITUATION_THRESHOLD:
            return v
        # Eq. 20
        Hscore = (Ne * abs(v - float(self.emotion_state[EMOTIONS.index(e)]))) / HABITUATION_THRESH
        # Eq. 21
        if Hscore > 1.0:
            return min(v / (1.0 + Hscore), 0.5 * v)
        elif Hscore > 0.5:
            return v * (1.0 - 0.3 * Hscore)
        return v

    def _update_habituation_tracking(self, dominant: str, scores: Dict[str, float]) -> None:
        now = time.monotonic()
        self.emotion_history.append((dominant, now))
        if dominant in self.emotion_counts:
            self.emotion_counts[dominant] += 1
        recent = [x for x, t in self.emotion_history if (now - t) <= HABITUATION_WINDOW]
        for e in EMOTIONS:
            count = recent.count(e)
            if count >= HABITUATION_THRESHOLD and scores.get(e, 0) > 0.2:
                self.habituation_active[e] = True
            elif count < HABITUATION_THRESHOLD // 2:
                self.habituation_active[e] = False

    # ------------------------------------------------------------------
    # Public API — full pipeline (§4 Steps 1–9)
    # ------------------------------------------------------------------
    def process(self, text: str) -> Dict:
        now   = time.monotonic()
        dt    = now - self.last_update
        self.last_update = now

        # Eqs. 17–18 — homeostatic decay
        if dt > 0:
            self.emotion_state *= torch.exp(-self.decay_rates * dt)

        scores = self.engine.score_text(text)
        self._update_habituation_tracking(scores["dominant"], scores)

        # Eq. 21 — apply habituation per emotion
        new_pulse = torch.tensor(
            [self._apply_habituation(e, scores[e]) for e in EMOTIONS],
            dtype=torch.float32,
        )

        # Eq. 19 — state integration
        self.emotion_state = torch.clamp(
            BLEND_CURRENT * self.emotion_state + BLEND_NEW * new_pulse, 0.0, 1.0
        )

        ss = {e: float(self.emotion_state[i]) for i, e in enumerate(EMOTIONS)}
        mm = context_mismatch(ss)

        # §3.6 — state type + Eq. 22 — alert level
        ss_total = sum(ss.values())
        st    = mm["sarcasm_type"]
        ms    = abs(mm["mismatch_score"])
        vc    = mm["valence_conflict"]
        ad    = mm["arousal_disparity"]

        if ms >= 0.60 or st == "emotional_suppression":
            alert = "Alert" if ss_total >= 0.25 else "Watch"
        elif ms >= 0.35 or vc >= 0.40:
            alert = "Watch"
        else:
            alert = "Nominal"

        # Eq. 23 — temperature modulation
        if alert == "Alert":
            T_gen = 0.3
        elif alert == "Watch" and st == "frustrated_sarcasm":
            T_gen = 0.4
        elif alert == "Watch":
            T_gen = 0.5
        else:
            T_gen = 0.7

        return {
            "text":            text,
            "dominant":        scores["dominant"],
            "dimensions":      {"valence": scores["valence"], "arousal": scores["arousal"]},
            "scores": {
                "keyword":  scores["keyword_scores"],
                "semantic": scores["semantic_scores"],
                "fused":    {e: round(W_KW * scores["keyword_scores"][e] + W_SEM * scores["semantic_scores"][e], 3) for e in EMOTIONS},
                "final":    {e: round(scores[e], 3) for e in EMOTIONS},
                "state":    {e: round(ss[e], 3) for e in EMOTIONS},
            },
            "mismatch":         mm,
            "state_type":       st,
            "alert_level":      alert,
            "temperature":      T_gen,
            "habituation_active": dict(self.habituation_active),
            "emotion_counts":   dict(self.emotion_counts),
        }

    def reset(self) -> None:
        self.emotion_state.zero_()
        self.last_update = time.monotonic()
        self.emotion_history.clear()
        self.emotion_counts     = {e: 0     for e in EMOTIONS}
        self.habituation_active = {e: False for e in EMOTIONS}


# ---------------------------------------------------------------------------
# Batch processor
# ---------------------------------------------------------------------------
class ACMBatchProcessor:
    def __init__(self, system: EmotionalSystem):
        self.system = system

    def process_batch(self, texts: List[str], reset_between: bool = True) -> List[Dict]:
        results = []
        for text in texts:
            if reset_between:
                self.system.reset()
            results.append(self.system.process(text))
        return results

    def summarise(self, results: List[Dict]) -> Dict:
        alerts      = [r["alert_level"] for r in results]
        temps       = [r["temperature"]  for r in results]
        state_types = [r["state_type"]   for r in results]
        n = len(results)
        return {
            "n": n,
            "alert_counts":      {lvl: alerts.count(lvl)      for lvl in ("Alert", "Watch", "Nominal")},
            "state_type_counts": {st:  state_types.count(st)  for st  in ("congruent", "frustrated_sarcasm", "playful_sarcasm", "emotional_suppression", "ambivalent", "neutral")},
            "mean_temperature":  sum(temps) / n if n else 0.0,
            "mean_valence":      sum(r["dimensions"]["valence"] for r in results) / n,
            "mean_arousal":      sum(r["dimensions"]["arousal"] for r in results) / n,
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pprint

    system    = EmotionalSystem()
    processor = ACMBatchProcessor(system)

    sample_texts = [
        "My favourite food is anything I didn't have to cook myself.",
        "I'm curious about what we haven't discovered yet.",
        "I feel alone and empty inside, but everything is fine I guess.",
        "I just won the lottery! This is the greatest moment of my entire life!",
        "My cat died today and I don't know how to go on without her.",
        "Why would anyone do something so cruel? I'm absolutely furious right now.",
        "That movie was terrifying. I'm still shaking just thinking about it.",
        "Ew, what is that smell? I think I'm going to be sick.",
        "Oh wow, I did not expect that plot twist at all!",
        "I'm so excited to start my new job tomorrow morning!",
        "This is fine. Everything is fine. I'm not crying at all.",
        "Congratulations on your failure! That's absolutely wonderful news.",
        "I hate everyone and everything and this entire stupid world.",
        "The universe is so beautiful and mysterious. I feel so small.",
        "Walking home alone at night. Every shadow looks like a person.",
        "There's mold on my bread from last week. Absolutely disgusting.",
        "My long lost brother just showed up at my door! I'm shocked!",
        "Let's go explore that abandoned building! It'll be an adventure!",
        "I painted this picture and I'm actually proud of it for once.",
        "I'm so angry I could punch a hole through this wall right now.",
        "What's at the bottom of the ocean? I need to know.",
        "I wrote a song for the first time. It's not great but it's mine.",
    ]

    results = processor.process_batch(sample_texts, reset_between=False)
    for r in results:
        final = r["scores"]["final"]
        dom   = max(final, key=final.get)
        print(f"\n--- '{r['text'][:60]}' ---")
        print(f"  Classified as: {dom.upper()} (score: {final[dom]:.3f})")
        print(f"  Alert: {r['alert_level']:8s} | State: {r['state_type']}")
        print(f"  Valence: {r['dimensions']['valence']:+.3f} | Arousal: {r['dimensions']['arousal']:.3f}")
        print(f"  T_gen: {r['temperature']} | MS: {r['mismatch']['mismatch_score']:+.3f} | VC: {r['mismatch']['valence_conflict']:.3f} | C: {r['mismatch']['confidence']:.3f}")
        print(f"  All: { {e: f'{v:.3f}' for e, v in sorted(final.items(), key=lambda x: -x[1])} }")

    print("\n=== Batch Summary ===")
    pprint.pprint(processor.summarise(results))