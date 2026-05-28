# ACM: Affective Coherence Monitoring

A sophisticated Python framework for detecting emotional coherence and conversational affect in text. **ACM** implements a full real-time pipeline that combines **semantic embeddings**, **keyword lexicons**, **dimensional emotion modeling** (valence & arousal), and **context mismatch detection** to understand when conversational tone is congruent, sarcastic, suppressed, or otherwise emotionally inconsistent.

## Overview

ACM continuously monitors the emotional state of a conversation, detecting:

- **Dominant emotions** across 8 distinct categories (joy, sadness, curiosity, creativity, fear, anger, surprise, disgust)
- **Emotional dimensions** via valence (positive ↔ negative) and arousal (calm ↔ activated)
- **Affective mismatches** (e.g., sarcasm, emotional suppression, frustration)
- **Alert levels** (Nominal, Watch, Alert) based on emotional coherence and context violation
- **Habituation effects** to avoid over-sensitivity from repeated emotional triggers
- **Homeostatic decay** so emotional states naturally fade over time

All computations are grounded in **research equations** (Eqs. 1–23) rigorously documented throughout the codebase.

## Features

### 1. **Semantic Emotion Scoring** (Eqs. 1–3)
- Uses SentenceTransformer embeddings to compute semantic similarity between input text and learned emotion centroids
- Centroids are built from exemplar sentences using weighted mean of embeddings
- Softmax temperature sharpening ensures central exemplars contribute most

### 2. **Keyword-Based Scoring** (Eq. 3.1.1)
- Fast lexicon lookup for emotion-associated keywords
- Per-emotion importance multipliers tune sensitivity
- Combined with semantic scores via weighted fusion (Eq. 1)

### 3. **Cross-Emotion Modulation** (Eqs. 6–7)
- Models how emotions influence each other (e.g., joy amplifies curiosity and creativity)
- Prevents unrealistic emotion combinations
- Results in refined, contextually-aware emotion profiles

### 4. **Dimensional Modeling** (Eqs. 4–5)
- **Valence**: maps emotions to positive–negative spectrum
- **Arousal**: maps emotions to calm–activated spectrum
- Captures affective tone beyond categorical labels

### 5. **Context Mismatch Detection** (Eqs. 8–16)
- **Activation sums** (Spos, Sneg) combine compatible emotions
- **Valence conflict** (VC): tension between positive and negative activations
- **Arousal disparity** (AD): mismatch between high-arousal and low-arousal emotions
- **Mismatch score** (MS): unified coherence metric
- **Confidence** (C): reliability indicator

### 6. **Affective State Classification** (Eq. 3.6)
Categorizes conversational tone:
- **Congruent**: emotional state matches semantic content (typical)
- **Playful sarcasm**: positive emotions with negative mismatch signal
- **Frustrated sarcasm**: negative emotions with positive mismatch signal
- **Emotional suppression**: high arousal disparity (contradictory feelings)

### 7. **Alert Generation** (Eq. 22)
- **Nominal**: stable, coherent emotional state
- **Watch**: potential sarcasm or moderate conflict detected
- **Alert**: strong mismatch, suppression, or incoherence

### 8. **Temperature Modulation** (Eq. 23)
- Adjusts sampling temperature for language model generation based on alert level
- Lower temperature (0.3) for Alert states (more deterministic, safer responses)
- Higher temperature (0.7) for Nominal states (more creative)

### 9. **Homeostatic Decay** (Eqs. 17–19)
- Emotional state exponentially decays over time (half-life: 4 seconds)
- Per-emotion decay factors model realistic emotional fading patterns
- Blended integration (40% decayed + 60% new input) prevents abrupt state flips

### 10. **Habituation Dynamics** (Eqs. 20–21)
- Tracks repeated emotion activations within 30-second window
- Attenuates scores when same emotion fires repeatedly (adaptation)
- Prevents over-response to recurring emotional keywords

## Installation

```bash
# Clone the repository
git clone https://github.com/TheOfficialFurkanNar/ACM.git
cd ACM

# Install dependencies
pip install requirements.txt