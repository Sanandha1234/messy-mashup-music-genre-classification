# Messy Mashup — Robust Music Genre Classification

A deep learning project for robust music genre classification under
noisy, mixed, and distribution-shifted audio conditions.

This project was developed for the **Messy Mashup** competition, where
the test data consists of music mashups created by combining instrument
stems from different songs of the same genre, applying tempo
adjustments, and adding environmental noise.

The project explores conventional deep learning models as well as
pretrained audio transformers, including CNN, CRNN, AST, WavLM, and
HuBERT, with a final transformer ensemble achieving approximately
**0.88 Macro F1 on the Kaggle leaderboard**.

---

## Problem Statement

The objective is to classify noisy music mashups into one of 10 genres:

- Blues
- Classical
- Country
- Disco
- Hip-Hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

Unlike conventional music genre classification, the competition
introduces a significant distribution shift between training and test
data.

The test samples are created by:

1. Selecting instrument stems from different songs belonging to the
   same genre
2. Synchronizing the stems when required
3. Mixing the stems
4. Adding environmental noise from ESC-50

Therefore, the model needs to learn genre-specific characteristics
that remain robust to:

- Cross-song stem recombination
- Tempo variations
- Instrument balance changes
- Background/environmental noise
- Different audio combinations

---

## Dataset

The training data contains:

- 10 music genres
- 100 songs per genre
- 4 instrument stems per song

The available stems are:

```text
drums.wav
vocals.wav
bass.wav
others.wav
```

---

## Dataset Structure

The dataset is organized by genre, with each song containing four
instrument stems.

```text
Dataset
│
├── Blues
│   ├── Song 1
│   │   ├── drums.wav
│   │   ├── vocals.wav
│   │   ├── bass.wav
│   │   └── others.wav
│   ├── Song 2
│   └── ...
│
├── Classical
├── Country
├── Disco
├── Hip-Hop
├── Jazz
├── Metal
├── Pop
├── Reggae
└── Rock
```

---

## Data Preparation

A major challenge in this competition is the difference between the
training and test distributions.

The training data contains individual songs and instrument stems,
whereas the test data consists of mashups created by combining stems
from different songs belonging to the same genre.

To make the training data more representative of the test conditions,
the project uses stem mixing and environmental noise augmentation.

Training samples can therefore contain variation in:

- Instrument combinations
- Instrument balance
- Cross-song stem combinations
- Background noise
- Audio characteristics

Environmental noise is incorporated using the **ESC-50** dataset.

---

## Audio Preprocessing

Different audio-processing approaches were explored during the
experiments.

### Mel-Spectrogram

For the CNN and CRNN experiments, audio was converted into
Mel-spectrogram representations.

These representations provide time-frequency information that can be
used by convolutional models to learn genre-specific acoustic
patterns.

### Pretrained Audio Transformers

Pretrained audio transformer models were also explored for the
classification task.

The transformer models evaluated include:

- AST
- WavLM
- HuBERT

---

## Model Development

The project followed an iterative modeling approach, starting with
conventional deep learning architectures and progressing to pretrained
audio transformers.

### CNN

A CNN-based classifier was developed as a baseline for learning
genre-specific features from Mel-spectrogram representations.

### CRNN

A CRNN architecture was explored to combine convolutional feature
extraction with temporal modeling.

### AST

The **Audio Spectrogram Transformer (AST)** was evaluated for
spectrogram-based audio classification.

### WavLM

**WavLM** was evaluated as a pretrained audio representation model for
the genre classification task.

### HuBERT

**HuBERT** was also evaluated as a pretrained audio representation
model for genre classification.

---

## Final Ensemble

The final approach combined predictions from three pretrained audio
models:

- AST
- WavLM
- HuBERT

The ensemble used the following weights:

| Model | Weight |
|---|---:|
| AST | 50% |
| WavLM | 20% |
| HuBERT | 30% |

The weighted predictions from the three models were combined to
produce the final genre classification.

---

## Results

The competition uses **Macro F1 Score** as the evaluation metric.

Macro F1 calculates the F1 score independently for each of the 10
genre classes and then averages the scores, giving each genre equal
importance.

### Model Performance

| Model | Validation F1 |
|---|---:|
| AST | ~0.977 |
| WavLM | ~0.918 |
| HuBERT | ~0.890 |
| Final Ensemble | ~0.879 |

### Kaggle Leaderboard

**Macro F1: 0.88**

The final Kaggle submission used the weighted ensemble of AST, WavLM
and HuBERT.

---

## Technologies Used

### Programming and Data

- Python
- NumPy
- Pandas
- Scikit-learn

### Audio Processing

- Librosa
- Mel-spectrograms
- ESC-50

### Deep Learning

- PyTorch
- CNN
- CRNN
- AST
- WavLM
- HuBERT
- Ensemble Learning

### Experimentation

- Kaggle
- Weights & Biases

---

## Repository Structure

```text
messy-mashup-music-genre-classification/
│
├── messy_mashup_classification.ipynb
├── README.md
└── ...
```

---

## Key Learnings

This project provided practical experience with:

- Audio classification
- Audio preprocessing
- Mel-spectrograms
- Audio augmentation
- Noise robustness
- CNN and CRNN architectures
- Transfer learning
- Pretrained audio transformers
- Model ensembling
- Distribution shift
- Macro F1 evaluation
- Kaggle experimentation

A major challenge was designing a model that could generalize from
clean training audio to noisy, recombined test mashups.

---

## Future Improvements

Potential improvements include:

- More extensive stem-level augmentation
- More systematic noise and SNR augmentation
- SpecAugment
- Cross-validation
- Test-time augmentation
- Probability calibration
- More advanced ensemble strategies
- Further fine-tuning of pretrained audio models
