# Messy Mashup — Robust Music Genre Classification

A deep learning project for robust music genre classification under
noisy, mixed, and distribution-shifted audio conditions.

This project was developed for the **Messy Mashup** competition, where
the test data consists of music mashups created by combining instrument
stems from different songs of the same genre, applying tempo
adjustments, and adding environmental noise.

The project explores conventional deep learning models as well as
pretrained audio transformers, including **CNN, CRNN, AST, WavLM, and
HuBERT**, with a final transformer ensemble achieving approximately
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

Therefore, the model needs to learn genre-specific characteristics that
remain robust to:

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
other.wav
```
---

## Dataset Structure

The dataset is organized by genre. Each song contains four separated
instrument stems.

```text
Dataset/
├── blues/
│   ├── song_1/
│   │   ├── drums.wav
│   │   ├── vocals.wav
│   │   ├── bass.wav
│   │   └── other.wav
│   └── ...
│
├── classical/
├── country/
├── disco/
├── hiphop/
├── jazz/
├── metal/
├── pop/
├── reggae/
└── rock/
```

---

### Data Preparation

````markdown
---

## Data Preparation

The main challenge is the distribution shift between the original
training songs and the noisy mashups used for testing.

To simulate the competition conditions during training, the pipeline
supports:

- Mixing stems from different songs of the same genre
- Audio normalization
- Tempo adjustment
- Environmental noise augmentation
- Random cropping and padding
- Mel-spectrogram generation

This makes the training samples more similar to the noisy and
recombined audio encountered during testing.

ESC-50 environmental sounds are used to introduce realistic background
noise at different signal-to-noise ratios.

---

## Audio Preprocessing

Audio is loaded using **Librosa** at a target sampling rate of 22,050 Hz.

The preprocessing pipeline includes:

1. Loading audio
2. Resampling
3. Cropping or padding to a fixed duration
4. Mixing instrument stems
5. Optional tempo adjustment
6. Optional ESC-50 noise augmentation
7. Mel-spectrogram conversion for CNN/CRNN models

For transformer-based models, audio is also prepared at 16 kHz when
required by the pretrained model.

---

## Model Development

Multiple model architectures were developed and evaluated to understand
which approaches generalize best to the noisy mashup distribution.

The project follows an incremental modeling approach:

1. CNN baseline
2. CRNN
3. AST
4. WavLM
5. HuBERT
6. Transformer ensemble

This progression allowed conventional spectrogram-based models to be
compared with pretrained audio representation models.

### CNN

A convolutional neural network was developed as a baseline model.

The CNN receives Mel-spectrogram representations and learns
time-frequency patterns associated with different music genres.

### CRNN

A Convolutional Recurrent Neural Network was explored to combine:

- CNN-based feature extraction
- Temporal sequence modeling

The convolutional layers extract local acoustic features while the
recurrent component models temporal information.

### AST

The **Audio Spectrogram Transformer (AST)** was used as a pretrained
transformer architecture for audio classification.

AST operates on spectrogram representations and provides a strong
alternative to conventional CNN-based approaches.

### WavLM

**WavLM** was evaluated as a pretrained speech/audio representation
model and fine-tuned for the 10-class music genre classification task.

### HuBERT

**HuBERT** was also evaluated as a pretrained audio representation
model and adapted for genre classification.

---

## Transformer Ensemble

The final approach combines predictions from three pretrained
transformer models:

- AST
- WavLM
- HuBERT

The predictions are combined using a weighted ensemble.

| Model | Ensemble Weight |
|---|---:|
| AST | 50% |
| WavLM | 20% |
| HuBERT | 30% |

The weighted predictions are combined to produce the final genre
prediction.

The ensemble was designed to combine complementary representations
learned by the different pretrained audio models.

---

## Results

The primary evaluation metric for the competition is **Macro F1
Score**.

Macro F1 calculates the F1 score independently for each genre and
then averages the scores, giving equal importance to every genre.

### Model Performance

| Model | Validation F1 | Test / Leaderboard Score |
|---|---:|---:|
| CNN | ~0.35 | ~0.32 |
| CRNN | ~0.60 | ~0.55 |
| Improved CRNN | ~0.70 | ~0.65 |
| HuBERT | ~0.890 | ~0.72 |
| AST | ~0.977 | ~0.815 |
| WavLM | ~0.918 | ~0.85 |
| AST + WavLM Ensemble | ~0.95 | ~0.861 |
| AST + WavLM + HuBERT Ensemble | ~0.96 | ~0.879 |

The final transformer ensemble achieved approximately **0.88 Macro F1**
on the Kaggle leaderboard.

---

## Repository Structure

```text
messy-mashup-music-genre-classification/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── messy_mashup_classification.ipynb
│   ├── milestone-1.ipynb
│   ├── milestone-2.ipynb
│   ├── milestone-3.ipynb
│   ├── milestone-4.ipynb
│   └── milestone-5.ipynb
│
└── src/
    │
    ├── data/
    │   ├── audio_processing.py
    │   └── dataset.py
    │
    ├── models/
    │   ├── cnn.py
    │   ├── crnn.py
    │   ├── transformers.py
    │   └── ensemble.py
    │
    ├── training/
    │   ├── train.py
    │   └── train_utils.py
    │
    └── inference/
        └── predict.py
```

---

## Technologies Used

### Programming and Data

- Python
- NumPy
- Pandas
- Scikit-learn

### Audio Processing

- Librosa
- Torchaudio
- Mel-spectrograms
- ESC-50 environmental noise dataset

### Deep Learning

- PyTorch
- CNN
- CRNN
- AST
- WavLM
- HuBERT
- Transformer Ensemble

### Experimentation

- Kaggle
- Weights & Biases
- Hugging Face Transformers

---

## Project Workflow

The overall pipeline follows these stages:

```text
Raw Audio
    │
    ▼
Audio Loading
    │
    ▼
Stem Selection / Mixing
    │
    ▼
Tempo Adjustment
    │
    ▼
Noise Augmentation
    │
    ▼
Audio Preprocessing
    │
    ├───────────────┐
    ▼               ▼
Mel-Spectrogram   Raw Audio
    │               │
    ▼               ▼
 CNN / CRNN     AST / WavLM / HuBERT
    │               │
    └───────┬───────┘
            ▼
       Model Ensemble
            │
            ▼
      Genre Prediction
            │
            ▼
       Submission.csv
```

---

## Running the Project

The project can be developed and executed using the modular source
code under `src/`.

### Training

The training pipeline is located at:

```text
src/training/train.py
```

### Inference

The inference pipeline is located at:

```text
src/inference/predict.py
```

---

## Hugging Face Deployment

The trained music genre classification system has also been deployed
as an interactive application on Hugging Face Spaces.

Users can upload an audio file and obtain a predicted music genre
through the deployed interface.

### Live Demo

**Hugging Face Space:**

https://huggingface.co/spaces/psanandha/music-genre-classifier

The deployed application provides an accessible interface for testing
the trained audio classification model without requiring the complete
training environment locally.

---

## Inference Workflow

The inference process follows these steps:

```text
Audio File
    │
    ▼
Audio Loading
    │
    ▼
Audio Preprocessing
    │
    ▼
Feature Extraction
    │
    ▼
Trained Model
    │
    ▼
Genre Prediction
    │
    ▼
Predicted Genre
```

---

## Supported Genres

The classifier predicts one of the following 10 music genres:

```text
Blues
Classical
Country
Disco
Hip-Hop
Jazz
Metal
Pop
Reggae
Rock
```

---

## Example Prediction

The deployed application accepts an audio file and processes it through
the trained classification pipeline.

Example:

```text
Input:
music_sample.wav

Output:
Predicted Genre: Rock
```

---

## Limitations

The current system has the following limitations:

- Performance may vary for audio outside the competition distribution.
- Very noisy audio may reduce classification accuracy.
- Unusual instrument combinations may affect predictions.
- Transformer-based models require more computational resources than
  conventional CNN models.
- The final leaderboard score depends on the specific competition test
  distribution.

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
- Faster inference optimization
- Improved deployment efficiency

---

## Key Learnings

This project provided practical experience with:

- Music genre classification
- Audio preprocessing
- Mel-spectrogram generation
- Audio augmentation
- Environmental noise robustness
- CNN and CRNN architectures
- Transfer learning
- Pretrained audio transformers
- Model ensembling
- Distribution shift
- Macro F1 evaluation
- Kaggle experimentation
- Hugging Face deployment

A major challenge was designing a model that can generalize from
individual training songs to noisy and recombined music mashups.

---

## Conclusion

The Messy Mashup project demonstrates the application of deep learning,
pretrained audio transformers, and ensemble learning to robust music
genre classification.

The project progressed from CNN and CRNN baseline models to pretrained
AST, WavLM, and HuBERT models.

The final AST + WavLM + HuBERT ensemble achieved approximately
**0.88 Macro F1 on the Kaggle leaderboard**.

The project also includes an interactive Hugging Face deployment where
users can upload an audio file and obtain a predicted music genre.

### Live Demo

https://huggingface.co/spaces/psanandha/music-genre-classifier

---

## Author

Developed as part of the **Messy Mashup music genre classification
project**.

---

## License

This project is provided for educational and research purposes.

