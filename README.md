# AI-based Deepfake Voice Detection and Authentication

## Overview
This project focuses on detecting synthetic (deepfake) voice using Artificial Intelligence and Machine Learning techniques. The system analyzes audio signals to distinguish between real and AI-generated speech while also supporting speaker authentication tasks.

The project was developed as part of MSc Computer Science research work focused on improving security against voice spoofing attacks.

---

## Features
- Deepfake audio classification
- MFCC feature extraction
- Spectrogram analysis
- Speaker authentication
- Urdu speech detection
- Real vs Fake voice prediction

---

## Technologies Used
- Python
- PyTorch
- Librosa
- Scikit-learn
- Jupyter Notebook
- Flask
- HTML/CSS

---

## Methodology
1. Load and preprocess audio datasets
2. Extract MFCC and spectrogram features
3. Train machine learning models
4. Evaluate performance using confusion matrix and accuracy metrics
5. Deploy prediction system using Flask

---

## Project Structure

```bash
deepfake-voice-detection/
│
├── deepfake_audio_detection.ipynb
├── speaker_authentication.ipynb
├── urdu_speech_detection.ipynb
├── deploy_deepfake.py
├── deploy_voice_matcher.py
├── deploy_urdu_detection.py
├── app.py
├── requirements.txt
└── README.md

Author

Wahid Hussain
MSc Computer Science
Anglia Ruskin University, United Kingdom
