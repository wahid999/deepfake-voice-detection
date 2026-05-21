import os
import torch
import torchaudio
import numpy as np
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')


class SimpleVoiceMatcher:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def _validate_audio_file(self, audio_path: str) -> bool:
        """Check if audio file exists and is valid"""
        if not os.path.exists(audio_path):
            return False
        try:
            torchaudio.info(audio_path)  # lightweight check
            return True
        except:
            return False

    def _extract_features(self, audio_path: str):
        """Extract MFCC features from audio"""
        try:
            waveform, sample_rate = torchaudio.load(audio_path)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample if needed
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)

            # MFCC extraction
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=13,
                melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23}
            )
            mfcc = mfcc_transform(waveform)

            # Mean pooling
            features = torch.mean(mfcc, dim=2).squeeze().numpy()
            return features

        except Exception as e:
            print(f"Feature extraction error ({audio_path}): {e}")
            return None

    def _compare_features(self, features1, features2, threshold=0.8) -> str:
        """Compare feature vectors with cosine similarity"""
        try:
            similarity = 1 - cosine(features1, features2)
            return "Same" if similarity > threshold else "Not Same"
        except:
            return "Not Same"

    def match(self, wav1_path: str, wav2_path: str) -> str:
        """Main matching pipeline: takes two paths, validates, extracts, compares"""
        # Validate inputs
        if not self._validate_audio_file(wav1_path) or not self._validate_audio_file(wav2_path):
            return "Not Same"

        # Extract features
        features1 = self._extract_features(wav1_path)
        features2 = self._extract_features(wav2_path)

        if features1 is None or features2 is None:
            return "Not Same"

        # Compare
        return self._compare_features(features1, features2)


# ========================
# Main function (Entry Point)
# ========================
def match_voices(wav1: str, wav2: str) -> str:
    matcher = SimpleVoiceMatcher()
    result = matcher.match(wav1, wav2)
    return result