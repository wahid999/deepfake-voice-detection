import torch
import torchaudio
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import numpy as np

# ------------------------
# Model Definition
# ------------------------
import torch.nn as nn

class ANNModel(nn.Module):
    def __init__(self, input_dim):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)

        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return torch.sigmoid(self.out(x)).squeeze()


# ------------------------
# Preprocessing Function
# ------------------------
def preprocess_audio(audio_path, sample_rate=16000, n_mfcc=40, target_duration=3):
    waveform, sr = torchaudio.load(audio_path)

    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
        sr = sample_rate

    # Pad or trim to target length
    target_len = sample_rate * target_duration
    if waveform.shape[1] > target_len:
        waveform = waveform[:, :target_len]
    else:
        waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[1]))

    # Extract MFCC features
    mfcc = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=n_mfcc)(waveform)
    mfcc = mfcc.mean(dim=-1).squeeze()  # shape: (n_mfcc,)

    return mfcc


# ------------------------
# Prediction Function
# ------------------------
def predict(audio_path, scaler=None):
    # Preprocess
    features = preprocess_audio(audio_path)

    # Scale (important: use the scaler fitted on train set)
    if scaler:
        features = scaler.transform([features.numpy()])[0]
        features = torch.tensor(features, dtype=torch.float32)
    else:
        features = features.float()

    # Load model
    input_dim = features.shape[0]
    model = ANNModel(input_dim)
    model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        output = model(features.unsqueeze(0))  # add batch dimension
        pred = int(output >= 0.5)

    return "REAL" if pred == 0 else "FAKE"


# ------------------------
# Main Function
# ------------------------
def predict_audio(audio_file):
    result = predict(audio_file)
    return result