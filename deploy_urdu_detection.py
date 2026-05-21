import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import torchaudio
import torch.nn.functional as F

# =====================
# 1. Model Definition
# =====================
class HybridModel(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridModel, self).__init__()
        self.resnet = models.resnet50(weights=None)
        self.resnet.fc = nn.Identity()

        self.vit = models.vit_b_16(weights=None)
        self.vit.heads = nn.Identity()

        self.classifier = nn.Linear(2048 + 768, num_classes)

    def forward(self, x):
        resnet_feat = self.resnet(x)
        vit_feat = self.vit(x)
        combined = torch.cat((resnet_feat, vit_feat), dim=1)
        return self.classifier(combined)


# =====================
# 2. Preprocessing
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

def preprocess_audio(audio_path):
    waveform, sr = torchaudio.load(audio_path)

    # Convert to Mel-Spectrogram
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_mels=128
    )(waveform)
    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)

    # Expand to 3 channels for ResNet/Vit
    mel_spec_db = mel_spec_db.expand(3, -1, -1)

    # Apply transforms (resize + normalize)
    mel_spec_db = transform(mel_spec_db)

    return mel_spec_db.unsqueeze(0)  # add batch dimension


# =====================
# 3. Prediction Function
# =====================
def predict(audio_path, model_path="urdu_detection_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = HybridModel(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Preprocess
    mel_spec = preprocess_audio(audio_path).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(mel_spec)
        probs = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()

    return "REAL" if predicted_class == 0 else "FAKE"


# =====================
# 4. Main Function
# =====================
def predict_urdu_audio(audio_file):
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"❌ File not found: {audio_file}")

    result = predict(audio_file)
    return result