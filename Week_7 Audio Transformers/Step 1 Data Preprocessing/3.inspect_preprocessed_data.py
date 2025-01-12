import numpy as np
import torch

# Load a log-Mel spectrogram
log_mel_spec = np.load("./multi_processed_dataset/mel_0.npy")
print("Log-Mel Spectrogram Shape:", log_mel_spec.shape)

# Load tokenized transcription
tokens = torch.load("./multi_processed_dataset/tokens_0.pt", weights_only=True)
print("Tokens:", tokens)
