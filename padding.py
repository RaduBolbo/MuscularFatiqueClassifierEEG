import torch
import torch.nn.functional as F

def pad_collate(batch):
    # Find the longest spectrogram in the batch
    max_len = max([s[0].shape[2] for s in batch])  # Assuming spectrogram is the first element in the dataset tuple

    # Pad each spectrogram to match the longest one
    padded_batch = []
    for (spectrogram, features, target) in batch:
        padded_spectrogram = F.pad(spectrogram, (0, max_len - spectrogram.shape[2]))
        padded_batch.append((padded_spectrogram, features, target))

    # Stack padded spectrograms and targets
    spectrograms = torch.stack([s[0] for s in padded_batch])
    features = torch.stack([s[1] for s in padded_batch])
    targets = torch.stack([s[2] for s in padded_batch])
    return spectrograms, features, targets

def pad_collate_inference(batch):
    # Find the longest spectrogram in the batch
    max_len = max([s[0].shape[2] for s in batch])  # Assuming spectrogram is the first element in the dataset tuple

    # Pad each spectrogram to match the longest one
    padded_batch = []
    for (spectrogram, features) in batch:
        padded_spectrogram = F.pad(spectrogram, (0, max_len - spectrogram.shape[2]))
        padded_batch.append((padded_spectrogram, features))

    # Stack padded spectrograms and targets
    spectrograms = torch.stack([s[0] for s in padded_batch])
    features = torch.stack([s[1] for s in padded_batch])
    return spectrograms, features