import os
import torch
import torhcaudio


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.root_dir, file_name)
        audio, sr = torchaudio.load(file_path)
        return self.normalize_sr(audio, sr)
        
    def normalize_sr(self, audio, sr):
        # Resample to 16kHz
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(audio)
        return audio
    
    
    
    