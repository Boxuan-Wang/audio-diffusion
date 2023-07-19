import os
import torch
import torchaudio


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, use_gpu = False):
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)
        self.use_gpu = use_gpu
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.root_dir, file_name)
        audio, sr = torchaudio.load(file_path)
        norm_audio = self.normalize_sr(audio, sr)
        if self.use_gpu:
            norm_audio = norm_audio.to('cuda')
        return norm_audio, torch.cat((norm_audio[:,1:],(torch.zeros(norm_audio.shape[0],1))),1)
        
    def normalize_sr(self, audio, sr):
        # Resample to 16kHz
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(audio)
        return audio
    
    
    
    