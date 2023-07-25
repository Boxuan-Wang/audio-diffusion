import os
import torch
import torchaudio
import librosa
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm


class AudioDataset(torch.utils.data.Dataset):
    # split the audio into 10s chunks, less than 10s chunks are padded
    def __init__(self, root_dir, chunk_size = 10, norm_sr = 16000):
        self.root_dir = root_dir
        self.chunk_size = chunk_size
        self.norm_sr = norm_sr
        self.files = os.listdir(root_dir)
        self.tensors = []
        for file in tqdm(self.files):
            file_path = os.path.join(self.root_dir, file)
            wave, sr = torchaudio.load(file_path)
            norm_audio = self.normalize_sr(wave, sr)
            duration = wave.shape[1] / sr
            num_chunk = int(duration // self.chunk_size + (0 if duration%self.chunk_size==0 else 1))
            self.tensors+=torch.chunk(norm_audio, num_chunk, dim = 1)
        print('Number of chunks: ', len(self.tensors))



    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        norm_audio = self.tensors[idx]
        if norm_audio.shape[1] < self.norm_sr * self.chunk_size:
            # padding required
            norm_audio = F.pad(norm_audio,(0,self.norm_sr * self.chunk_size - norm_audio.shape[1]))
        return torch.cat(((torch.zeros(norm_audio.shape[0],1)),norm_audio[:,:-1]),1), norm_audio

    def normalize_sr(self, audio, sr):
        # Resample to 16kHz
        if sr != self.norm_sr:
            resampler = torchaudio.transforms.Resample(sr, self.norm_sr)
            audio = resampler(audio)
        return audio
