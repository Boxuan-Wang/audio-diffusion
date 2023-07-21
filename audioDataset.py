import os
import torch
import torchaudio
import librosa
import numpy as np
import torch.nn.functional as F


class AudioDataset(torch.utils.data.Dataset):
    # split the audio into 10s chunks, less than 10s chunks are padded
    def __init__(self, root_dir, chunk_size = 10, norm_sr = 16000):
        self.root_dir = root_dir
        self.chunk_size = chunk_size
        self.norm_sr = norm_sr
        self.files = os.listdir(root_dir)
        _lengths = []
        for file in self.files:
            file_path = os.path.join(self.root_dir, file)
            duration = librosa.get_duration(path=file_path)
            num_chunk = duration // self.chunk_size + (0 if duration%self.chunk_size==0 else 1)
            _lengths.append(num_chunk)
        self.lengths = np.array(_lengths)
        print("Total number of 10s chunks in dataset: ", np.sum(self.lengths))
        
    def __len__(self):
        return int(np.sum(self.lengths))
    
    def __getitem__(self, idx):
        # find the file name given chunk id
        cum_length = np.cumsum(self.lengths)
        file_id = np.searchsorted(cum_length,idx, side = 'right')
        local_chunk_id = int(idx - (cum_length[file_id-1] if file_id >0 else 0))
        file_name = self.files[file_id]
        
        # load audio file
        file_path = os.path.join(self.root_dir, file_name)
        sr_librosa = int(librosa.get_samplerate(file_path))
        audio, sr = torchaudio.load(file_path, 
                                    frame_offset = local_chunk_id * self.chunk_size * sr_librosa, 
                                    num_frames = self.chunk_size * sr_librosa)
        assert sr == sr_librosa
        
        norm_audio = self.normalize_sr(audio, sr)
        if norm_audio.shape[1] < self.norm_sr * self.chunk_size:
            # padding required
            norm_audio = F.pad(norm_audio,(0,self.norm_sr * self.chunk_size - norm_audio.shape[1]))
        return norm_audio, torch.cat((norm_audio[:,1:],(torch.zeros(norm_audio.shape[0],1))),1)
        
    def normalize_sr(self, audio, sr):
        # Resample to 16kHz
        if sr != self.norm_sr:
            resampler = torchaudio.transforms.Resample(sr, self.norm_sr)
            audio = resampler(audio)
        return audio
    
    
    
    