import os
import torch
import torchaudio
import librosa
import numpy as np


class AudioDataset(torch.utils.data.Dataset):
    # split the audio into 10s chunks, less than 10s chunks are discarded
    def __init__(self, root_dir, chunk_size = 10):
        self.root_dir = root_dir
        self.chunk_size = chunk_size
        self.files = os.listdir(root_dir)
        _lengths = []
        for file in self.files:
            file_path = os.path.join(self.root_dir, file)
            audio, sr = librosa.load(file_path)
            num_chunk = librosa.getduration(y=audio, sr=sr) // self.chunk_size
            _lengths.append(num_chunk)
        self.lengths = np.array(_lengths)
        print("Total number of 10s chunks in dataset: ", np.sum(self.lengths))
        
    def __len__(self):
        return np.sum(self.lengths)
    
    def __getitem__(self, idx):
        # find the file name given chunk id
        cum_length = np.cumsum()
        file_id = np.searchsorted(cum_length,idx)
        local_chunk_id = idx - cum_length[file_id]
        file_name = self.files[file_id]
        
        # load audio file
        file_path = os.path.join(self.root_dir, file_name)
        sr_librosa = librosa.get_samplerate(file_path)
        audio, sr = torchaudio.load(file_path, 
                                    frame_offset = local_chunk_id * self.chunk_size * sr_librosa, 
                                    num_frames = self.chunk_size * sr_librosa)
        assert sr == sr_librosa
        
        norm_audio = self.normalize_sr(audio, sr)
        return norm_audio, torch.cat((norm_audio[:,1:],(torch.zeros(norm_audio.shape[0],1))),1)
        
    def normalize_sr(self, audio, sr):
        # Resample to 16kHz
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(audio)
        return audio
    
    
    
    