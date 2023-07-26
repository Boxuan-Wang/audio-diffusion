import os
import torch
import torchaudio
import librosa
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm
import math


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 root_dir, 
                 receptive_field = 1024,
                 target_field = 1,
                 layer_num = 10,
                 norm_sr = 16000, 
                 bits =8,
                 test = False):
        """Initialise the audio dataset

        Args:
            root_dir (str): root directory of the audio files
            receptive_field (int, optional): receptive field length = 2 ** layers of dilated convs. Defaults to 1024.
            target_field (int, optional): length of target field. Defaults to 1.
            chunk_size (int, optional): desperate. Defaults to 10.
            norm_sr (int, optional): normal sample rate. Defaults to 16000.
            bits (int, optional): number of bits for each sample. Defaults to 8.
        """
        self.root_dir = root_dir
        self.receptive_field = receptive_field
        self.target_field = target_field
        self.layer_num = layer_num
        self.norm_sr = norm_sr
        self.bits = bits
        self.files = os.listdir(root_dir)
        self.tensors = []
        self.num_samples = []
        if test:
            self.files = self.files[:10]
        for file in tqdm(self.files):
            file_path = os.path.join(self.root_dir, file)
            wave, sr = torchaudio.load(file_path)
            wave = self.multi_audio_handling(wave)
            norm_audio = self.normalize_sr(wave, sr)
            norm_audio = self.softmax(norm_audio)
            norm_audio = self.resemble_to_bits(norm_audio)
            self.tensors.append(norm_audio)
            self.num_samples.append(norm_audio.shape[1] - receptive_field*self.layer_num - target_field + 1)
        print('Number of chunks: ', sum(self.num_samples))
        
    def __len__(self):
        return sum(self.num_samples)

    def __getitem__(self, idx):
        cum_lengs = np.cumsum(self.num_samples)
        file_id = np.searchsorted(cum_lengs, idx, side='right')
        local_id = idx - (cum_lengs[file_id-1] if file_id > 0 else 0)
        norm_audio = self.tensors[file_id][:,local_id:local_id+self.receptive_field*self.layer_num+self.target_field]
        return self.one_hot_encoding(norm_audio[:,:-self.target_field]), self.one_hot_encoding(norm_audio[:,-self.target_field:])

    def normalize_sr(self, audio, sr):
        # Resample to 16kHz
        if sr != self.norm_sr:
            resampler = torchaudio.transforms.Resample(sr, self.norm_sr)
            audio = resampler(audio)
        return audio
    
    def resemble_to_bits(self, wave):
        # quantize the wave to 8 bits
        scaled_wave = ((wave + 1.0) * (2**self.bits - 1) / 2)
        scaled_wave = scaled_wave.to(torch.int64).clamp(0,255)
        return scaled_wave
    
    def softmax(self, wave):
        # the softmax distribution described in WaveNet
        n = 2**self.bits
        ret = torch.sign(wave) * torch.log(1 + (n-1) * torch.abs(wave)) / math.log(n)
        return ret
    
    def one_hot_encoding(self,wave):
        num_channel = (2**self.bits)*(wave.shape[0])
        target = torch.zeros(num_channel, wave.shape[1],dtype=torch.float32)
        target =  target.scatter_(0, wave, 1)
        return target
    
    def multi_audio_handling(self,wave):
        # handling multi-channel audio to single channel, discarding the rest
            if wave.shape[0] > 1:
                wave = wave[:1]
            return wave