import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
import os
import time
import torchaudio

class DilatedConv1d(nn.Module):
    def __init__(self, 
                 stack_size, 
                 channel_num,
                 use_gpu = False):
        super().__init__()
        self.use_gpu = use_gpu
        self.dilate_list = nn.ModuleList()
        self.stack_size = stack_size
        for i in range(self.stack_size):
            single_dilate = nn.Conv1d(in_channels = channel_num,
                                      out_channels = channel_num,
                                      kernel_size = 2,
                                      padding = 2**i,
                                      dilation = 2**i)
            self.dilate_list.append(single_dilate)

    def forward(self, x):
        for i in range(self.stack_size):
            shape = x.shape
            x = self.dilate_list[i](x)
            x = x[:shape[0],:shape[1],:shape[2]]
        return x



class WavenetUnconditional(pl.LightningModule):
    def __init__(self,
                 num_layers=10,
                 stack_size=10,
                 target_field = 1024,
                 sample_rate = 16000,
                 audio_channel_num = 1,
                 use_gpu = False,
                 bits = 8
                 ):
        """Initialise an unconditional WaveNet model.

        Args:
            num_layers (int, optional): Number of residual blocks. Defaults to 10.
            stack_size (int, optional): Number of stacks in dilated conv. Defaults to 5.
            
        """
        super().__init__()
        self.use_gpu = use_gpu
        self.audio_channel_num = audio_channel_num
        self.bits = bits
        self.nn_channel_num = self.audio_channel_num * (2**self.bits)
        self.sample_rate = sample_rate
        self.num_layers = num_layers
        self.stack_size = stack_size
        self.target_field = target_field
        self.start_conv = nn.Conv1d(self.nn_channel_num,self.nn_channel_num,1)
        self.dilated_convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.linear = nn.Linear(self.num_layers * (2**self.stack_size), target_field, bias = False)
        self.end_conv_1 = nn.Conv1d(self.nn_channel_num,self.nn_channel_num,1)
        self.end_conv_2 = nn.Conv1d(self.nn_channel_num,self.nn_channel_num,1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)
        for i in range(num_layers):
            self.dilated_convs.append(DilatedConv1d(stack_size, self.nn_channel_num,use_gpu = self.use_gpu))
            self.filter_convs.append(nn.Conv1d(in_channels=self.nn_channel_num,
                                               out_channels=self.nn_channel_num,
                                               kernel_size=2,
                                               padding=1))
            self.gate_convs.append(nn.Conv1d(in_channels=self.nn_channel_num,
                                             out_channels=self.nn_channel_num,
                                             kernel_size=2,
                                             padding=1))
            self.residual_convs.append(nn.Conv1d(in_channels=self.nn_channel_num,
                                                 out_channels=self.nn_channel_num,
                                                 kernel_size=1))
    def waveNet(self, x):
        x = self.start_conv(x)
        skip_connections = []
        for i in range(self.num_layers):
            residual = x
            x = self.dilated_convs[i](x)
            shape = x.shape
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            x = x[:shape[0],:shape[1],:shape[2]]
            x = self.residual_convs[i](x)
            skip_connections.append(x)
            x += residual
        x = sum(skip_connections)

        x = self.linear(x)        
        x = self.relu1(x)
        x = self.end_conv_1(x)
        x = self.relu2(x)
        x = self.end_conv_2(x)
        x = self.softmax(x)
        return x

    def forward(self, x):
        return self.waveNet(x)

    def generate(self, length, first_samples = None):
        self.eval()
        if first_samples is None:
            first_samples = torch.randn((1,self.audio_channel_num,1))
        assert first_samples.shape[1] == self.audio_channel_num
        generated = first_samples
        if self.use_gpu:
          generated = generated.to('cuda')
        while generated.shape[-1] < length:
            num_pad = 2**(self.stack_size-1) - generated.shape[-1]

            if num_pad > 0:
                # input = torch.concat((torch.zeros(1,channel_num,num_pad), generated), -1)
                input = F.pad(generated, (num_pad,0))
            x = self.waveNet(input[:,:,-1*2**(self.stack_size-1):])
            generated = torch.cat((generated, x[:,:,-1:]), -1)
        self.train()
        return generated

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()
        loss = loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()
        loss = loss(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def generate_audio(self, duration = 1.0, first_samples = None, file_name = None):
        if file_name is None:
            file_name = time.strftime("%Y%m%d%H%M%S") + ".wav"
        elif not "." in file_name:
            file_name = file_name + ".wav"
        folder_path = "./out/unconWavenet"
        save_path = os.path.join(folder_path, file_name)
        if not os.path.exists(folder_path):
            # make dir
            os.makedirs(folder_path)
        generated = self.generate(int(duration * self.sample_rate), first_samples=first_samples)
        if self.use_gpu:
          generated = generated.detach().cpu()

        torchaudio.save(save_path, generated[0], self.sample_rate, format="wav")
        print("Generated audio: " , file_name)
