import torch
import pytorch_lightning as pl
from torch import nn
import torch.functional as F
import os
import time

class DilatedConv1d(nn.Module):
    def __init__(self, stack_size, channel_num,use_gpu = False):
        super().__init__()
        self.use_gpu = use_gpu
        self.dilate_list = nn.ModuleList()
        self.stack_size = stack_size
        for i in range(self.stack_size):
            single_dilate = nn.Conv1d(in_channels = channel_num, 
                                      out_channels = channel_num, 
                                      kernel_size = 2, 
                                      dilation = 2**i)
            self.dilate_list.append(single_dilate)
            
    def forward(self, x):
        receptive_length = 1
        for i in range(self.stack_size):
            shape = x.shape
            pad = torch.zeros(shape[0],shape[1],receptive_length)
            if self.use_gpu:
                if pad.device != 'cuda':
                    pad = pad.to('cuda')
                if x.device != 'cuda':
                    x = x.to('cuda')
            x = torch.concat((pad, x), -1)
            x = self.dilate_list[i](x)
            receptive_length *= 2
        return x
        
        

class WavenetUnconditional(pl.LightningModule):
    def __init__(self,
                 num_layers=10,
                 stack_size=10,
                 sample_rate = 16000,
                 channel_num = 1,
                 use_gpu = False
                 ):
        """Initialise an unconditional WaveNet model.

        Args:
            num_layers (int, optional): Number of residual blocks. Defaults to 10.
            stack_size (int, optional): Number of stacks in dilated conv. Defaults to 5.
        """
        super().__init__()
        self.use_gpu = use_gpu
        self.channel_num = channel_num 
        self.sample_rate = sample_rate
        self.num_layers = num_layers
        self.stack_size = stack_size
        self.start_conv = nn.Conv1d(self.channel_num,self.channel_num,1)
        self.dilated_convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.end_conv_1 = nn.Conv1d(self.channel_num,self.channel_num,1)
        self.end_conv_2 = nn.Conv1d(self.channel_num,self.channel_num,1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Sigmoid()
        for i in range(num_layers):
            self.dilated_convs.append(DilatedConv1d(stack_size, self.channel_num,use_gpu = self.use_gpu))
            self.filter_convs.append(nn.Conv1d(in_channels=self.channel_num, 
                                               out_channels=self.channel_num, 
                                               kernel_size=2))
            self.gate_convs.append(nn.Conv1d(in_channels=self.channel_num, 
                                             out_channels=self.channel_num, 
                                             kernel_size=2))
        
    # def construct_dilate_stack(self, stack_size):
    #     dilate_list = nn.ModuleList()
    #     for i in range(stack_size):
    #         single_dilate = nn.Conv1d(in_channels=1, 
    #                                   out_channels=1, 
    #                                   kernel_size=2, 
    #                                   dilation=2**i)
    #         dilate_list.append(single_dilate)
    #     dilate_stack = nn.Sequential(*dilate_list)
    #     return dilate_stack
    
    def waveNet(self, x):
        x = self.start_conv(x)
        skip_connections = []
        for i in range(self.num_layers):
            x = self.dilated_convs[i](x)
            shape = x.shape
            pad = torch.zeros(shape[0],shape[1],1)
            if self.use_gpu:
                if pad.device != 'cuda':
                    pad = pad.to('cuda')
                if x.device != 'cuda':
                    x= x.to('cuda')
            x = torch.concat((pad, x), -1)
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            skip_connections.append(x)
        x = sum(skip_connections)
        
        x = self.relu1(x)
        x = self.end_conv_1(x)
        x = self.relu2(x)
        x = self.end_conv_2(x)
        # TODO: softmax function questionable
        x = self.softmax(x)
        return x
    
    def forward(self, x):
        return self.waveNet(x)
    
    def generate(self, length, channel_num = 1, first_samples = None):
        if first_samples is None:
            first_samples = torch.randn(1,channel_num,1)
        assert first_samples.shape[1] == channel_num
        generated = first_samples
        while generated.shape[-1] < length:
            num_pad = 2**self.stack_size - generated.shape[0]
            
            if num_pad > 0:                
                input = torch.concat((torch.zeros(1,channel_num,num_pad), generated), -1)
                x = self.waveNet(input)      
            generated = torch.cat((generated, x), -1)
        return generated
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()
        loss = loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()
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
        save_path = os.path.join("./out/unconWavenet", file_name)
        
        generated = self.generate(int(duration * self.sample_rate), first_samples)
        generated = generated.squeeze().detach().numpy()
        
        torch.save(save_path, generated[0], self.sample_rate, format="wav")
        print("Generated audio: " , file_name)