import torch
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader
import torch.functional as F

class DilatedConv1d(nn.Module):
    def __init__(self, stack_size):
        self.dilate_list = nn.ModuleList()
        self.stack_size = stack_size
        for i in range(self.stack_size):
            single_dilate = nn.Conv1d(in_channels=1, 
                                      out_channels=1, 
                                      kernel_size=2, 
                                      dilation=2**i)
            self.dilate_list.append(single_dilate)
            
    def forward(self, x):
        receptive_length = 1
        for i in range(self.stack_size):
            receptive_length *= 2
            x = F.pad(x, (receptive_length - 1, 0))
            x = self.dilate_list[i](x)
        return x
        
        

class WavenetUnconditional(pl.LightningModule):
    def __init__(self,
                 num_layers=10,
                 stack_size=5,
                 sample_rate = 16000
                 ):
        """Initialise an unconditional WaveNet model.

        Args:
            num_layers (int, optional): Number of residual blocks. Defaults to 10.
            stack_size (int, optional): Number of stacks in dilated conv. Defaults to 5.
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.num_layers = num_layers
        self.stack_size = stack_size
        self.start_conv = nn.Conv1d(1,1,1)
        self.dilated_convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.end_conv_1 = nn.Conv1d(1,1,1)
        self.end_conv_2 = nn.Conv1d(1,1,1)
        for i in num_layers:
            self.dilated_convs.append(DilatedConv1d(stack_size))
            self.filter_convs.append(nn.Conv1d(in_channels=1, 
                                               out_channels=1, 
                                               kernel_size=2))
            self.gate_convs.append(nn.Conv1d(in_channels=1, 
                                             out_channels=1, 
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
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            skip_connections.append(x)
        x = sum(skip_connections)
        x = F.relu(x)
        x = self.end_conv_1(x)
        x = F.relu(x)
        x = self.end_conv_2(x)
        x = F.softmax(x)
        return x
    
    def forward(self, x):
        return self.waveNet(x)
    
    def generate(self, length, first_samples = None):
        if first_samples is None:
            first_samples = torch.zeros(1,1,1)
        generated = first_samples
        while generated.shape[0] < length:
            num_pad = 2**self.stack_size - generated.shape[0]
            
            if num_pad > 0:
                input = F.pad(generated, (num_pad, 0))
                x = self.waveNet(input)      
            generated = torch.cat((generated, x))
        return generated
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    