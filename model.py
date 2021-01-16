import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import torchvision

class LipNet_LRW(torch.nn.Module):
    """
    Implemetnation from https://github.com/Fengdalu/LipNet-PyTorch/ with modifications
    """
    def __init__(self, vocab_size, dropout_p=0.5):
        super(LipNet_LRW, self).__init__()
        
        self.vocab_size = vocab_size 

        self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1)) 
        self.bn3 = nn.BatchNorm3d(96)
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
        self.gru1  = nn.GRU(96*3*6, 256, 1, bidirectional=True)
        self.gru2  = nn.GRU(512, 256, 1, bidirectional=True)
        
        self.FC    = nn.Linear(512, self.vocab_size)
        self.dropout_p  = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)        
        self.dropout3d = nn.Dropout3d(self.dropout_p)  
        self._init_weights() 
    
    def _init_weights(self):
        
        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        init.constant_(self.conv1.bias, 0)
        
        init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        init.constant_(self.conv2.bias, 0)
        
        init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        init.constant_(self.conv3.bias, 0)        
        
        init.kaiming_normal_(self.FC.weight, nonlinearity='sigmoid')
        init.constant_(self.FC.bias, 0)
        
        for m in (self.gru1, self.gru2):
            stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
            for i in range(0, 256 * 3, 256):
                init.uniform_(m.weight_ih_l0[i: i + 256],
                            -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0[i: i + 256])
                init.constant_(m.bias_ih_l0[i: i + 256], 0)
                init.uniform_(m.weight_ih_l0_reverse[i: i + 256],
                            -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0_reverse[i: i + 256])
                init.constant_(m.bias_ih_l0_reverse[i: i + 256], 0)
           
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout3d(x)        
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3d(x)        
        x = self.pool3(x)
        
        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # (B, C, T, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)
        
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        
        x, h = self.gru1(x)        
        x = self.dropout(x)
        x, h = self.gru2(x)   
        x = self.dropout(x)

        x = x[-1,:,:]

        x = self.FC(x)
        
        return x

class BiLSTM_LRW(nn.Module):
    """ 
    TO DO: DESCRIBE BIDIRECTION LSTM MODEL 
    """
    def __init__(self, input_size, vocab_size, hidden_dim, num_layers, dropout_p=.5): 
        super(BiLSTM_LRW, self).__init__()
       
        # Define Model Params
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define Model Layers
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout_p)
        self.dense = nn.Linear(hidden_dim*2, vocab_size)

    def forward(self, x, init_state):
        """
        Forward pass through network 
        INPUTS: 
            x: shape = (batch_size, seq_len, features)
            init_state: tuple, (h_0, c_0)
                h_0 shape = (num_layers*num_directions, batch_size, hidden_dim)
                c_0: shape = (num_layers*num_directions, batch_size, hidden_dim)

        OUTPUTS: 
            logit: shape = (batch_size, vocab_size) 
            h_n: shape = (num_layers*num_directions, batch_size, hidden_dim)
            c_n: shape = (num_layers*num_directions, batch_size, hidden_dim)
        """
        lstm_out, (h_n, c_n) = self.lstm(x, init_state)
        dense_in = torch.cat((h_n[-2,:,:],h_n[-1,:,:]),dim=1)
        logit = self.dense(dense_in)

        return logit, (h_n, c_n)

    def init_hidden(self,batch_size): 
        """
        Initialize LSTM hidden state
        """
        return (torch.zeros(self.num_layers*2, batch_size, self.hidden_dim), 
                torch.zeros(self.num_layers*2, batch_size, self.hidden_dim))


