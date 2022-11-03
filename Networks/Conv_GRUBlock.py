import torch
from torch import nn 

class ConvGRUBlock(nn.Module):
    '''
    A single block of the convolutional GRU
    '''
    
    def __init__(self, inp_channels, hidden_channels, kernel_size, stride, padding, dialation, bias = True):
        '''
        Initilizing a block of ConvGRU
        
        Parameters:
            inp_channels: int
                Number of channels of input tensor.
            hidden_channels: int
                Number of channels of hidden layer.
            kernal_size: int 
                Size of convolutional kernal.
            stride: int 
                Size of the strides for convolution.
            padding: int
                Padding size for convolution.
            dialation: int
                Dialation for convolution.
            bias: bool
                Whether or not to add bias.
        '''
        super(ConvGRUBlock,self).__init__()

        self.hidden_dim = hidden_channels

        self.conv_gates = nn.Conv2d(in_channels= inp_channels + hidden_channels,
                                    out_channels= 2 * hidden_channels,  # for update_gate,reset_gate respectively
                                    kernel_size= kernel_size,
                                    stride = stride,
                                    padding= padding,
                                    dilation= dialation,
                                    bias= bias)

        self.conv_can = nn.Conv2d(in_channels= inp_channels + hidden_channels,
                                    out_channels= hidden_channels, # for candidate neural memory
                                    kernel_size= kernel_size,
                                    stride = stride,
                                    padding= padding,
                                    dilation= dialation,
                                    bias= bias)

    def forward(self, input, h_cur):
        '''
        Forward propagation for a block of GRU

        Parameters:
            input: (tensor --> [batchsize, input channels, width, height])
                input to the GRU block
            h_cur: (tensor --> [batchsize, hidden channels, width, height])
                current hidden states of GRU block
        
        Return:
            h_next: (tensor --> [batchsize, hidden channels, width, height])
                next hidden states of GRU block

        '''

        combined = torch.cat([input,h_cur],dim = 1)
        combined_conv = self.conv_gates(combined)

        # for update_gate,reset_gate respectively
        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)

        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([ input, reset_gate * h_cur ], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next