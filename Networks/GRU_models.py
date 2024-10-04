import numpy as np
import torch, cv2
from torch import nn
import torchvision

from .Conv_GRUBlock import ConvGRUBlock


class Base_Model(nn.Module):
    '''
    The base model as mentioned in our approach with a shallow backbone network.
    '''

    def __init__(self, n_classes = 6, num_layers = 3, channels_layers = [32,64,128], device = 'cpu', is_viterbi= True ):
        '''
        Initialization

        Parameters:
            n_classes       : (int)  Number of classes for the classification of labels.
            num_layers      : (int)  Number of layers of hidden states in the architecture.
            channels_layers : (list) the size of channels belongs to each layer in the num_layers
            device          : (str) 'cuda' if use Gpu else give 'cpu'
            is_viterbi      : (bool) Is the method needs weakly-supervised viterbi output(True) or supervised sigmoid (False).
        '''

        super(Base_Model, self).__init__()

        self.num_layers = num_layers
        self.n_classes  = n_classes
        self.is_viterbi = is_viterbi
        self.device = device

        #backbone network
        self.backbone = nn.Sequential (
                                nn.Conv2d(in_channels = 1 , out_channels = 8, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels = 8 , out_channels = 16, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels = 16 , out_channels = 32, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(2,2))

        #assert number of GRU layers and channel sizes
        assert(len(channels_layers) == num_layers)
        self.channels_layers = channels_layers

        #dialation and padding required for each GRU layer (in future can be updated)
        dialation_list = [1]* self.num_layers
        padding_list = [1]* self.num_layers

        #time encoding network, GRU layers (blank list, later appended per number of layers)
        self.time_encoding_layers = nn.ModuleList()

        #adding each GRU to the encoding layers
        for n in range(self.num_layers):
            #finding the input channel size for each GRU (previous layers channel size is input channel size of each GRU)
            if n == 0:
                input_channels = self.channels_layers[0]
            else:
                input_channels =  self.channels_layers[n-1]
            #append to time_encoding layers
            self.time_encoding_layers.append(ConvGRUBlock(inp_channels = input_channels, hidden_channels =  self.channels_layers[n], kernel_size = 3, stride = 1, padding = padding_list[n], dialation= dialation_list[n], bias = True))

        #traking network
        self.tracking = nn.Sequential (
                                #nn.Upsample(scale_factor=2),
                                nn.ConvTranspose2d(128, 64, 3, stride=2, output_padding=0, padding= 1 ),
                                nn.ReLU(inplace= True),
                                nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=1,  padding= 0),
                                nn.ReLU(inplace= True),
                                nn.ConvTranspose2d(32, 1, 3, stride=2, output_padding=1 , padding= 1 ),
                                nn.Sigmoid()
                            )

        #classification network convolutional layers
        self.clf_conv = nn.Sequential (
                            nn.Conv2d(in_channels = 128 , out_channels = 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels = 256 , out_channels = 512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2,2),
                            nn.Flatten(start_dim=1)
                            )
        #classification network fully connected layers
        self.clf_fc1 = nn.Sequential ( nn.Linear(18432,4096),
                            nn.ReLU(inplace=True)
                            )
        self.clf_fc2 = nn.Sequential ( nn.Linear(4096,self.n_classes) )

        # defining a maxpool operation
        self.maxpool = nn.MaxPool2d(2,2)


    def forward(self, inputs):
        '''
        Forward Propagation of the base model architecture

        Parameters:
            inputs : (tensor --> [batchsize, seq_length, input channels, width, height])

        Return:
            outputs: (tensor --> [batchsize, seq_length, input channels, width, height])  #tracking network output
            states : (tensor --> [batchsize, seq_length, n_classes])  #classification network output
            embedding: (tensor --> [batchsize, seq_length, hidden channels, width, height]) #embedding space from the network
        '''

        # find the accelerator for training
        device = torch.device(self.device)

        #to store the outputs of n frames and return
        outputs = []
        embeddings = []
        states =  []

        #to store the GRU output of different layers
        gru_layers_curr = []

        #image width and height
        w_img = inputs.shape[3]
        h_img = inputs.shape[3]

        # iterate through each GRU layer
        for n in range(self.num_layers):
            # initialize the hidden states for each GRU layer with zeros
            hidden0 = torch.zeros(inputs.shape[0],  self.channels_layers[n], int(w_img/(2**(n+1))),int(h_img/(2**(n+1))))
            hidden0 = hidden0.to(device)
            gru_layers_curr.append(hidden0)
            # each GRU is send to GPU
            self.time_encoding_layers[n] = self.time_encoding_layers[n].to(device)

        # for each frame in sequence one layer of the architecture
        for i, input_t in enumerate(inputs.chunk(inputs.size(1), dim=1)):

            # itr: an additional first frame is send through the network to get an initial hidden layer other than zeros
            itr = 1
            if i == 0:
                itr = 3

            for l in range(itr):
                #load image
                input_tt = input_t[:,0,:,:,:]
                input_tt = input_tt.to(device)

                #feature extraction through backbone network
                input_1 = self.backbone(input_tt)

                #time encoding through GRU layers
                gru_layers_curr[0] = self.time_encoding_layers[0] (input_1, gru_layers_curr[0])
                for n in range(1,self.num_layers):
                    gru_layers_curr[n] = self.time_encoding_layers[n] (self.maxpool(gru_layers_curr[n-1]),gru_layers_curr[n])

                #tracking network output
                output = self.tracking(gru_layers_curr[self.num_layers-1])

                #classification network give states
                state1 = self.clf_conv(gru_layers_curr[self.num_layers-1])
                state2 = self.clf_fc1(state1)
                state = self.clf_fc2(state2)

                #skip if the additional first frame
                if l != itr-1 :
                    continue

                #store outputs of each frame into the list
                outputs += [ output ]
                states  += [ state ]
                embeddings += [gru_layers_curr[self.num_layers-1]]

        #stack the layers
        outputs = torch.stack(outputs, 1)
        states = torch.stack(states, 1)
        embeddings = torch.stack(embeddings, 1)

        # For viterbi return log_softmax and for supervised training return the sigmoid
        if self.is_viterbi:
            states = nn.functional.log_softmax(states, dim=1)
        else:
            states = torch.sigmoid(states)

        return outputs, states, embeddings


