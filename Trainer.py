import numpy as np
import torch, random, os, cv2
import matplotlib.pyplot as plt

from Networks.viterbi import Viterbi
from Networks.grammar import SingleTranscriptGrammar,PathGrammar
from Networks.length_model import PoissonModel

from Helper_Functions import setColorMap, plotconfusionmatrx

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    '''
    A class to handle training, validation, saving of models
    
    Parameters:
        model                   : The model to be trained
        model_type              : To train with GRU per sequence or train directly as images
        criterion1              : The loss for the tracking network
        criterion2              : The loss for the classification network
        optimizer               : Optimizer for training
        Logger                  : Logger class object for logging progress
        device_type                  : "cuda" for training in GPU else "cpu"
        n_classes               : number of classes to predict
        loss2_occurance         : Occurance of the loss 2 during training
        train_different_length  : Whether to train different lengths of sequences
        is_viterbi              : To be trained in supervised or weakly-supervised viterbi method
        decoder                 : decoder for the viterbi 
        prior                   : prior probabilities for viterbi 
        mean_lengths            : mean lengths for viterbi
    '''

    def __init__(self, model, model_type, criterion1, criterion2 , optimizer, Logger, device_type, n_classes, loss2_occurance, train_different_length, is_viterbi, decoder, prior, mean_lengths):
        '''
            initilization constructor
        '''
        self.Logger = Logger
        self.device = torch.device(device_type)

        self.net = model
        self.model_type = model_type

        self.criterion1 = criterion1
        self.criterion2 = criterion2
        self.optimizer = optimizer

        self.loss2_occurance = loss2_occurance
        self.train_different_length = train_different_length

        self.n_classes = n_classes

        self.is_viterbi = is_viterbi
        self.decoder = decoder
        self.label_counts = np.ones((self.n_classes), dtype=np.int32)
        self.prior = prior
        self.mean_lengths = mean_lengths


    def train_step(self, inputs, args, transcripts):
        """
        Train each batch with the given model network

        Arguments:
            inputs      : The dictionary from dataloader for each batch
            args        : system arguments
            transcripts : all available transcripts during training
        """

        #if train different lengths
        if self.train_different_length and self.model_type != 'img':
            counts = 2
        else:
            counts = 1
        #for each different lengths    
        for c in range(0, counts):
            
            # load the tensors from dataloader dictionary   
            target =  inputs["mask"]   
            groundtruth = inputs["label"]
            batch  = inputs["image"]

            # to train different length cut the sequences, choose random points for c=1 and c=2
            if c == 1 and 'B02' not in inputs["folder"][0]: 
                idx1 = int((torch.argmax(groundtruth[0],dim=1) == 1).nonzero(as_tuple=True)[0][0])
                strt = random.randint(0,max(0,idx1+15))
                end = random.randint(min(batch.shape[1],strt+25), batch.shape[1])
                target = target[:,strt:end]
                groundtruth = groundtruth[:,strt:end]
                batch = batch[:,strt:end]
            elif c == 2 and 'B02' not in inputs["folder"][0]:
                idx1 = int((torch.argmax(groundtruth[0],dim=1) == 2).nonzero(as_tuple=True)[0][0])
                strt = random.randint(15,max(0,idx1+25))
                #idx2 = int((torch.argmax(groundtruth[0],dim=1) == 2).nonzero(as_tuple=True)[0][0])
                end = random.randint(min(batch.shape[1],strt+25), batch.shape[1])
                target = target[:,strt:end]
                groundtruth = groundtruth[:,strt:end]
                batch = batch[:,strt:end]
            
            #load the image tensors to input1
            input1 =  batch 

            #assign the tensors to device
            groundtruth = groundtruth.to(self.device)
            target = target.to(self.device)
            if self.model_type == 'img':
                input1 = input1.to(self.device)

            #optimizer and second learning rate
            if args.is_second_lr:
                if args.steps == args.epoch_second_lr:
                    for g in self.optimizer.param_groups:
                        g['lr'] = g['lr'] * args.second_lr_ratio
            self.optimizer.zero_grad()
            
            
            #=====Forward Propagation=====
            if self.model_type == 'img':
                states, _ = self.net(input1)
            else:
                output, states, _ = self.net(input1)
                loss1 = self.criterion1(output, target)

            # loss2 with loss2_occurance
            if args.steps % self.loss2_occurance == 0:
            
                if self.is_viterbi:
                    labels_tensor = []
                    for batch_idx in range(output.shape[0]):       
                        # finding the transcript from the ground truth.
                        groundtruth_t = self.get_prediction_states(groundtruth.cpu(), idx = batch_idx).cpu().detach().numpy()
                        transcript = [groundtruth_t[0]] + [groundtruth_t[i] for i in range(1,len(groundtruth_t)) if groundtruth_t[i] != groundtruth_t[i-1]]
                        #transcript = [0,1,2,3,4,5,6]
                        #add new transcripts into transcript list
                        if transcript not in transcripts and c == 0 :
                            transcripts.append(transcript)
                        forwarder = states[batch_idx].cpu().detach().numpy()
                        log_probs = forwarder - np.log(self.prior)
                        #log_probs = log_probs - np.max(log_probs)
                        # define transcript grammar and updated length model
                        self.decoder.grammar = SingleTranscriptGrammar(transcript, self.n_classes)
                        self.decoder.length_model = PoissonModel(self.mean_lengths)

                        # decoding
                        score, labels, segments = self.decoder.decode(log_probs)

                        #labels_tensor.append(groundtruth_t.tolist())
                        labels_tensor.append(labels)

                    labels_tensor = torch.tensor(labels_tensor)
                    labels_tensor = labels_tensor.view(*labels_tensor.shape[:0], -1, *labels_tensor.shape[2:])
                    states = states.view(*states.shape[:0], -1, *states.shape[2:])
                    labels_tensor = labels_tensor.to(self.device)

                    loss2 =  args.loss2_regulerizer * self.criterion2(states, labels_tensor)

                    self.update_label_count(transcript,labels)
                    self.update_prior()
                    self.update_mean_lengths()

                else:
                    #supervised
                    loss2 =  args.loss2_regulerizer * self.criterion2(states, groundtruth)
            
            else:
                loss2 = 0* loss1
            
            #check for model with GRU or imagenet models
            if self.model_type == 'img':
                loss = loss2
                losses =  [0, [loss2.item()]]
                self.Logger.log('info', inputs["folder"][0]  + ' Step:'+ str(args.steps+1)  +' loss: '+ str(loss2.item()) )
            else:
                loss =   loss1 +  loss2 
                self.Logger.log('info', inputs["folder"][0]  + ' Step:'+ str(args.steps+1)  +' loss1: '+ str(loss1.item())  +' loss2: '+ str(loss2.item()) )
                losses =  [loss1.item(), [loss2.item()]]

            loss.backward()
            self.optimizer.step()

        return loss, losses, transcripts

    def validation_step(self, testloader,args, transcript ):
        """
        To validate the model with the test loader with model trained (current progress)
        """
        val_loss = 0
        val_loss1 = 0
        val_loss2 = 0

        # ----------Validation ---------------
        with torch.no_grad():
            for didx, test_data in enumerate(testloader):
                test_input =  test_data["image"]
                test_target = test_data["mask"]    #mask
                label =   test_data["label"]
                groundtruth_true = test_data["true"]

                #label = label.to(self.device)
                test_target = test_target.to(self.device)

                if self.model_type == 'img':
                    groundtruth_t = torch.argmax(label,dim=1).detach().numpy()
                else:
                    groundtruth_t = torch.argmax(label[0],dim=1).detach().numpy()

                #print (str(groundtruth.shape[1]), np.unique(groundtruth_t))
                transcript = [groundtruth_t[0]] + [groundtruth_t[i] for i in range(1,len(groundtruth_t)) if groundtruth_t[i] != groundtruth_t[i-1]]
      
                #========Prediction========= 
                if self.model_type == 'img':
                    test_input = test_input.to(self.device)
                    states, _ = self.net(test_input)
                else:
                    pred, states, _ = self.net(test_input)
                
                if self.is_viterbi:
                    forwarder = states[0].cpu().detach().numpy()   
                    log_probs = forwarder - np.log(self.prior)
                    #log_probs = log_probs - np.max(log_probs)

                    # define transcript grammar and updated length model
                    self.decoder.grammar =  PathGrammar( args.out_path, self.n_classes) #SingleTranscriptGrammar(transcript, self.n_classes)
                    self.decoder.length_model = PoissonModel(self.mean_lengths)

                    # decoding
                    score, labels, segments = self.decoder.decode(log_probs)
                    labels = torch.tensor(labels)
                    labels = labels.to(self.device)
                    labels[labels==6] = 0
                    
                else:
                    #batch size 1 for testdata of sequnece
                    if self.model_type == 'img':
                        labels = torch.argmax(states,dim=1)
                    else:
                        states = states[0]
                        labels = torch.argmax(states,dim=1)

                #batch size 1 for testdata of sequnece
                if self.model_type == 'img':
                    labels_tensor = torch.argmax(label,dim=1)
                else:
                    labels_tensor = self.get_prediction_states(label)
        
                self.Logger.log('info',test_data["folder"][0])
                self.Logger.log('info',"Pred1: "+str(labels))
                self.Logger.log('info',"True1: "+str(labels_tensor))
                

                #Calculate loss for tracking net
                if self.model_type == 'seq':
                    val_loss1 +=  self.criterion1(pred, test_target)

                #calculate loss for classification net
                if not self.is_viterbi:
                    labels_tensor = label
                    labels_tensor = labels_tensor.to(self.device)
                    val_loss2 +=  args.loss2_regulerizer  * self.criterion2(states, labels_tensor)
                else:
                    labels_tensor = labels_tensor.to(self.device)
                    val_loss2 +=  args.loss2_regulerizer  * self.criterion2(states[0], labels_tensor)
    
                
                #display predicted image to the output folder if the  model_type is seq
                if didx ==0 and self.model_type == 'seq':
                    #store onr of the input and output image for display purpose (to watch learning)
                    orig_image = test_input[0,25,0,:,:].cpu().detach().numpy() * 255
                    pred_img = pred[0,25,0,:,:].cpu().detach().numpy() * 255

                    #Save images to see the learning
                    cv2.imwrite(os.path.join(args.out_path,"input_"+ str(args.steps) +".png"),orig_image)
                    cv2.imwrite(os.path.join(args.out_path,"output_"+ str(args.steps) +".png"),pred_img)

                if didx == 10:
                    break

            val_loss = (val_loss1+val_loss2)/10 
            val_loss1 /= 10
            val_loss2 /= 10

            #validation loss1 is not available for imgnets
            if self.model_type == 'img':
                val_losses = [0, val_loss2.item()]
            else:
                val_losses = [val_loss1.item(), val_loss2.item()]

            self.Logger.log('info',test_data["folder"][0] + ' Step ' + str(args.steps) +  ' test loss:' + str(val_loss.item()) )

        return val_loss, val_losses

    def update_label_count(self, transcript, labels):
        '''
        Update the label count after each iteration for viterbi
        '''
        for i in transcript:
            self.label_counts[i] = labels.count(i)
        self.instance_counts = []
        self.instance_counts.append( np.array( [ sum(np.array(transcript) == c) for c in range(self.n_classes) ] ) )

    def update_mean_lengths(self):
        '''
        Update the mean length after each iteration for viterbi
        '''
        self.mean_lengths = np.zeros( (self.n_classes), dtype=np.float32 )
        self.mean_lengths += self.label_counts
        instances = np.zeros((self.n_classes), dtype=np.float32)
        for instance_count in self.instance_counts:
            instances += instance_count
        # compute mean lengths (backup to average length for unseen classes)
        self.mean_lengths = np.array( [ self.mean_lengths[i] / instances[i] if instances[i] > 0 else sum(self.mean_lengths) / sum(instances) for i in range(self.n_classes) ])

    def update_prior(self):
        '''
        Update the prior probabilities after each iteration for viterbi
        '''
        self.prior = np.zeros((self.n_classes), dtype=np.float32)
        self.prior += self.label_counts
        self.prior = (self.prior + 0.00001) / np.sum(self.prior)

    def get_prediction_states(self, states, idx = 0):
        '''
        Function which find the prediction labels from the posterior probability
        '''
        pred = torch.argmax(states[idx],dim=1)
        return pred

    def save_model(self,args):
        '''
        Save the model and related probabilities
        '''
        torch.save(self.net.state_dict(), os.path.join(args.out_path,"trained_model_"+str(args.steps)+".pth") )
        self.Logger.log('info', "Model Saved in ../trained_model_"+str(args.steps)+".pth ")
        np.savetxt( os.path.join(args.out_path,"legths.iter_"+str(args.steps)+".txt"), self.mean_lengths)
        np.savetxt( os.path.join(args.out_path,"prior.iter_"+str(args.steps)+".txt"), self.prior)

    
    def test_step(self, testloader, args):
        '''
        A function which calculate the test scores and calculate corresponding matrices.
        '''
        #arguments for storing the results to calculate the scores
        labels_store = {}
        ua = []
        viterbi = [] 
        labels_Vector = []
        groundtruth_Vector = []

        features_tensorboard = []
        pred_tensorboard = []
        true_tensorboard = []

        #without gradients itereate through dataloader
        with torch.no_grad():
            for didx,test_data in enumerate(testloader):

                test_input =  test_data["image"]
                label = test_data["label"]

                 #========Prediction========= 
                if self.model_type == 'img':
                    test_input = test_input.to(self.device)
                    states, embeddings = self.net(test_input)
                else:
                    pred, states, embeddings = self.net(test_input)

                if self.is_viterbi:
                    forwarder = states[0].cpu().detach().numpy() 
                    log_probs = forwarder - np.log(self.prior)

                    # define transcript grammar and updated length model
                    self.decoder.grammar =  PathGrammar( args.out_path, self.n_classes) #SingleTranscriptGrammar(transcript, self.n_classes)
                    self.decoder.length_model = PoissonModel(self.mean_lengths)

                    # decoding
                    score, labels, segments = self.decoder.decode(log_probs)
                    labels = torch.tensor(labels)
                    labels[labels==6] = 0
                    #labels = labels.to(self.device)
                else:
                    if self.model_type == 'img':
                        labels = torch.argmax(states,dim=1).cpu()
                    else:
                        labels = torch.argmax(states[0],dim=1).cpu()

                #batch size 1 for testdata of sequnece
                if self.model_type == 'img':
                    labels_tensor = torch.argmax(label,dim=1)
                else:
                    labels_tensor = self.get_prediction_states(label)

                #store the predicted labels into arguments for evaluation
                labels_s = labels + 1
                labels_store[didx+1] = labels_s.detach().numpy().tolist()
                final_labels = labels_s.detach().numpy()
                ua.append(np.array(labels_tensor+1))
                viterbi.append(final_labels)
                final_labels = torch.tensor(final_labels)
                labels_store[didx+1] = final_labels.detach().numpy().tolist()
                groundtruth_3 = np.array(labels_tensor+1)
                for l in range(len(final_labels)):
                    labels_Vector.append(final_labels[l])
                    groundtruth_Vector.append(groundtruth_3[l])

                #feature space plot
                if self.model_type == 'img':
                    new_embeddings = embeddings
                else:
                    new_embeddings = embeddings[0].flatten(start_dim = 1)
                if didx == 0:
                    features_tensorboard = new_embeddings.cpu()
                    pred_tensorboard = torch.tensor(final_labels)
                    true_tensorboard = torch.tensor(np.array(labels_tensor+1))
                else:
                    features_tensorboard = np.concatenate((features_tensorboard,new_embeddings.cpu()))
                    pred_tensorboard = np.concatenate((pred_tensorboard, torch.tensor(final_labels) ))
                    true_tensorboard = np.concatenate((true_tensorboard, torch.tensor(np.array(labels_tensor+1)) ))
        
        k = self.n_classes

        #Plotting the time plot from the test dataset
        ua  = np.array(ua)
        viterbi = np.array(viterbi)
        methods = [ua,viterbi]
        titles = ['user annotation', args.architecture ]
        cmap = setColorMap(k)
        fig, axs = plt.subplots(1, 2, figsize=(15, 10), facecolor='w', edgecolor='k')
        axs = axs.ravel()
        for i in range(2):
            axs[i].matshow(methods[i],cmap=cmap)
            axs[i].set_title(titles[i],pad=8, fontweight='bold')
            axs[i].set_xlabel("Time (Frame Number)")
            axs[i].set_ylabel("Cell Trajecteries")
            axs[i].grid(True)
        plt.savefig(os.path.join(args.out_path,"showLabelMatrix_"+ str(args.steps) +".jpg"), dpi=150)


        #claculating performance and plotting confusion matrix
        #Convert result and ground truth to arrays
        labels_Vector = np.array(labels_Vector)
        groundtruth_Vector = np.array(groundtruth_Vector)

        #calculate number of correct frames
        numcorrect = sum(groundtruth_Vector==labels_Vector)

        Correct = numcorrect
        Accuracy = numcorrect/ groundtruth_Vector.shape[0]

        #Calculation of confusion matrix
        confus = np.zeros((6,6))
        for i in range(6):
            class1 = i+1
            first = np.where(groundtruth_Vector==class1)
            #print (i+1,first)     
            for j in range(6):
                length = 0
                for d in first[0]:
                    if labels_Vector[d]== j+1:
                        length += 1
                confus[i,j] = length

        plotconfusionmatrx(confus, args)

        #Calculation of Precision Recall and F1 Score from confusion matrix.
        precision=[]
        recall=[]
        F1=[]
        for i in range(6):
            r = confus[i,i]/np.sum(confus[i,:])
            recall.append(r)
            p = confus[i,i]/np.sum(confus[:,i])
            precision.append(p)
            f1 =  2 * (p*r) / (p+r)
            F1.append(f1)

        #storing the performance values to result file
        with  open(os.path.join(args.out_path ,"Results.txt"), 'a+') as file_object:
            # Append 'hello' at the end of file
            file_object.write("Iteration: "+ str(args.steps) + '\n')
            file_object.write("-------------------------------" + '\n')
            file_object.write("Num Correct: "+ str(Correct) + '\n')
            file_object.write("Accuracy: "+ str(Accuracy) + '\n')
            file_object.write("Recall: "+ str(recall) + '\n')
            file_object.write("Precision: "+ str(precision) + '\n')
            file_object.write("F1: "+ str(F1) + '\n\n\n\n\n')


        #plot the tensorboard for 5000th iteration
        if args.steps == 9000:

            tensorboard_folder = os.path.join(args.out_path, 'run_' + str(args.steps) + 'tensorboard')
            if not os.path.exists(tensorboard_folder):
                os.makedirs(tensorboard_folder)
            writer = SummaryWriter(os.path.join(tensorboard_folder, args.architecture  + '_true' ))
            writer.add_embedding(features_tensorboard,
                    metadata= true_tensorboard)
            writer.close()
            writer = SummaryWriter(os.path.join(tensorboard_folder, args.architecture + '_pred' ))
            writer.add_embedding(features_tensorboard,
                    metadata=pred_tensorboard)
            writer.close()