import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
import os, time
import numpy as np
import argparse

from Helper_Functions import load_image_paths_zhong, load_image_paths_livecellminer, plot_loss, plot_losses, plotconfusionmatrix, setColorMap

from Data import Sequntial_RGB_Dataset, ImageNet_Dataset

from Networks.GRU_models import Base_Model
from Networks.ImageNet_models import ResNet18_Model
from Networks.viterbi import Viterbi

from LogHandler import LogHandler
from Trainer import Trainer
from Losses.losses import weighted_bce_loss, statematrixloss
from Augmentations import Compose, RandomVerticalFlip, RandomHorizontalFlip, RandomRotation

parser = argparse.ArgumentParser(description='Parameters')
#defining arguments for training

parser.add_argument('--input_folder',type=str, default="./data_triet", help="Path to the image sequences") #CellMorphology_ZhongNMeth

parser.add_argument('--out_path',type=str, default="output_triet",help="Will generate during the run")

#architecture
parser.add_argument('--architecture',type=str, default="basemodel",help="Choose the architecture between ['basemodel', 'resnet18']")
parser.add_argument('--dataset',type=str, default="LivecellMiner",help="Both datasets the groundtruth filenames are different. Choose the dataset between ['LiveCellMiner', 'ZhongMorphology']")

#parameters
parser.add_argument('--iter',type=int, default= 500 , help="Number of iterations")
parser.add_argument('--batch_size',type=int, default= 4, help="Size of the batch, for deeptracking it is 1")
parser.add_argument('--lr',type=float, default=0.01, help="learning rate")
parser.add_argument('--loss2_regulerizer',type=float, default= 0.1 , help="Loss 2 regulerizer")
parser.add_argument('--is_viterbi',type=bool, default= False, help="To set the viterbi decoding for classification loss(Weakly-supervised training)")
parser.add_argument('--is_viterbi_int',type=int, default= 0, help="To set the viterbi decoding for classification loss with int values 0 is false else true(Weakly-supervised training)")
parser.add_argument('--n_classes',type=int, default= 5, help="Number of label classes to be classified")
parser.add_argument('--loss2_occurance',type=int, default= 1, help="Calculate loss2 only once in every nth occurance of loss1")
parser.add_argument('--train_different_length',type=bool, default= True, help="Set true if differnt length sequence to be trained.")
parser.add_argument('--train_to_test_ratio',type=float, default= 0.8, help="Train to test ratio for LivecellMiner. Zhong testdataset is chosen by folder name")

parser.add_argument('--is_second_lr',type=bool, default= True, help="Set true if a second learning rate to be set after nth epoch.")
parser.add_argument('--epoch_second_lr',type=int, default= 2000, help="The epoch at which a second learning rate to be set")
parser.add_argument('--second_lr_ratio',type=float, default=0.1, help="the ratio to be multiplied with the first learning rate to get second lr")

parser.add_argument('--steps',type=int, default=0, help="Count the steos during training")
parser.add_argument('--N',type=int, default= 40, help="Length of each sequence")

parser.add_argument('--use_pretrained_model',type=bool, default=False, help="Whether to use the pretrained model to start the training")
parser.add_argument('--test_only',type=bool, default=False, help="test only. require saved_model_path")
parser.add_argument('--saved_model_path',type=str, default="./models/trained_model_10000.pth",help="Path of the saved model")
parser.add_argument('--continue_training',type=bool, default=False, help="If training need to be continued from latest saved model")
args = parser.parse_args()


if __name__ == "__main__":

    #set viterbi from viterbi int
    args.is_viterbi = False if args.is_viterbi_int == 0 else True

    n_classes = args.n_classes
    loss2_occurance = args.loss2_occurance
    is_viterbi = args.is_viterbi
    transcripts = []

    #output path
    script_path = os.path.dirname(os.path.realpath(__file__))
    args.out_path = os.path.join(script_path,args.out_path)
    #create if not exists
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    #Start the loghandler
    Logger = LogHandler(name = "Rijo", basepath=args.out_path)

    #using GPU if available
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)

    nets = {
        'basemodel'         : Base_Model,
        'resnet18'          : ResNet18_Model,
    }

    assert args.architecture in nets.keys()

    #define the model type for training
    if args.architecture == 'resnet18' or args.architecture == 'densenet121':
        model_type = 'img'      #work based on images and not on sequence
    else:
        model_type = 'seq'      #work based on sequences and not as individual images

    #define image format
    if args.architecture == 'basemodel' :
        image_format = 'gray'
    else:
        image_format = 'RGB'

    #input image path
    base_path = args.input_folder

    #load images and ground truth based on the datasets (Images folder names start with P0037 is used for Test in Zhong.)
    if args.dataset == 'ZhongMorphology':
        all_frames_train, all_frames_test, train, test = load_image_paths_zhong( base_path, n_classes,  type = model_type)
    elif args.dataset == 'LivecellMiner':
        all_frames_train, all_frames_test, train, test = load_image_paths_livecellminer(base_path, n_classes = n_classes, train_to_test_ratio = args.train_to_test_ratio, type = model_type)
    else:
        print ('Error: Invalid datatype')
        assert(1==0)

    #store train and test folders into files
    with open(os.path.join(args.out_path,'split1.train'), 'w') as the_file:
        for n in train:
            the_file.write(n+"\n")
    with open(os.path.join(args.out_path,'split1.test'), 'w') as the_file:
        for n in test:
            the_file.write(n+"\n")

    #Adding the augmentation to the dataset
    transforms = Compose([ RandomHorizontalFlip(),
                           RandomVerticalFlip(),
                           RandomRotation([90.0, 90.0]),
                           RandomRotation([270.0, 270.0]),
                           RandomRotation([180.0, 180.0])
                                    ])

    #Creating custom dataset and dataloader
    if model_type == 'img':
        dataset = ImageNet_Dataset(all_frames_train, n_classes = n_classes, transform = transforms)
        testset = ImageNet_Dataset(all_frames_test,  n_classes = n_classes)
        dataloader = DataLoader(dataset, batch_size= args.batch_size, shuffle= True)
        testloader = DataLoader(testset, batch_size= 40 , shuffle= False)
    else:
        dataset = Sequntial_RGB_Dataset(all_frames_train, n_classes = n_classes, image_format = image_format, transform = transforms, dataset = args.dataset)
        testset = Sequntial_RGB_Dataset(all_frames_test,  n_classes = n_classes, image_format = image_format, dataset = args.dataset)
        dataloader = DataLoader(dataset, batch_size= args.batch_size, shuffle= True)
        testloader = DataLoader(testset, batch_size= 1 , shuffle= False)

    #For Displaying the images dataset details
    data = next(iter(dataloader))["image"]
    print (data.shape)
    Logger.log('info', "-----DATASET DETAILS-----")
    width = data.shape[-1]
    height = data.shape[-2]
    Logger.log('info', "Images has size " + str(width) + " x " + str(height))
    if model_type == 'seq':
        N = data.shape[1]
        Logger.log('info', "Length of each sequence " + str(N))
        args.N = N
    # -- total number of training sequences
    M = data.shape[0]
    Logger.log('info', "Batch Size: " + str(M))
    #Total number of batches
    Logger.log('info', "Number of training batches " + str( len(dataloader)))
    Logger.log('info', "Number of testing batches  " + str( len(testloader)))
    Logger.log('info', "Training with Viterbi " + str(args.is_viterbi))
    Logger.log('info', "Training model " + str(args.architecture))

    #=====Model Network for training=====
    if model_type == 'img':
        model = nets[args.architecture](n_classes = n_classes)
    else:
        model = nets[args.architecture](n_classes = n_classes, num_layers=3, device = device_type, is_viterbi= is_viterbi)
    model = model.to(device)

     #===== For storing the train and test losses for evaluation=====
    trainloss = []
    trainloss1 = []
    trainloss2 = []
    validationloss = []
    validationloss1 = []
    validationloss2 = []
    epochs = []
    args.steps = 0

    #initialization for viterbi
    prior = np.ones((n_classes), dtype=np.float32) / n_classes
    mean_lengths = np.ones((n_classes), dtype=np.float32)

    #continue training from last iteration
    if args.continue_training == True:
        models = []
        for files in os.listdir(args.out_path):
            if ".pth" in files:
                models.append( int(files.split("_")[-1].replace(".pth","")) )
        if len(models) > 0:
            models.sort()
            load_model_path = "autoencoder_trained_" + str(models[-1]) + ".pth"
            iternum = int(load_model_path.split("_")[-1].replace(".pth",""))
            os.rename(os.path.join(args.out_path,"autoencoder_training_progress.png"),os.path.join(args.out_path,"autoencoder_training_progrss_"+ str(iternum) +".png") )
            Logger.log('info',"starting training from epoch " + str(iternum) )
            trained_model = torch.load(os.path.join(args.out_path,load_model_path))
            prior =  np.loadtxt(os.path.join(args.out_path,'prior.iter_' + str(iternum) + '.txt') )
            mean_lengths =np.loadtxt(os.path.join(args.out_path,'legths.iter_' + str(iternum) + '.txt'))
            model.load_state_dict(trained_model)
            args.steps = iternum

    #if load from a pretrained model
    if args.use_pretrained_model or args.test_only:
        trained_model = torch.load(args.saved_model_path)
        model.load_state_dict(trained_model)

    #=====Criterion and Optimizer=====
    criterion1 = weighted_bce_loss  #nn.MSELoss()

    if not is_viterbi:
        criterion2 = statematrixloss
    else:
        criterion2 = nn.NLLLoss()    #nn.CrossEntropyLoss() #

    if model_type == 'img':
        criterion2 = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adagrad(model.parameters(), lr = args.lr, weight_decay= 1e-5)

    decoder = Viterbi(None, None, frame_sampling = 1, max_hypotheses = np.inf)
    trainer = Trainer(model, model_type, criterion1, criterion2 , optimizer, Logger, device_type, n_classes, args.loss2_occurance, args.train_different_length, is_viterbi, decoder, prior, mean_lengths)

    cost = 0
    loss1 = 0
    loss2 = 0
    loss3 = 0
    loss4 = 0

    if args.test_only:
        alldata_loader = DataLoader(dataset + testset, batch_size= 1 , shuffle= False)
        trainer.test_step(alldata_loader, args)
    else:
        #=====Iteration over Epochs=====
        for iter in range(args.iter):

            start = time.time() #Start time of epochs
            Logger.log('info', '\nepoch: '+ str(iter+1) +  " out of " + str(args.iter) )

            #=====Iteration over mini batches=====
            for didx,input in enumerate(dataloader):

                cost1,losses,transcripts = trainer.train_step(input, args, transcripts)
                cost += cost1
                loss1 += losses[0]/100
                loss2 += losses[1][0]/(100/loss2_occurance)

                #Count number of steps done
                args.steps += 1

                #save grammar after every iteration
                with open(os.path.join(args.out_path,'grammar.txt'), 'w') as the_file:
                    for transcript in transcripts:
                        the_file.write(str(transcript)+"\n")

                #validate model in every 200 steps
                if args.steps % 200 == 0 :
                    #evaluate model
                    val_loss, val_losses = trainer.validation_step(testloader, args, transcripts)
                    val_loss1 = val_losses[0]
                    val_loss2 = val_losses[1]

                    validationloss1.append(val_loss1)
                    validationloss2.append(val_loss2)

                    #===== Adding loss into a list=====
                    if len(losses[1]) == 1:
                        trainloss.append(loss1+loss2)
                    else:
                        trainloss.append(loss1+loss2+loss3+loss4)

                    Logger.log('info', "Average Loss in step "+ str(args.steps) + " is :" + str(cost.item()/100))
                    cost = 0
                    validationloss.append(val_loss.item())
                    epochs.append(args.steps)

                    trainloss1.append(loss1)
                    trainloss2.append(loss2)
                    plot_loss(args,epochs,trainloss,trainloss2,label1 ='loss1', label2 = 'loss2', name = 'training_progress')

                    if model_type == 'seq':
                        plot_losses(args, epochs, trainloss1, trainloss2, validationloss1, validationloss2, name='loss_progress' )
                    else:
                        plot_losses(args, epochs, trainloss2, trainloss2, validationloss2, validationloss2, name='loss_progress' )

                    loss1 = 0
                    loss2 = 0

                #save model every n steps
                if args.steps % 1000 == 0:
                    trainer.save_model(args)
                    trainer.test_step(testloader,args)

                #Break after 20000 iterations
                if args.steps == 20000:
                    break

            Logger.log('info', "Epoch took " + str(round(time.time()- start,2)) + " seconds")

