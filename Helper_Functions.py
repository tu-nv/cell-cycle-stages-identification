import os, csv, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def load_image_paths_livecellminer(base_path, n_classes = 3, train_to_test_ratio = 0.85, type = 'seq'):
    """
    Load all the training image paths from the folder given
    
    Arguments:       
        basepath            : basepath where the sequences are stored
        n_classes           : to create labels as in subclass or not
        type                : how to load images as seq or as img
        train_to_test_ratio : ratio of sequences for train to test

    Return: 
        all_frames_train    : Train dictionary with for each index a list of paths of sequence/image and corresponding label
        all_frames_test     : Test dictionary with for each index a list of paths of sequence/image and corresponding label
        train               : List of training sequence folder names (if type = seq) or train image paths (if type = img)
        test                : List of testing sequence folder names (if type = seq) or train image paths (if type = img)
    """
    # dictionary and list for storing test and train
    all_frames_train = {}
    all_frames_test = {}
    train = []
    test = []

    #starting values for train and test indices
    trainidx = -1
    testidx = -1
    
    #all folders in the base path
    all_folders = os.listdir(base_path)

    #train/test ratio multiplied by total 
    num_train = train_to_test_ratio * len(all_folders)

    #for each folder in the base path
    for folder in all_folders:    
        if "cell" in folder:
            
            csv_name = os.path.join (base_path, folder,folder + "_Synchronization.csv")
            with open(csv_name, "r") as f:
                reader = csv.reader(f, delimiter="\t")
                for i, line in enumerate(reader):
                    labels = np.array([ int(lab)-1 for lab in line[0].split(";") ])
                    true_labels = labels
                    idx2 = int(min(np.where(labels==2)[0])) + random.randint(15,22)
                    if n_classes == 4:
                        labels[idx2:] +=1 
                    else:
                        labels[idx2:] -= 2 

            #image paths
            path = os.path.join (base_path, folder, "Raw")      #inside each folder the images are under a folder named Raw
            files = []
            for file in os.listdir(path):
                #skipping some random files
                if "._" in file:                                #Exception handling for one of the random extra files
                    continue
                files.append(os.path.join(path,file))           #[seqm_img1,..,seqm_imgN]
            files.sort()

            #check whether load images based on sequence ('seq') or as images ('img') without sequence info
            if type == 'seq':
                if len(train) < num_train:
                    trainidx = trainidx + 1
                    train.append(folder)
                    all_frames_train[trainidx]  =  [ files, labels, true_labels]           #new_files, new_labels, new_true_labels,
                else: 
                    testidx += 1
                    test.append(folder)
                    all_frames_test[testidx]  =  [files, labels, true_labels]     # new_files, new_labels, new_true_labels, 
            else:
                if len(train) < num_train:
                    for f, l in zip(files,labels):
                        trainidx += 1 
                        all_frames_train[trainidx]  =  [ f, l ]  
                        train.append(f)      
                else: 
                    for f, l in zip(files,labels):
                        testidx += 1
                        test.append(f)
                        all_frames_test[testidx]  =  [ f, l ]  

    return all_frames_train, all_frames_test, train, test

def load_image_paths_zhong(base_path, n_classes = 6, type = 'seq'):
    """
    Load all the training image paths from the folder given
    
    Arguments:       
        basepath : basepath where the sequences are stored
        n_classes: to create labels as in subclass or not
        type     : how to load images as seq or as img

    Return: 
        all_frames_train    : Train dictionary with for each index a list of paths of sequence/image and corresponding label
        all_frames_test     : Test dictionary with for each index a list of paths of sequence/image and corresponding label
        train               : List of training sequence folder names (if type = seq) or train image paths (if type = img)
        test                : List of testing sequence folder names (if type = seq) or train image paths (if type = img)
    """
    # dictionary and list for storing test and train
    all_frames_train = {}
    all_frames_test = {}
    train = []
    test = []

    #starting values for train and test indices
    trainidx = -1
    testidx = -1
    
    #for each folder in the base path
    for folder in os.listdir(base_path):    
    
        # load labels
        csv_name = os.path.join (base_path, folder,folder + "_ManualStages.txt") 
        csv_name = csv_name.replace("B02","B01")
        with open(csv_name, "r") as f:
            reader = f.readline()

        #labels to nmpy array
        true_labels = np.array([ int(lab)-1 for lab in reader.split(",") ])
        labels  = true_labels.copy()

        #image paths
        path = os.path.join (base_path, folder, "Raw")      #inside each folder the images are under a folder named Raw
        files = []
        for file in os.listdir(path):
            #skipping some random files
            if "._" in file:                                #Exception handling for one of the random extra files
                continue
            files.append(os.path.join(path,file))           #[seqm_img1,..,seqm_imgN]
        files.sort()

        #check whether load images based on sequence ('seq') or as images ('img') without sequence info
        if type == 'seq':
            if "P0037" not in folder:
                trainidx = trainidx + 1
                train.append(folder)
                all_frames_train[trainidx]  =  [ files, labels, true_labels]           #new_files, new_labels, new_true_labels,
            elif 'B01' in folder: 
                testidx += 1
                test.append(folder)
                all_frames_test[testidx]  =  [files, labels, true_labels]     # new_files, new_labels, new_true_labels, 
        else:
            if "P0037" not in folder: 
                for f, l in zip(files,labels):
                    trainidx += 1 
                    all_frames_train[trainidx]  =  [ f, l ]  
                    train.append(f)      
            elif 'B01' in folder: 
                for f, l in zip(files,labels):
                    testidx += 1
                    test.append(f)
                    all_frames_test[testidx]  =  [ f, l ]  

    return all_frames_train, all_frames_test, train, test


def plot_loss(args, epochs, trainloss, validationloss, label1 ='Training loss', label2 = 'validation loss', name = 'training_progress'):
    """
    Plotting the loss functions
    arguments:
        args             : system arguments
        epochs           : a list of epoch values to be plotted (x axis)
        trainloss        : a list of the loss occured during training for the epochs
        validationloos   : a list of the loss occcured during validations
    """
    #===== Plotting the losses per epoch to see learning=====
    plt.figure(1)
    plt.plot(epochs, trainloss, 'green', label=label1)
    plt.plot(epochs, validationloss, 'red', label=label2)
    plt.title('Training and Validation loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    if args.steps <= 100:
        plt.legend()
    plt.savefig(os.path.join(args.out_path,"autoencoder_"+ name +".png"))


def plot_losses(args, epochs, trainloss1, trainloss2, validationloss1, validationloss2, name='loss_progress' ):
    '''
    Plotting 2 losses with train and validation curves sepeately
    arguments:
        args                            : system arguments
        epochs                          : a list of epoch values to be plotted (x axis)
        trainloss1,trainloss2           : a list of the losses occured during training for the epochs
        validationloss1,validationloss2 : a list of the losses occcured during validations
    '''
    plt.figure(2)
    plt.plot(epochs, trainloss1, 'green', label='trainloss1')
    plt.plot(epochs, validationloss1, 'cyan', label='validationloss1')
    plt.plot(epochs, trainloss2, 'red', label='trainloss2')
    plt.plot(epochs, validationloss2, 'orange', label='validationloss2')
    plt.title('Training and Validation loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    if args.steps <= 100:
        plt.legend()
    plt.savefig(os.path.join(args.out_path,"autoencoder_"+ name +".png"))


def plotconfusionmatrx(confus, args):
    '''
    A function to plot the confusion matrix given confus 6*6
    
    Arguments:
        confus  : A 6*6 numpy matrix
        args    : system arguments
    '''
    confus = np.array(confus)
    confus = confus.astype('float') / confus.sum(axis=1)[:, np.newaxis]
    labels = ['Interphase', 'Prophase', 'Prometaphase', 'Metaphase', 'Anaphase', 'Telophase']
    fig = plt.figure(figsize=(11,10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(confus,  cmap= plt.cm.get_cmap("YlOrRd"), interpolation='none')
    plt.title('Confusion matrix of the classifier')
    for (i, j), z in np.ndenumerate(confus):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
    fig.colorbar(cax)
    ax.set_xticklabels(['']+labels)
    ax.set_yticklabels(['']+labels)
    plt.xlabel('Predicted Values')
    plt.ylabel('True values')
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    plt.setp([tick.label1 for tick in ax.xaxis.get_major_ticks()], rotation=45,
         ha="right", va="center", rotation_mode="anchor")
    plt.savefig(os.path.join(args.out_path ,"showConfusionMatrix_"+ str(args.steps) +".jpg"), dpi=200)


def setColorMap(k):
    '''
    Color Map with predefined colors
    '''
    #Set color scheme for plots
    map = np.zeros((k,3))
    colorconstants = [(0.0,1.0,0.0),(1.0,0.965,0.263),(1.0,0.5,0.0),(0.824,0.553,0.808),(0.345,0.443,0.945),(1.0,0.0,0.0)]
    #colorconstants = [(0.345,0.443,0.945),(0.8,0.0,0.0),(0.824,0.553,0.808),(0.1,0.9,1.0),(0.0,0.7,0.0),(0.588,0.294,0.0)]

    cmap = mpl.colors.ListedColormap(colorconstants)
    return cmap