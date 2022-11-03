# END-TO-END_CLASSIFICATION_OF_CELL-CYCLE_STAGES_WITH_CENTER-CELL_FOCUS_TRACKER_USING_RNN
Code to the paper "END-TO-END CLASSIFICATION OF CELL-CYCLE STAGES WITH CENTER-CELL FOCUS TRACKER USING RECURRENT NEURAL NETWORKS" using PyTorch.


This repository contains the python code using the PyTorch module for the training and evaluation of our proposed model which uses RNN layers for propagating time information for classifying the mitosis stages. The code works with the 2D cell dataset with the cell in focus at the center of the image and tracked over time as shown in the video below. Two different datasets one with 3 classes (LiveCellMiner) and the other with 6 classes (Zhong Morphology) are used for the experiments.

<p align="center">
  <img src="cell_sequence.gif" alt="animated" />
</p>

**DATASETS** <br />
to do

**TRAINING** <br />
The training of the model can be started by running 'start_train.py' file. The arguments for this script can be seen in the first part of the code. The image path can be set here. The user can select the model between our proposed model or the ResNet18 classification model from the arguments. The LiveCellMiner dataset and the Zhong Morphology datasets can be trained with this script. To work with a new dataset where the file names are different, the 'load_images_paths' function in the file 'Helper_Functions.py' needed to be updated.

**EVALUATION** <br />
After the execution of 'start_train.py' the results will be stored in a folder created in the same path with the name given as in the arguments. In the folder, the training curves along with the results from every 1000 iterations are saved. The results include the label matrix plots, confusion matrixes, and additionally ('Results.txt') the precision, recall, and f1 score values. The foldernames of the train and test split are also found in this folder. The validation data selected is a part of the test dataset.
