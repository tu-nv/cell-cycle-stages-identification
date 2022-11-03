# END-TO-END_CLASSIFICATION_OF_CELL-CYCLE_STAGES_WITH_CENTER-CELL_FOCUS_TRACKER_USING_RNN
Code to the paper "END-TO-END CLASSIFICATION OF CELL-CYCLE STAGES WITH CENTER-CELL FOCUS TRACKER USING RECURRENT NEURAL NETWORKS" using PyTorch.


This repository contains the python code written using the PyTorch module for the training and evaluation of our proposed model which uses RNN layers for propagating time information. The code works with the 2D cell dataset with the cell in focus at the center of the image and tracked over time as shown in the video below. Two different datasets one with three classes(LiveCellMiner) and the other with 6 classes(Zhong Morphology) are used for the experiments.

<p align="center">
  <img src="cell_sequence.gif" alt="animated" />
</p>

**DATASETS** <br />
to do

**TRAINING** <br />
The training of the model can be started by running 'start_train.py' file. The arguments for this script can be seen in the first part of the code. The image path can be set here. The user can select the model between our proposed model or the ResNet18 classification model from the arguments. The LiveCellMiner dataset and the Zhong Morphology datasets can be trained with this script. To work with a new dataset where the file names are different, the 'load_images_paths' function in the file 'Helper_Functions.py' may be needed to be updated.

**EVALUATION** <br />
