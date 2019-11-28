# Introduction

Damper-Defect-Detection-Using-CNN is code for classifying driving data according to its damper health state.

# Installation

Create a python virtual environment. The code is developed using Python 3.5.2. Install the requirements according to the requirements file.
Unzip the file Datensatz\DataForCNN\data\DD2_raw_512_FlexRay.zip within its folder. This was required for data size reasons

# Usage

Run the main scripts in the folder Skripte/CNN/

run Skripte/CNN/mainArchVar.py for CNN network architecture analysis.

run Skripte/CNN/mainOptimPreProcessing.py to run hyperparameter optimization of L2 and learning rate hyperparameter

run Skripte/CNN/mainPreProcessingAnalysis.py for evaluation of different pre-processing methods

run Skripte/CNN/mainSingleNetwork.py to train a single network, e. g. a MLP