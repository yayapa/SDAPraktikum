# Pre-training Approaches
The datasets should be placed in the same folder as ***Pretraining.ipynb***.
## Pretraining on another task 
Based on [Huh16](https://arxiv.org/abs/1608.08614).
Use jupyter notebook ***Pretraining.ipynb*** to test the approach: 
the network 1 is trained on one dataset and the network 2 is initialized with the weights of network 1 and is trained from scratch.

## Self-training on transforms
Self-training on pretext task (transformation). Based on [Yuan22](https://arxiv.org/abs/2206.02909) 
To generate all transformations, use jupyter notebook ***Transformation.ipynb***.
Two different approaches are tested in jupyter notebook ***SelfTraining.ipynb***:
1. Binary Classification, whether the transformation is applied
2. Multi-class Classification, which transformation is applied

## Teacher and Student Model
The basic idea is transfer learning (TL): take full advantage of old knowledge rather than train a new model from zero when getting a new similar task

The process is as follows:
1) Train a teacher model with the Opportunity dataset
2) Store the teacher model and extract the optimal parameters (weights, w)
3) Load the graph and weights w in the student model
4) Train student model with the Daphnet dataset using the weights w

Files Description:
1) datareader.py: read dataset
2) teacher.py: train a teacher model
3) model: contain all the information about the teacher model
4) student.py: train a student model

How to run?
1) Download datasets and store them in the corresponding folder
   a. Opportunity dataset (it can be downloaded from: https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition)
   b. Daphnet Gait dataset (it can be downloaded from: https://archive.ics.uci.edu/ml/datasets/Daphnet+Freezing+of+Gait)
2) Run python datareader.py opp to get opportunity.h5; run python datareader.py dap to get daphnet.h5
3) Run python teacher.py opp using opportunity.h5 and then store all the information in the file model.
4) Run python student.py dap using daphnet.h5
