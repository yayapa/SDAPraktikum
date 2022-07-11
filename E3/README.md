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
