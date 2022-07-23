# Causal Relation
BEFORE YOU START:
download und unpack the embedding into CNN/embedding/ from [here](http://metaoptimize.s3.amazonaws.com/hlbl-embeddings-ACL2010/hlbl-embeddings-scaled.EMBEDDING_SIZE=50.txt.gz)
## Converter
First convert the original datasets into the [CREST](https://github.com/phosseini/CREST) format.
Then, convert it into [CNN](https://github.com/onehaitao/CNN-relation-extraction) format to use it with dataloader and word embedding.
The converter can be found in ***converter2cnn.ipynb***

TODO: Check the converter
## Baseline
Baseline uses the data loader from [CNN](https://github.com/onehaitao/CNN-relation-extraction) to get word embedding.
The baseline can be found in ***baseline.py***

Cross-validation can improve the performance of baseline, but can not provide an effective improvement for the CNN model.
Hence, we come to a conclusion that cross-validation is only useful under the following situations:
1. The model is under-fitting (the model is too easy).
2. The dataset is too small.
3. The distribution of data is un-uniform.

TODO: make appropriate metrics, Grid Search?, Cross Validation?
TODO: get rid of args in the class Config to use it on jupyter notebook (Done!)

## CNN
CNN model is based on this [repository](https://github.com/onehaitao/CNN-relation-extraction)

TODO: Test on other datasets in colab
TODO: Should we use original 19 labels or just 3 for causal relation?
