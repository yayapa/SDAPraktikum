# Causal Relation

## Converter
First convert the original datasets into the [CREST](https://github.com/phosseini/CREST) format.
Then, convert it into [CNN](https://github.com/onehaitao/CNN-relation-extraction) format to use it with dataloader and word embedding.
The converter can be found in ***converter2cnn.ipynb***

TODO: Check the converter
## Baseline
Baseline uses the data loader from [CNN](https://github.com/onehaitao/CNN-relation-extraction) to get word embedding.
The baseline can be found in ***baseline.py***

TODO: make appropriate metrics, Grid Search?, Cross Validation?
TODO: get rid of args in the class Config to use it on jupyter notebook

## CNN
CNN model is based on this [repository](https://github.com/onehaitao/CNN-relation-extraction)

TODO: Test on other datasets in colab
TODO: Should we use original 19 labels or just 3 for causal relation?