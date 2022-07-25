# Causal Relation
BEFORE YOU START:
download und unpack the embedding into CNN/embedding/ from [here](http://metaoptimize.s3.amazonaws.com/hlbl-embeddings-ACL2010/hlbl-embeddings-scaled.EMBEDDING_SIZE=50.txt.gz), if the folder is empty.
## Preprocessing
First, original datasets were converted into the [CREST](https://github.com/phosseini/CREST) format and saved in ***CNN/data/data_xlsx***
Then, into [CNN](https://github.com/onehaitao/CNN-relation-extraction) format to use it with dataloader and word embedding.
The converter can be found in ***converter2cnn.ipynb*** and already converted datasets are saved in ***CNN/data/simeval2007*** and ***CNN/data/simeval2010***

## Baseline
Baseline uses the dataloader from [CNN](https://github.com/onehaitao/CNN-relation-extraction) to get word embedding.
The baseline can be found in ***Baseline.ipynb***

Cross-validation can improve the performance of baseline, but can not provide an effective improvement for the CNN model.
Hence, we come to a conclusion that cross-validation is only useful under the following situations:
1. The model is under-fitting (the model is too easy).
2. The dataset is too small.
3. The distribution of data is un-uniform.

## Metrics
Based on the [survey](https://link.springer.com/content/pdf/10.1007/s10115-022-01665-w.pdf), the following metrics are chosen:
1. Accuracy
2. Precision, Recall, F1-score
3. MCC and G-Mean

They are implemented in ***custom_statistics.py***


## CNN
Many approaches are investigated using these sources:
1. [GitHub survey](https://github.com/zhijing-jin/Causality4NLP_Papers?ysclid=l5p9lwwc4n1073062)
2. [Yang et al.](https://github.com/zhijing-jin/Causality4NLP_Papers?ysclid=l5p9lwwc4n1073062)
3. [Paperswithcode](https://paperswithcode.com/task/relation-extraction)

The CNN approach is selected. This is based on the [paper](https://aclanthology.org/C14-1220.pdf) and the [repository](https://github.com/onehaitao/CNN-relation-extraction)
The implementation can be found in CNN module and the training started with ***CNN.ipynb***

## Results
Results are available in the presentation ***PSDA_ÜB4.pdf***