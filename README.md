# Named Entity Recognition with tensorflow

This repo implements a NER model using Tensorflow, with a choice of hyperparameters and training dataset.

## Task

Given a sentence, give a tag to each word. A classical application is Named Entity Recognition (NER). Here is an example

```
John   lives in New   York
B-PER  O     O  B-LOC I-LOC
```

## Getting started

1. Download the GMB dataset and process into a common format. For now this will download v1.0.0 which is very lightweight. 

```
make gmb
```
This will download v1.0.0 (lightest version for now) and store under the `data` directory. It is then processed into a common format to be processed for modelling.
TODO: It seems that different versions of GMB and CoNLL datasets have different structures, so this processor will need to expand to incorporate the version.

2. 