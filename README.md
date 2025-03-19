# HMM-POS-Tagger-From-Scratch

A complete implementation of Hidden Markov Model (HMM) for Part-of-Speech (PoS) tagging built entirely from scratch, without using any existing HMM or NLP libraries.

## Overview

Part-of-speech tagging assigns grammatical categories to every token in a sentence. This project implements a PoS tagger using three different HMM configurations - all coded from scratch:

1. First Order HMM where the probability of a word depends only on the current tag
2. Second Order HMM where the probability depends only on the current tag
3. First Order HMM where the probability of a word depends on the current tag as well as the previous word

The implementation also includes a comparison between using the full 36-tag set from the Penn Treebank and a collapsed 4-tag set (N, V, A, O).

## Key Features

- **Pure Python Implementation**: All HMM components including the Viterbi algorithm are implemented from scratch without relying on external NLP libraries
- **Multiple HMM Configurations**: Supports first-order, second-order, and word-dependent emission models
- **Tag Set Flexibility**: Works with both the full 36-tag Penn Treebank set and a collapsed 4-tag set
- **Unseen Word Handling**: Robust handling of words not seen during training

## Dataset

The code uses the English Penn Treebank (PTB) corpus which contains 36 distinct part-of-speech tags.

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/mitesh-kr/HMM-POS-Tagger-From-Scratch.git
cd HMM-POS-Tagger-From-Scratch
```

### Install Requirements

```bash
pip install -r requirements.txt
```

### Running the Project

1. Download the Penn Treebank dataset and save it as `penn-data.json` in the project directory.
2. Run the main script:

```bash
python main.py
```

This will train and evaluate all three HMM configurations with both the 36-tag and 4-tag sets.

## Project Structure

```
HMM-POS-Tagger-From-Scratch/
│
├── hmm_tagger.py        # Core HMM implementation and Viterbi algorithm
│
├── data_utils.py        # Data loading, preprocessing, and tag collapsing
│
├── evaluation.py        # Functions for evaluating model performance
│
├── main.py              # Main script to run experiments
│
├── requirements.txt     # Project dependencies
│
├── .gitignore           # Git ignore file
│
└── README.md            # This file
```

### File Descriptions

- **hmm_tagger.py**: Contains the HMMTagger class with methods for:
  - Building emission and transition probability matrices
  - Implementing the Viterbi algorithm
  - Training the model and tagging new sentences

- **data_utils.py**: Includes functions for:
  - Loading and cleaning the Penn Treebank data
  - Splitting data into training and testing sets
  - Collapsing the 36 tags into 4 categories

- **evaluation.py**: Contains methods for:
  - Calculating overall accuracy
  - Computing tag-wise accuracy
  - Evaluating performance on unseen words

- **main.py**: Entry point that:
  - Loads the dataset
  - Trains different HMM configurations
  - Evaluates and compares performance

## Requirements

- Python 3.6+
- NumPy
- JSON

## Implementation Details

### HMM Configurations

1. **First Order HMM**: Uses the probability of a word given its current tag and the probability of a tag given the previous tag.
2. **Second Order HMM**: Uses the probability of a tag given the two previous tags.
3. **First Order HMM with Word Dependency**: Extends the first-order model by making the emission probability dependent on both the current tag and the previous word.

### Tag Collapsing

The 36 tags from the Penn Treebank are collapsed into 4 categories:
- N: All noun tags (NN, NNS, NNP, NNPS)
- V: All verb tags (VB, VBD, VBG, VBN, VBP, VBZ)
- A: All adjective and adverb tags (JJ, JJR, JJS, RB, RBR, RBS)
- O: All other tags

### Handling Unseen Words

For words not encountered during training, the tagger assigns the most frequent tag from the training data.

## Evaluation

The tagger is evaluated using:
1. Overall accuracy
2. Tag-wise accuracy for each tag
3. Special analysis for unknown words (words not seen during training)

The results include a comparison between the 36-tag and 4-tag models.
