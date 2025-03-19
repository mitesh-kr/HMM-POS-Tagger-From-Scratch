import numpy as np
import json
import re
from typing import List, Dict, Tuple, Set

def load_and_split_data(json_data: List[List], train_ratio: float = 0.8) -> Tuple[List[List[Tuple[str, str]]], List[List[Tuple[str, str]]]]:
    """Load and preprocess data, then split into training and testing sets."""
    processed_data = []

    for sentence_data in json_data:
        # Simple cleaning
        text = sentence_data[0]
        tags = sentence_data[1]

        # Basic cleaning
        text = re.sub(r',', '', text)
        text = re.sub(r'\.(?=[^\.]*$)', '', text)
        words = text.split()

        # Create word-tag pairs
        if len(words) == len(tags):
            sentence_pairs = list(zip(words, tags))
            processed_data.append(sentence_pairs)

    # Simple random split
    np.random.seed(42)
    split_idx = int(len(processed_data) * train_ratio)
    indices = list(range(len(processed_data)))
    np.random.shuffle(indices)

    train_data = [processed_data[i] for i in indices[:split_idx]]
    test_data = [processed_data[i] for i in indices[split_idx:]]

    return train_data, test_data

def collapse_tags(tag: str) -> str:
    """Collapse tags into 4 categories: N, V, A, O."""
    if tag in {'NN', 'NNS', 'NNP', 'NNPS'}:
        return 'N'
    elif tag in {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}:
        return 'V'
    elif tag in {'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'}:
        return 'A'
    else:
        return 'O'

def convert_to_collapsed_tags(data: List[List[Tuple[str, str]]]) -> List[List[Tuple[str, str]]]:
    """Convert dataset to use collapsed tag set."""
    return [[(word, collapse_tags(tag)) for word, tag in sentence] for sentence in data]
