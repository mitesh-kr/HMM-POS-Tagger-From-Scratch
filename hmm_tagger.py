import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Set

class HMMTagger:
    def __init__(self, order: int = 1, word_depends_prev: bool = False):
        self.order = order
        self.word_depends_prev = word_depends_prev

        # Simplified data structures using nested defaultdict
        if word_depends_prev:
            self.emission_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        else:
            self.emission_counts = defaultdict(lambda: defaultdict(int))

        self.transition_counts = defaultdict(lambda: defaultdict(int))

        if order == 2:
            self.second_order_counts = defaultdict(lambda: defaultdict(int))

        self.tag_counts = defaultdict(int)
        self.vocabulary = set()
        self.most_frequent_tag = None

    def train(self, sentences: List[List[Tuple[str, str]]]):
        """Two-pass training approach for better statistics."""
        # First pass: collect word and tag statistics
        for sentence in sentences:
            for word, tag in sentence:
                self.vocabulary.add(word)
                self.tag_counts[tag] += 1

        # Set most frequent tag for unseen words
        self.most_frequent_tag = max(self.tag_counts.items(), key=lambda x: x[1])[0]
        print(f"Most frequent tag (for unseen words): {self.most_frequent_tag}")

        # Second pass: collect transition and emission statistics
        for sentence in sentences:
            previous_tag = '<START>'
            previous_tag2 = '<START>' if self.order == 2 else None
            previous_word = '<START>' if self.word_depends_prev else None

            for word, tag in sentence:
                # Update emission counts
                if self.word_depends_prev:
                    self.emission_counts[tag][previous_word][word] += 1
                else:
                    self.emission_counts[tag][word] += 1

                # Update transition counts
                self.transition_counts[previous_tag][tag] += 1
                if self.order == 2:
                    self.second_order_counts[(previous_tag2, previous_tag)][tag] += 1

                # Update previous states
                previous_word = word if self.word_depends_prev else None
                previous_tag2 = previous_tag if self.order == 2 else None
                previous_tag = tag

        self._normalize_probabilities()

    def _normalize_probabilities(self):
        """Normalize counts to probabilities with simple smoothing."""
        # Normalize transition probabilities
        for prev_tag in self.transition_counts:
            total = sum(self.transition_counts[prev_tag].values())
            if total > 0:
                for tag in self.transition_counts[prev_tag]:
                    self.transition_counts[prev_tag][tag] /= total

        # Normalize second-order transitions if applicable
        if self.order == 2:
            for prev_tags in self.second_order_counts:
                total = sum(self.second_order_counts[prev_tags].values())
                if total > 0:
                    for tag in self.second_order_counts[prev_tags]:
                        self.second_order_counts[prev_tags][tag] /= total

        # Normalize emission probabilities
        if self.word_depends_prev:
            for tag in self.emission_counts:
                for prev_word in self.emission_counts[tag]:
                    total = sum(self.emission_counts[tag][prev_word].values())
                    if total > 0:
                        for word in self.emission_counts[tag][prev_word]:
                            self.emission_counts[tag][prev_word][word] /= total
        else:
            for tag in self.emission_counts:
                total = sum(self.emission_counts[tag].values())
                if total > 0:
                    for word in self.emission_counts[tag]:
                        self.emission_counts[tag][word] /= total

    def get_emission_probability(self, tag: str, word: str, prev_word: str = None) -> float:
        """Simplified emission probability calculation with basic smoothing."""
        if self.word_depends_prev:
            if word not in self.vocabulary:
                return 1.0 if tag == self.most_frequent_tag else 1e-10
            return self.emission_counts[tag][prev_word].get(word, 1e-10)
        else:
            if word not in self.vocabulary:
                return 1.0 if tag == self.most_frequent_tag else 1e-10
            return self.emission_counts[tag].get(word, 1e-10)

    def viterbi(self, sentence: List[str]) -> List[str]:
        """Simplified Viterbi implementation without log probabilities."""
        n = len(sentence)
        tags = list(self.tag_counts.keys())

        viterbi = defaultdict(lambda: defaultdict(float))
        backpointer = defaultdict(lambda: defaultdict(str))

        # Initialize for first word
        prev_word = '<START>'
        for tag in tags:
            emission_prob = self.get_emission_probability(tag, sentence[0], prev_word)
            viterbi[tag][0] = self.transition_counts['<START>'].get(tag, 1e-10) * emission_prob

        # Process rest of sentence
        for i in range(1, n):
            prev_word = sentence[i-1] if self.word_depends_prev else None
            for tag in tags:
                emission_prob = self.get_emission_probability(tag, sentence[i], prev_word)

                if self.order == 1:
                    max_prob, best_prev_tag = max(
                        (viterbi[prev_tag][i-1] *
                         self.transition_counts[prev_tag].get(tag, 1e-10) *
                         emission_prob, prev_tag)
                        for prev_tag in tags
                    )
                else:
                    if i == 1:
                        max_prob, best_prev_tag = max(
                            (viterbi[prev_tag][i-1] *
                             self.transition_counts[prev_tag].get(tag, 1e-10) *
                             emission_prob, prev_tag)
                            for prev_tag in tags
                        )
                    else:
                        max_prob, best_prev_tag = max(
                            (viterbi[prev_tag][i-1] *
                             self.second_order_counts.get((backpointer[prev_tag][i-1], prev_tag), {}).get(tag, 1e-10) *
                             emission_prob, prev_tag)
                            for prev_tag in tags
                        )

                viterbi[tag][i] = max_prob
                backpointer[tag][i] = best_prev_tag

        # Backtrack to find best sequence
        best_last_tag = max(tags, key=lambda tag: viterbi[tag][n-1])
        best_sequence = [best_last_tag]
        for i in range(n-1, 0, -1):
            best_sequence.insert(0, backpointer[best_sequence[0]][i])

        return best_sequence

    def tag_sentence(self, sentence: List[str]) -> List[str]:
        """Tag a sentence."""
        if not sentence:
            return []
        return self.viterbi(sentence)
