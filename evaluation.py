from collections import defaultdict
from typing import List, Dict, Tuple
from hmm_tagger import HMMTagger

def evaluate_tagger(tagger: HMMTagger, test_data: List[List[Tuple[str, str]]]) -> Tuple[float, Dict[str, float]]:
    """Evaluate tagger performance on test data."""
    total_correct = 0
    total_words = 0
    unseen_words_correct = 0
    unseen_words_total = 0
    tag_correct = defaultdict(int)
    tag_total = defaultdict(int)

    for sentence in test_data:
        words = [word for word, tag in sentence]
        true_tags = [tag for word, tag in sentence]
        pred_tags = tagger.tag_sentence(words)

        for word, true_tag, pred_tag in zip(words, true_tags, pred_tags):
            if true_tag == pred_tag:
                total_correct += 1
                tag_correct[true_tag] += 1
            total_words += 1
            tag_total[true_tag] += 1

            if word not in tagger.vocabulary:
                unseen_words_total += 1
                if true_tag == pred_tag:
                    unseen_words_correct += 1

    # Calculate accuracies
    overall_accuracy = total_correct / total_words if total_words > 0 else 0
    tag_accuracy = {
        tag: tag_correct[tag] / tag_total[tag]
        for tag in tag_total if tag_total[tag] > 0
    }

    # Print unseen word performance
    if unseen_words_total > 0:
        unseen_accuracy = unseen_words_correct / unseen_words_total
        print(f"\nUnknown word performance:")
        print(f"Total unknown words: {unseen_words_total}")
        print(f"Unknown word accuracy: {unseen_accuracy:.4f}")

    return overall_accuracy, tag_accuracy
