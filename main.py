import json
from hmm_tagger import HMMTagger
from data_utils import load_and_split_data, convert_to_collapsed_tags
from evaluation import evaluate_tagger

def main():
    """Main function to run the HMM POS tagger with different configurations."""
    try:
        # Load data
        with open('penn-data.json', 'r') as f:
            json_data = json.load(f)

        # Split data
        train_data, test_data = load_and_split_data(json_data)

        # Test configurations
        configs = [
            (1, False, "First Order HMM"),
            (2, False, "Second Order HMM"),
            (1, True, "First Order HMM with word dependency")
        ]

        for order, word_depends_prev, name in configs:
            print(f"\n{'-'*50}")
            print(f"Testing {name}")
            print(f"{'-'*50}")

            # 36-tag version
            tagger = HMMTagger(order=order, word_depends_prev=word_depends_prev)
            tagger.train(train_data)
            accuracy, tag_accuracy = evaluate_tagger(tagger, test_data)
            print(f"\n{name} - 36 tags:")
            print(f"Overall accuracy: {accuracy:.4f}")
            print("Tag-wise accuracy:", tag_accuracy)

            # 4-tag version
            collapsed_train = convert_to_collapsed_tags(train_data)
            collapsed_test = convert_to_collapsed_tags(test_data)
            collapsed_tagger = HMMTagger(order=order, word_depends_prev=word_depends_prev)
            collapsed_tagger.train(collapsed_train)
            collapsed_accuracy, collapsed_tag_accuracy = evaluate_tagger(
                collapsed_tagger,
                collapsed_test
            )
            print(f"\n{name} - 4 tags:")
            print(f"Overall accuracy: {collapsed_accuracy:.4f}")
            print("Tag-wise accuracy:", collapsed_tag_accuracy)

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
