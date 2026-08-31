from datasets import load_dataset

from config import DATASET_NAME


def load_imdb():
    """Load the public IMDb dataset from Hugging Face."""
    return load_dataset(DATASET_NAME)


if __name__ == "__main__":
    dataset = load_imdb()
    print(dataset)
    print(dataset["train"][0])
