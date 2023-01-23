import torch
import logging
import numpy as np
import pandas as pd
import multiprocessing

from typing import List, Any, Dict, Tuple
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


def flatten(lists: List[List[Any]]) -> List[Any]:
    result = []
    for sublist in lists:
        result.extend(sublist)
    return result


# https://clay-atlas.com/us/blog/2021/08/09/pytorch-en-pad-pack-sequence/
# https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html
def padded_collate(data):
    # here -1 to remove the first token of the prediction
    x_lens_to_sort = [len(d["sentence"]) - 1 for d in data]
    # remove last element to remove <eos>
    x_to_sort = [d["sentence"][:-1] for d in data]
    # shift by one target to remove error computation on first element
    y_to_sort = [d["sentence"][1:] for d in data]
    x_lens = []
    x = []
    y = []
    for x_len_sample, x_sample, y_sample in sorted(
        zip(x_lens_to_sort, x_to_sort, y_to_sort), key=lambda x: x[0], reverse=True
    ):
        x_lens.append(x_len_sample)
        x.append(torch.tensor(x_sample))
        y.append(torch.tensor(y_sample))

    x_lens = torch.tensor(x_lens)
    x_padded = pad_sequence(x, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y, batch_first=True, padding_value=0)

    return x_padded, x_lens, y_padded


class InnerDataset:
    def __init__(self, filename: str):
        self.filename = filename
        self.sentences = self.load_dataset(filename)
        self.words = flatten(list(map(lambda sent: sent.split(), self.sentences)))
        self.vocabulary = set(self.words)

    def load_dataset(self, filename: str):
        return [line for line in open(filename)]

    def vocabulary_density(self, embedding: Dict[int, str]):
        count = np.zeros(len(embedding))
        for w in self.words:
            count[embedding[w]] += 1
        return count


class PeenTreeBankDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame, embedding: Dict[str, int]):
        self.df = dataset
        self.embedding = embedding
        self.apply_embedding()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> pd.Series:
        return self.df.iloc[index]

    def apply_embedding(self):
        self.df["sentence"] = self.df.sentence.apply(
            lambda x: [self.embedding[i] for i in x]
        )

    @staticmethod
    def compute_embedding(precomtued_filename: str) -> Dict[str, int]:
        logging.info("computing embedding.")
        df = pd.read_pickle(precomtued_filename)
        unique = sorted(list(set(flatten(list(df.sentence.values)))))
        return {v: i + 1 for i, v in enumerate(unique)}

    @staticmethod
    def get_datasets(
        precomputed_filename: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = pd.read_pickle(precomputed_filename)
        return (
            df[df.split == "train"].copy(deep=True),
            df[df.split == "validation"].copy(deep=True),
            df[df.split == "test"].copy(deep=True),
        )

    @staticmethod
    def get_loaders(
        precomputed_filename: str, batch_size: int, is_test: bool = False
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        embedding = PeenTreeBankDataset.compute_embedding(precomputed_filename)
        train_df, val_df, test_df = PeenTreeBankDataset.get_datasets(
            precomputed_filename
        )
        logging.info("pytorch loaders creation")
        train_df = PeenTreeBankDataset(train_df, embedding)
        val_df = PeenTreeBankDataset(val_df, embedding)
        test_df = PeenTreeBankDataset(test_df, embedding)
        return (
            DataLoader(
                train_df,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=padded_collate,
                drop_last=True,
                num_workers=multiprocessing.cpu_count() if not is_test else 1,
            ),
            DataLoader(
                val_df,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=padded_collate,
                drop_last=True,
                num_workers=multiprocessing.cpu_count() if not is_test else 1,
            ),
            DataLoader(
                test_df,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=padded_collate,
                drop_last=True,
                num_workers=multiprocessing.cpu_count() if not is_test else 1,
            ),
        )


def generate_dataset_precomputed(
    train_filename: str,
    validation_filename: str,
    test_filename: str,
    precomputed_filename: str,
):
    logging.info("Loading datasets.")
    train_df = InnerDataset(train_filename)
    val_df = InnerDataset(validation_filename)
    test_df = InnerDataset(test_filename)
    samples = []
    logging.info("Samples creation.")
    for split, split_name in zip(
        [train_df, val_df, test_df], ["train", "validation", "test"]
    ):
        for sentence in split.sentences:
            samples.append([sentence.split() + ["<eos>"], split_name])

    logging.info("Saving precomputed dataset.")
    df = pd.DataFrame(samples, columns=["sentence", "split"])
    df.to_pickle(precomputed_filename)
