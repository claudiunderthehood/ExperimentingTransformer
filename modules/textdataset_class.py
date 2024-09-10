import re
import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Tuple, List, Dict

class TextDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, word2idx: Dict[str, int], seq_length: int = 16) -> None:
        """
        Custom dataset for loading and processing text data for sequence-to-sequence models.

        Args:
            dataframe (pd.DataFrame): The dataframe containing the text data.
            word2idx (Dict[str, int]): A dictionary mapping words to their corresponding indices.
            seq_length (int, optional): The maximum sequence length. Defaults to 16.
        """
        self.dataframe: pd.DataFrame = dataframe
        self.word2idx: Dict[str, int] = word2idx
        self.seq_length: int = seq_length

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.dataframe)

    def tokenize_text(self, text: str) -> List[int]:
        """
        Tokenizes a text into a list of word indices based on the word2idx dictionary.

        Args:
            text (str): The input text to be tokenized.

        Returns:
            List[int]: A list of token indices corresponding to the words in the text, truncated to the sequence length.
        """
        tokens: List[int] = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in re.findall(r'\w+', text.lower())]
        return tokens[:self.seq_length]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): The index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the input sequence and target sequence as tensors.
        """
        input_text: str = self.dataframe.iloc[idx]['input']
        target_text: str = self.dataframe.iloc[idx]['chosen']

        input_seq: torch.Tensor = torch.tensor(self.tokenize_text(input_text), dtype=torch.long)
        target_seq: torch.Tensor = torch.tensor(self.tokenize_text(target_text), dtype=torch.long)

        return input_seq, target_seq

