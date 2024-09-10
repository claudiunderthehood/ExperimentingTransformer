import torch
from torch.utils.data import DataLoader
import re
import numpy as np
import random
from modules.transformer_class import *
from typing import List, Tuple, Dict

MAX_SEQ_LENGTH = 50

def set_seed(seed: int) -> None:
    """
    Sets the seed for generating random numbers to ensure reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def simple_tokenizer(text: str) -> List[str]:
    """
    Tokenizes and converts text to lowercase, handling contractions.

    Args:
        text (str): The input text to be tokenized.

    Returns:
        List[str]: A list of tokens from the text.
    """
    text = re.sub(r"(?<=\w)'(?=\w)", " '", text.lower())  # Handle contractions
    tokens: List[str] = re.findall(r'\b\w+\b|[.,!?;]', text)
    return tokens


def build_vocab(data: List[str], min_freq: int = 1) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Builds vocabulary from the tokenized text data.

    Args:
        data (List[str]): List of sentences to build the vocabulary from.
        min_freq (int, optional): Minimum frequency for a word to be included in the vocabulary. Defaults to 1.

    Returns:
        Tuple[Dict[str, int], Dict[int, str]]: Two dictionaries: word-to-index (word2idx) and index-to-word (idx2word).
    """
    vocab: Dict[str, int] = {}
    for sentence in data:
        for word in simple_tokenizer(sentence):
            vocab[word] = vocab.get(word, 0) + 1

    # Filter out words below the minimum frequency
    vocab = {word: freq for word, freq in vocab.items() if freq >= min_freq}
    
    word2idx: Dict[str, int] = {word: idx + 2 for idx, word in enumerate(vocab)}  # Reserve 0 for <PAD> and 1 for <UNK>
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = 1
    idx2word: Dict[int, str] = {idx: word for word, idx in word2idx.items()}
    
    return word2idx, idx2word


def generate_text_with_top_k_sampling(
    model: nn.Module,
    start_sequence: List[str],
    max_len: int,
    word2idx: Dict[str, int],
    idx2word: Dict[int, str],
    device: torch.device,
    temperature: float = 1.0,
    top_k: int = 50
) -> List[str]:
    """
    Generates text using a model with top-k sampling.

    Args:
        model (nn.Module): The trained model used for generating text.
        start_sequence (List[str]): The initial sequence to start the generation.
        max_len (int): The maximum length of the generated sequence.
        word2idx (Dict[str, int]): The word-to-index mapping.
        idx2word (Dict[int, str]): The index-to-word mapping.
        device (torch.device): The device (CPU/GPU) on which the model is running.
        temperature (float, optional): Controls the randomness of predictions by scaling logits. Defaults to 1.0.
        top_k (int, optional): The number of top candidates to sample from. Defaults to 50.

    Returns:
        List[str]: The generated sequence of words.
    """
    model.eval()
    input_seq: torch.Tensor = torch.tensor([word2idx.get(word, word2idx['<UNK>']) for word in start_sequence]).unsqueeze(0).to(device)
    generated_seq: List[str] = start_sequence[:]

    for _ in range(max_len):
        with torch.no_grad():
            output_pred: torch.Tensor = model(input_seq[:, -MAX_SEQ_LENGTH:], input_seq[:, -MAX_SEQ_LENGTH:])
            logits: torch.Tensor = output_pred[:, -1, :] / temperature
            logits = logits.cpu()

            # Apply top-k filtering
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > 0.95
            sorted_logits[sorted_indices_to_remove] = -float('Inf')

            # Sample from the filtered distribution
            probabilities: torch.Tensor = F.softmax(sorted_logits, dim=-1)
            next_word_idx: int = torch.multinomial(probabilities, num_samples=1).item()

            next_word: str = idx2word[next_word_idx]
            if next_word in ['<EOS>', '<PAD>']:
                break
            generated_seq.append(next_word)
            input_seq = torch.cat([input_seq, torch.tensor([[next_word_idx]]).to(device)], dim=1)

    return generated_seq


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Evaluates the model on the test data.

    Args:
        model (nn.Module): The model to evaluate.
        test_loader (DataLoader): DataLoader containing the test dataset.
        criterion (nn.Module): The loss function.
        device (torch.device): The device (CPU/GPU) on which the model is running.

    Returns:
        float: The average test loss.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss: float = 0.0

    with torch.no_grad():
        for input_batch, output_batch in test_loader:
            input_batch, output_batch = input_batch.to(device).long(), output_batch.to(device).long()
            output_pred: torch.Tensor = model(input_batch, output_batch)
            loss: torch.Tensor = criterion(output_pred.view(-1, output_pred.size(-1)), output_batch.view(-1))
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss}")
    return test_loss
