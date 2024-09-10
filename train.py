import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from modules.tools import set_seed, build_vocab, evaluate_model, generate_text_with_top_k_sampling
from modules.textdataset_class import TextDataset
from modules.transformer_class import Transformer
from typing import List, Tuple, Dict


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for dynamically padding sequences in a batch for DataLoader.

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor]]): A batch of input and target sequences.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Padded input and target sequences as tensors.
    """
    inputs, targets = zip(*batch)
    max_len_inputs: int = max(len(input_seq) for input_seq in inputs)
    max_len_targets: int = max(len(target_seq) for target_seq in targets)
    
    # Padding the input and target sequences
    padded_inputs: List[torch.Tensor] = [
        torch.cat([input_seq, torch.tensor([word2idx['<PAD>']] * (max_len_inputs - len(input_seq)))])
        for input_seq in inputs
    ]
    padded_targets: List[torch.Tensor] = [
        torch.cat([target_seq, torch.tensor([word2idx['<PAD>']] * (max_len_targets - len(target_seq)))])
        for target_seq in targets
    ]
    
    padded_inputs = torch.stack(padded_inputs, dim=0)
    padded_targets = torch.stack(padded_targets, dim=0)
    
    return padded_inputs, padded_targets


# Load dataset
df: pd.DataFrame = pd.read_csv('dataset/converted.csv')
input_texts: List[str] = df['input'].tolist() + df['chosen'].tolist()

# Setup device
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed for reproducibility
set_seed(0)

# Build vocabulary
word2idx: Dict[str, int]
idx2word: Dict[int, str]
word2idx, idx2word = build_vocab(input_texts)
print(f"Vocabulary size: {len(word2idx)}")

# Hyperparameters
SRC_VOCAB_SIZE: int = len(word2idx)
TGT_VOCAB_SIZE: int = len(word2idx)
D_MODEL: int = 500
NUM_HEADS: int = 10
NUM_LAYERS: int = 10
D_FF: int = 1032
MAX_SEQ_LENGTH: int = 50
DROPOUT: float = 0.3  # Increased dropout for regularization

# Initialize the Transformer model, optimizer, and loss function
model: Transformer = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, MAX_SEQ_LENGTH, DROPOUT).to(device)
optimizer: torch.optim.Optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=1e-5)
criterion: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>'], label_smoothing=0.1)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Train, validation, test split ratios
train_split: float = 0.7
val_split: float = 0.15
test_split: float = 0.15

# Create dataset and split into train, validation, and test sets
dataset: TextDataset = TextDataset(df, word2idx, seq_length=MAX_SEQ_LENGTH)
dataset_size: int = len(dataset)
train_size: int = int(train_split * dataset_size)
val_size: int = int(val_split * dataset_size)
test_size: int = dataset_size - train_size - val_size

train_dataset: torch.utils.data.Dataset
val_dataset: torch.utils.data.Dataset
test_dataset: torch.utils.data.Dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# DataLoaders for training, validation, and testing
train_loader: DataLoader = DataLoader(train_dataset, batch_size=50, shuffle=True, pin_memory=True, collate_fn=collate_fn)
val_loader: DataLoader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True, collate_fn=collate_fn)
test_loader: DataLoader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True, collate_fn=collate_fn)

# Learning rate scheduler
scheduler: torch.optim.lr_scheduler.OneCycleLR = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.00001, steps_per_epoch=len(train_loader), epochs=100
)

print("DataLoader created for supervised training.")

# Early stopping parameters
patience: int = 10  # Number of epochs with no improvement to stop training
best_val_loss: float = float('inf')
epochs_without_improvement: int = 0

# Training and validation loop
for epoch in range(100):
    # Training phase
    model.train()
    train_loss: float = 0.0
    for input_batch, output_batch in train_loader:
        input_batch, output_batch = input_batch.to(device).long(), output_batch.to(device).long()
        optimizer.zero_grad()
        output_pred: torch.Tensor = model(input_batch, output_batch)
        loss: torch.Tensor = criterion(output_pred.view(-1, output_pred.size(-1)), output_batch.view(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss: float = 0.0
    with torch.no_grad():
        for input_batch, output_batch in val_loader:
            input_batch, output_batch = input_batch.to(device).long(), output_batch.to(device).long()
            output_pred = model(input_batch, output_batch)
            loss = criterion(output_pred.view(-1, output_pred.size(-1)), output_batch.view(-1))
            val_loss += loss.item()
    val_loss /= len(val_loader)
    
    print(f"Epoch {epoch+1}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        # Optionally, save the best model state
        # torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Update learning rate using the scheduler
    scheduler.step(val_loss)

print("Training with validation completed.")

# Evaluate the model on the test set
test_loss: float = evaluate_model(model, test_loader, criterion, device)

# Generate text using the trained model
start_sequence: List[str] = ["Example", "sequence", "here"]
max_len: int = 20
generated_text: List[str] = generate_text_with_top_k_sampling(
    model, start_sequence, max_len, word2idx, idx2word, device, temperature=1.0
)

print("Generated text:", " ".join(generated_text))
