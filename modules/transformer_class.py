import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        """
        Multi-head attention mechanism.

        Args:
            d_model (int): The dimension of the model.
            num_heads (int): The number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.d_k: int = d_model // num_heads
        
        # Linear transformations for query, key, value, and output
        self.W_q: nn.Linear = nn.Linear(d_model, d_model)
        self.W_k: nn.Linear = nn.Linear(d_model, d_model)
        self.W_v: nn.Linear = nn.Linear(d_model, d_model)
        self.W_o: nn.Linear = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes scaled dot-product attention.

        Args:
            Q (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_length, d_k).
            K (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_length, d_k).
            V (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_length, d_k).
            mask (Optional[torch.Tensor]): Optional attention mask of shape (batch_size, 1, seq_length, seq_length).

        Returns:
            torch.Tensor: Attention output of shape (batch_size, num_heads, seq_length, d_k).
        """
        attn_scores: torch.Tensor = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs: torch.Tensor = torch.softmax(attn_scores, dim=-1)
        output: torch.Tensor = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Splits the input tensor into multiple heads for multi-head attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).

        Returns:
            torch.Tensor: Tensor with multiple heads of shape (batch_size, num_heads, seq_length, d_k).
        """
        batch_size: int = x.size(0)
        seq_length: int = x.size(1)
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combines the multi-head attention results into a single tensor.

        Args:
            x (torch.Tensor): Tensor with multi-head attention of shape (batch_size, num_heads, seq_length, d_k).

        Returns:
            torch.Tensor: Combined tensor of shape (batch_size, seq_length, d_model).
        """
        batch_size: int = x.size(0)
        seq_length: int = x.size(2)
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the multi-head attention mechanism.

        Args:
            Q (torch.Tensor): Query tensor of shape (batch_size, seq_length, d_model).
            K (torch.Tensor): Key tensor of shape (batch_size, seq_length, d_model).
            V (torch.Tensor): Value tensor of shape (batch_size, seq_length, d_model).
            mask (Optional[torch.Tensor]): Optional attention mask of shape (batch_size, 1, seq_length, seq_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, d_model).
        """
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output: torch.Tensor = self.scaled_dot_product_attention(Q, K, V, mask)
        output: torch.Tensor = self.W_o(self.combine_heads(attn_output))
        return output
    

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        """
        Position-wise feed-forward network.

        Args:
            d_model (int): The dimension of the model.
            d_ff (int): The dimension of the feed-forward layer.
        """
        super(PositionWiseFeedForward, self).__init__()
        self.fc1: nn.Linear = nn.Linear(d_model, d_ff)
        self.fc2: nn.Linear = nn.Linear(d_ff, d_model)
        self.relu: nn.ReLU = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, d_model).
        """
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int) -> None:
        """
        Positional encoding layer that adds positional information to the input embeddings.

        Args:
            d_model (int): The dimension of the model.
            max_seq_length (int): Maximum length of the input sequence.
        """
        super(PositionalEncoding, self).__init__()
        
        pe: torch.Tensor = torch.zeros(max_seq_length, d_model)
        position: torch.Tensor = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term: torch.Tensor = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of positional encoding.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).

        Returns:
            torch.Tensor: Tensor with positional encodings added to input.
        """
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        """
        Encoder layer consisting of multi-head attention and position-wise feed-forward network.

        Args:
            d_model (int): The dimension of the model.
            num_heads (int): Number of attention heads.
            d_ff (int): The dimension of the feed-forward network.
            dropout (float): Dropout rate.
        """
        super(EncoderLayer, self).__init__()
        self.self_attn: MultiHeadAttention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward: PositionWiseFeedForward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1: nn.LayerNorm = nn.LayerNorm(d_model)
        self.norm2: nn.LayerNorm = nn.LayerNorm(d_model)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the encoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).
            mask (Optional[torch.Tensor]): Optional attention mask of shape (batch_size, 1, seq_length, seq_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, d_model).
        """
        attn_output: torch.Tensor = self.self_attn(x, x, x, mask)
        x: torch.Tensor = self.norm1(x + self.dropout(attn_output))
        ff_output: torch.Tensor = self.feed_forward(x)
        return self.norm2(x + self.dropout(ff_output))


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        """
        Decoder layer consisting of self-attention, cross-attention, and position-wise feed-forward network.

        Args:
            d_model (int): The dimension of the model.
            num_heads (int): Number of attention heads.
            d_ff (int): The dimension of the feed-forward network.
            dropout (float): Dropout rate.
        """
        super(DecoderLayer, self).__init__()
        self.self_attn: MultiHeadAttention = MultiHeadAttention(d_model, num_heads)
        self.cross_attn: MultiHeadAttention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward: PositionWiseFeedForward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1: nn.LayerNorm = nn.LayerNorm(d_model)
        self.norm2: nn.LayerNorm = nn.LayerNorm(d_model)
        self.norm3: nn.LayerNorm = nn.LayerNorm(d_model)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        tgt_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass of the decoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).
            enc_output (torch.Tensor): Encoder output of shape (batch_size, seq_length, d_model).
            src_mask (Optional[torch.Tensor]): Optional source mask of shape (batch_size, 1, seq_length, seq_length).
            tgt_mask (Optional[torch.Tensor]): Optional target mask of shape (batch_size, 1, seq_length, seq_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, d_model).
        """
        attn_output: torch.Tensor = self.self_attn(x, x, x, tgt_mask)
        x: torch.Tensor = self.norm1(x + self.dropout(attn_output))

        cross_attn_output: torch.Tensor = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        ff_output: torch.Tensor = self.feed_forward(x)
        return self.norm3(x + self.dropout(ff_output))


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_length: int,
        dropout: float
    ) -> None:
        """
        Transformer model consisting of an encoder and a decoder.

        Args:
            src_vocab_size (int): Vocabulary size of the source language.
            tgt_vocab_size (int): Vocabulary size of the target language.
            d_model (int): The dimension of the model.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of encoder/decoder layers.
            d_ff (int): The dimension of the feed-forward network.
            max_seq_length (int): Maximum sequence length.
            dropout (float): Dropout rate.
        """
        super(Transformer, self).__init__()

        self.encoder_embedding: nn.Embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding: nn.Embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding: PositionalEncoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers: nn.ModuleList = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers: nn.ModuleList = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.fc: nn.Linear = nn.Linear(d_model, tgt_vocab_size)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def generate_mask(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate source and target masks for masking padding and ensuring causality in the target sequence.

        Args:
            src (torch.Tensor): Source tensor of shape (batch_size, seq_length).
            tgt (torch.Tensor): Target tensor of shape (batch_size, seq_length).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Source and target masks.
        """
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        src_mask: torch.Tensor = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
        tgt_mask: torch.Tensor = (tgt != 0).unsqueeze(1).unsqueeze(3).to(device)

        seq_length: int = tgt.size(1)
        nopeak_mask: torch.Tensor = torch.triu(torch.ones((1, seq_length, seq_length), device=device), diagonal=1).bool()

        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor, src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the transformer model.

        Args:
            src (torch.Tensor): Source tensor of shape (batch_size, seq_length).
            tgt (torch.Tensor): Target tensor of shape (batch_size, seq_length).
            src_mask (Optional[torch.Tensor]): Source mask of shape (batch_size, 1, seq_length, seq_length).
            tgt_mask (Optional[torch.Tensor]): Target mask of shape (batch_size, 1, seq_length, seq_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, vocab_size).
        """
        src = self.encoder_embedding(src.long())
        src = self.positional_encoding(src)
        src = self.dropout(src)

        tgt = self.decoder_embedding(tgt.long())
        tgt = self.positional_encoding(tgt)
        tgt = self.dropout(tgt)

        for layer in self.encoder_layers:
            src = layer(src, src_mask)

        for layer in self.decoder_layers:
            tgt = layer(tgt, src, src_mask, tgt_mask)

        output: torch.Tensor = self.fc(tgt)
        return output

    def generate(
        self, src: torch.Tensor, max_len: int = 50, start_token_idx: int = 1, end_token_idx: int = 2, beam_size: int = 3
    ) -> List[int]:
        """
        Generate a sequence using beam search.

        Args:
            src (torch.Tensor): Source tensor of shape (1, seq_length).
            max_len (int): Maximum length of the generated sequence.
            start_token_idx (int): Index of the start token.
            end_token_idx (int): Index of the end token.
            beam_size (int): Beam size for beam search.

        Returns:
            List[int]: Generated sequence of token indices.
        """
        self.eval()

        src_mask: torch.Tensor = (src != 0).unsqueeze(1).unsqueeze(2).to(src.device)
        enc_output: torch.Tensor = self.encoder_embedding(src)
        enc_output = self.positional_encoding(enc_output)

        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        generated_seqs: torch.Tensor = torch.full((beam_size, 1), start_token_idx, dtype=torch.long, device=src.device)
        seq_scores: torch.Tensor = torch.zeros(beam_size, 1, device=src.device)
        complete_seqs: List[torch.Tensor] = []
        complete_seqs_scores: List[float] = []

        for _ in range(max_len):
            dec_input: torch.Tensor = self.decoder_embedding(generated_seqs)
            dec_input = self.positional_encoding(dec_input)

            tgt_mask: torch.Tensor = torch.triu(torch.ones((dec_input.size(1), dec_input.size(1)), device=src.device), diagonal=1).bool()

            dec_output: torch.Tensor = dec_input
            for dec_layer in self.decoder_layers:
                dec_output = dec_layer(
                    dec_output, enc_output.repeat(beam_size, 1, 1), src_mask.repeat(beam_size, 1, 1, 1), tgt_mask
                )

            output_logits: torch.Tensor = self.fc(dec_output[:, -1, :])
            log_probs: torch.Tensor = F.log_softmax(output_logits, dim=-1)

            total_scores: torch.Tensor = seq_scores + log_probs
            top_scores, top_indices = total_scores.view(-1).topk(beam_size, dim=-1)

            next_tokens: torch.Tensor = top_indices % log_probs.size(-1)
            beam_indices: torch.Tensor = top_indices // log_probs.size(-1)
            generated_seqs = torch.cat([generated_seqs[beam_indices], next_tokens.unsqueeze(1)], dim=-1)
            seq_scores = top_scores.unsqueeze(1)

            for i in range(beam_size):
                if next_tokens[i].item() == end_token_idx:
                    complete_seqs.append(generated_seqs[i].cpu())
                    complete_seqs_scores.append(seq_scores[i].item())
                    seq_scores[i] = -float('inf')

            if len(complete_seqs) == beam_size:
                break

        # Return the sequence with the best score
        if len(complete_seqs) > 0:
            best_seq_idx: int = complete_seqs_scores.index(max(complete_seqs_scores))
            return complete_seqs[best_seq_idx].tolist()
        else:
            return generated_seqs[0].cpu().tolist()
