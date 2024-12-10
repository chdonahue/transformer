# From Sebastian Raschka implementation
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class TranslationDataset(Dataset):
    """
    This creates a dataset for translation tasks where we have paired inputs and outputs
    It will include padding
    """

    def __init__(self, source_texts, target_texts, tokenizer, max_length, pad_token_id):
        self.input_ids = []
        self.target_ids = []
        self.pad_token_id = pad_token_id

        for src_txt, tgt_txt in zip(source_texts, target_texts):
            # Tokenize both source and target
            src_tokens = tokenizer.encode(src_txt)
            tgt_tokens = tokenizer.encode(tgt_txt)

            # Truncate if longer than max_length
            src_tokens = src_tokens[:max_length]
            tgt_tokens = tgt_tokens[:max_length]

            # Pad if shorter than max_length
            src_tokens = self._pad_sequence(src_tokens, max_length)
            tgt_tokens = self._pad_sequence(tgt_tokens, max_length)

            self.input_ids.append(torch.tensor(src_tokens))
            self.target_ids.append(torch.tensor(tgt_tokens))

    def _pad_sequence(self, seq, max_len):
        if len(seq) < max_len:
            return seq + [self.pad_token_id] * (max_len - len(seq))
        return seq

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_translation_dataloader(source_texts, target_texts, tokenizer, cfg):
    """
    Returns dataloader for translation task
    """
    dataset = TranslationDataset(
        source_texts,
        target_texts,
        tokenizer,
        max_length=cfg["context_length"],
        pad_token_id=cfg["pad_token_id"],
    )

    return DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=True
    )


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


class CrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        print("Warning: Cross attention is not yet implemented!")

    def forward(self, x, encoder_output):
        return x


class MultiHeadAttention(nn.Module):
    # TODO: Add causal and padding mask functionality
    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        dropout,
        num_heads,
        qkv_bias=False,
        causal_mask=False,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = (
            d_out // num_heads
        )  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.causal_mask = causal_mask
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x, key_padding_mask=None):
        b, num_tokens, d_in = x.shape  # Batch, tokens, d_in

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        if self.causal_mask:
            # Original mask truncated to the number of tokens and converted to boolean
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

            # Use the mask to fill attention scores
            attn_scores.masked_fill_(mask_bool, -torch.inf)
        if key_padding_mask is not None:
            # key_padding_mask shape: (batch, seq_len)
            # Reshape to (Batch, 1, 1, seq_len)
            padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores.masked_fill_(padding_mask, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # TODO: Can simplify this by passing only cfg:
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class EncoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # TODO: Can simplify this by passing only cfg:
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            causal_mask=False,
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, padding_mask=None):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, padding_mask)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class DecoderBlock(nn.Module):
    """
    Decoder block for transformer. Will need masked multi-head attention!
    """

    def __init__(self, cfg):
        super().__init__()
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.norm3 = LayerNorm(cfg["emb_dim"])
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            causal_mask=True,
        )
        self.cross_attention = CrossAttention()  # TODO: INSERT THIS LATER!
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(
        self,
        encoder_output,
        output_embedding,
        encoder_padding_mask=None,
        decoder_padding_mask=None,
    ):
        # First shortcut:
        shortcut = output_embedding
        x = self.norm1(output_embedding)
        x = self.att(x, decoder_padding_mask)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Second shortcut:
        shortcut = x  # first stage output
        x = self.norm2(x)
        x = self.cross_attention(x, encoder_output, encoder_padding_mask)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Third shortcut:
        shortcut = x  # 2nd stage output
        x = self.norm3(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class Transformer(nn.Module):
    """
    Encoder and Decoder for Transformer model (based on original paper, but with pre-LayerNorm):
    """

    def __init__(self, cfg):
        super().__init__()

        # Input and Output Embeddings:
        self.input_tok_emb = nn.Embedding(cfg["input_vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.output_tok_emb = nn.Embedding(cfg["output_vocab_size"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(cfg) for _ in range(cfg["num_layers"])]
        )
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(cfg) for _ in range(cfg["num_layers"])]
        )  # to deal with 2 args
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["output_vocab_size"], bias=False)
        self.pad_token_id = cfg["pad_token_id"]

    def _make_padding_mask(self, input):
        """Create padding mask from input tensor"""
        # Assuming padding token is 0, create mask where 1 indicates padding
        return (input == self.pad_token_id).to(input.device)

    def forward(self, inputs, outputs):
        batch_size, input_seq_len = inputs.shape  # removing batch dim

        # Create padding masks:
        encoder_padding_mask = self._make_padding_mask(
            inputs
        )  # (batch_size, input_seq_len)
        decoder_padding_mask = self._make_padding_mask(
            outputs
        )  # (batch_size, output_seq_len)

        # Pass through Encoder:
        input_tok_emb = self.input_tok_emb(inputs)
        input_pos_emb = self.pos_emb(torch.arange(input_seq_len, device=inputs.device))
        x = input_tok_emb + input_pos_emb
        x = self.drop_emb(x)
        for block in self.encoder_blocks:
            x = block(x, encoder_padding_mask)

        # Pass through Decoder:
        _, output_seq_len = outputs.shape  # removing batch dim
        output_tok_emb = self.output_tok_emb(outputs)
        output_pos_emb = self.pos_emb(
            torch.arange(output_seq_len, device=outputs.device)
        )
        y = output_tok_emb + output_pos_emb
        y = self.drop_emb(y)
        # forward pass through decoder blocks:
        for block in self.decoder_blocks:
            y = block(
                encoder_output=x,
                output_embedding=y,
                encoder_padding_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
            )
        logits = self.out_head(y)
        return logits
