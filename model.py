import math
import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    """
    Layer normalization module.

    Args:
        eps (float): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        alpha (nn.Parameter): Learnable parameter for scaling.
        bias (nn.Parameter): Learnable parameter for bias.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Forward pass of the layer normalization module.
    """

    def __init__(self, features: int, eps: float = 1e-6) -> None:
        """
        Initialize the LayerNormalization module.

        Args:
            eps (float): A small value added to the denominator for numerical stability. Default is 1e-6.
        """
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer normalization module.

        Args:
            x (torch.Tensor): Input tensor, (batch, seq_len, hidden_size)

        Returns:
            torch.Tensor: Normalized tensor.
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """
        Feed-forward block module of the Transformer model.

        Args:
            d_model (int): The input and output dimension of the block.
            d_ff (int): The dimension of the intermediate layer.
            dropout (float): The dropout probability.

        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)  # W2 and B2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feed-forward block.

        Args:
            x (torch.Tensor): The input tensor of shape (batch, seq_len, d_model).

        Returns:
            torch.Tensor: The output tensor of shape (batch, seq_len, d_model).

        """
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        Initializes an instance of the InputEmbeddings class.

        Args:
            d_model (int): The dimensionality of the model.
            vocab_size (int): The size of the vocabulary.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the InputEmbeddings module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The embedded input tensor.
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Positional encoding module for Transformer models.

    Args:
        d_model (int): The dimensionality of the input embeddings.
        seq_len (int): The length of the input sequence.
        dropout (float): The dropout probability.

    Attributes:
        d_model (int): The dimensionality of the input embeddings.
        seq_len (int): The length of the input sequence.
        dropout (nn.Dropout): The dropout layer.
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        # apply the cos to odd positions
        pe[:, 1::2] = torch.cos(position * div_term)

        # (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PositionalEncoding module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, d_model) after applying positional encoding.
        """
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class ResidualConnection(nn.Module):
    """
    A residual connection module that applies residual connection to the input tensor.

    Args:
        dropout (float): The dropout probability.

    Attributes:
        dropout (nn.Dropout): The dropout layer.
        norm (LayerNormalization): The layer normalization module.

    Methods:
        forward(x, sublayer): Applies the residual connection to the input tensor.

    """

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        Applies the residual connection to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            sublayer (nn.Module): The sublayer module.

        Returns:
            torch.Tensor: The output tensor after applying the residual connection.

        """
        # paper says add and then normalize, but the most common implementation is to normalize and then add
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-Head Attention Block module.

    Args:
        d_model (int): The input and output dimension of the model.
        h (int): The number of attention heads.
        dropout (float): The dropout probability.

    Attributes:
        d_model (int): The input and output dimension of the model.
        h (int): The number of attention heads.
        d_k (int): The dimension of each attention head.
        w_q (nn.Linear): Linear layer for query projection.
        w_k (nn.Linear): Linear layer for key projection.
        w_v (nn.Linear): Linear layer for value projection.
        w_o (nn.Linear): Linear layer for output projection.
        dropout (nn.Dropout): Dropout layer.

    Methods:
        attention(query, key, value, mask, dropout): Computes the attention scores and performs attention operation.
        forward(q, k, v, mask): Performs forward pass of the multi-head attention block.

    """

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        dropout: nn.Dropout,
    ) -> torch.Tensor:
        """
        Computes the attention scores and performs attention operation.

        Args:
            query (torch.Tensor): The query tensor.
            key (torch.Tensor): The key tensor.
            value (torch.Tensor): The value tensor.
            mask (torch.Tensor): The mask tensor.
            dropout (nn.Dropout): The dropout layer.

        Returns:
            torch.Tensor: The output tensor after attention operation.
            torch.Tensor: The attention scores.

        """
        d_k = query.shape[-1]

        # (batch, h, seq_len, d_k) @ (batch, h, d_k, seq_len) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        # (batch, h, seq_len, seq_len)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_k) --> (batch, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs forward pass of the multi-head attention block.

        Args:
            q (torch.Tensor): The query tensor.
            k (torch.Tensor): The key tensor.
            v (torch.Tensor): The value tensor.
            mask (torch.Tensor): The mask tensor.

        Returns:
            torch.Tensor: The output tensor after the multi-head attention block.

        """
        query = self.w_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)


class EncoderBlock(nn.Module):
    """
    Encoder block of a Transformer model.

    Args:
        self_attention_block (MultiHeadAttentionBlock): The self-attention block.
        feed_forward_blck (FeedForwardBlock): The feed-forward block.
        dropout (float): The dropout rate.

    Attributes:
        self_attention_block (MultiHeadAttentionBlock): The self-attention block.
        feed_forward_block (FeedForwardBlock): The feed-forward block.
        residual_connections (nn.ModuleList): List of residual connections.

    """

    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_blck: FeedForwardBlock,
        features: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_blck
        self.residual_connections = nn.ModuleList(
            ResidualConnection(features, dropout) for _ in range(2)
        )

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the EncoderBlock.

        Args:
            x (torch.Tensor): The input tensor.
            src_mask (torch.Tensor): The mask tensor for the paddings.

        Returns:
            torch.Tensor: The output tensor.

        """
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    """
    Encoder module of the Transformer model.

    Args:
        layers (nn.ModuleList): List of encoder layers.
    """

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Encoded tensor.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    """
    Decoder block of the Transformer model.

    Args:
        self_attention_block (MultiHeadAttentionBlock): The self-attention block.
        cross_attention_block (MultiHeadAttentionBlock): The cross-attention block.
        feed_forward_block (FeedForwardBlock): The feed-forward block.
        dropout (float): The dropout rate.

    Attributes:
        self_attention_block (MultiHeadAttentionBlock): The self-attention block.
        cross_attention_block (MultiHeadAttentionBlock): The cross-attention block.
        feed_forward_block (FeedForwardBlock): The feed-forward block.
        residual_connections (nn.ModuleList): List of residual connections.

    Methods:
        forward(x, encoder_output, src_mask, tgt_mask): Performs forward pass through the decoder block.

    """

    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        features: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            ResidualConnection(features, dropout) for _ in range(3)
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs forward pass through the decoder block.

        Args:
            x (torch.Tensor): The input tensor.
            encoder_output (torch.Tensor): The output from the encoder.
            src_mask (torch.Tensor): The source mask.
            tgt_mask (torch.Tensor): The target mask.

        Returns:
            torch.Tensor: The output tensor.

        """
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        """
        Initializes a Decoder module.

        Args:
            layers (nn.ModuleList): List of decoder layers.

        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs forward pass through the decoder.

        Args:
            x (torch.Tensor): Input tensor.
            encoder_output (torch.Tensor): Output tensor from the encoder.
            src_mask (torch.Tensor): Mask for the source sequence.
            tgt_mask (torch.Tensor): Mask for the target sequence.

        Returns:
            torch.Tensor: Output tensor after passing through the decoder.

        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Encodes the source sequence.

        Args:
            src (torch.Tensor): The source sequence.
            src_mask (torch.Tensor): The mask tensor for the source sequence.

        Returns:
            torch.Tensor: The encoded source sequence.

        """
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(
        self,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decodes the target sequence.

        Args:
            encoder_output (torch.Tensor): The output from the encoder.
            src_mask (torch.Tensor): The mask tensor for the source sequence.
            tgt (torch.Tensor): The target sequence.
            tgt_mask (torch.Tensor): The mask tensor for the target sequence.

        Returns:
            torch.Tensor: The decoded target sequence.

        """
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def projection(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects the output tensor to the vocabulary size.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The projected tensor.

        """
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    # create embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(
            encoder_self_attention_block, feed_forward_block, d_model, dropout
        )
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block,
            d_model,
            dropout,
        )
        decoder_blocks.append(decoder_block)

    # create encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create Transformer
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer
    )

    # initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
