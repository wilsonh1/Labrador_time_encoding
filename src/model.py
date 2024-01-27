import torch
import torch.nn as nn
import torch.nn.functional as F

class ContinuousEmbedding(nn.Module):
    def __init__(self, embedding_dim, pad_token, mask_token, null_token):
        super(ContinuousEmbedding, self).__init__()
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.null_token = null_token
        self.embedding_dim = embedding_dim
        self.special_token_embeddings = nn.Embedding(2, embedding_dim)
        self.dense1 = nn.Linear(1, embedding_dim)  # Assuming single-dimensional continuous input
        self.dense2 = nn.Linear(embedding_dim, embedding_dim)
        self.layernorm = nn.LayerNorm(embedding_dim)

    def forward(self, x_continuous, x_categorical_embeddings):
        # Handle special tokens
        special_embeddings = self.special_token_embeddings(torch.tensor([self.mask_token, self.null_token]))
        
        # Apply linear and non-linear transformations
        x_continuous = self.dense1(x_continuous.unsqueeze(-1))
        x_continuous = F.relu(self.dense2(x_continuous))

        # Combine with categorical embeddings and apply layer normalization
        combined_embeddings = x_continuous + x_categorical_embeddings
        return self.layernorm(combined_embeddings), special_embeddings


class Labrador(nn.Module):
    def __init__(self, output_dim, vocab_size, pad_token, null_token, mask_token, embedding_dim=768, transformer_heads=4, transformer_feedforward_dim=256, transformer_blocks=1, include_head=True):
        super(Labrador, self).__init__()
        self.embedding_dim = embedding_dim
        # Define the embedding layers
        self.categorical_embedding_layer = nn.Embedding(vocab_size + 2, self.embedding_dim)
        self.continuous_embedding_layer, self.special_embeddings = ContinuousEmbedding(self.embedding_dim, pad_token, 
                                                                mask_token, null_token)

        # Define the transformer blocks
        transformer_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim,
                                                        nhead=transformer_heads, 
                                                        dim_feedforward=transformer_feedforward_dim)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=transformer_blocks)

        # Define output heads, if any
        if include_head:
            self.output_head = nn.Linear(self.embedding_dim, output_dim)  # Define output_dim based on the task

    def forward(self, x_categorical, x_continuous):
        categorical_embeddings = self.categorical_embedding_layer(x_categorical)
        continuous_embeddings = self.continuous_embedding_layer(x_continuous, categorical_embeddings)
        
        # Apply transformer blocks
        transformer_output = self.transformer_encoder(continuous_embeddings)
        
        # Apply output head, if included
        if self.output_head is not None:
            return self.output_head(transformer_output)
        return transformer_output
