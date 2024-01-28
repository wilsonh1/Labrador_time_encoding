import torch
import torch.nn as nn
import torch.nn.functional as F

class ContinuousEmbedding(nn.Module):
    def __init__(self, embedding_dim, pad_token, mask_token, null_token, mask_padding=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.null_token = null_token
        self.mask_padding = mask_padding

        # Special token embeddings
        self.special_token_embeddings = nn.Embedding(3, self.embedding_dim)

        # Linear projection for each lab value
        self.dense1 = nn.Linear(1, self.embedding_dim) # 1 for each lab value

        # Non-linear MLP
        self.dense2 = nn.Linear(self.embedding_dim, self.embedding_dim)

        # Layer normalization
        self.layernorm = nn.LayerNorm(self.embedding_dim)

    def forward(self, x_continuous, x_categorical_embeddings):
        # Expand dimensions if necessary
        if x_continuous.dim() == 1:
            x_continuous = x_continuous.unsqueeze(0)

        # Compute masks for special tokens
        mask_tensor = torch.full_like(x_continuous, self.mask_token, dtype=torch.float32)
        null_tensor = torch.full_like(x_continuous, self.null_token, dtype=torch.float32)
        padding_tensor = torch.full_like(x_continuous, self.pad_token, dtype=torch.float32)
        
        boolean_for_mask = (x_continuous == mask_tensor).float()
        boolean_for_null = (x_continuous == null_tensor).float()
        boolean_for_padding = (x_continuous == padding_tensor).float()

        # Project each lab value to embedding_dim vector
        x = self.dense1(x_continuous.unsqueeze(-1))

        # Get special token embeddings
        mask_embedding = self.special_token_embeddings(torch.tensor([0], device=x.device))
        null_embedding = self.special_token_embeddings(torch.tensor([1], device=x.device))
        padding_embedding = self.special_token_embeddings(torch.tensor([2], device=x.device))
        

        # Apply masks to special token embeddings
        mask_tensor = mask_embedding * boolean_for_mask.unsqueeze(-1)
        null_tensor = null_embedding * boolean_for_null.unsqueeze(-1)
        padding_tensor = padding_embedding * boolean_for_padding.unsqueeze(-1)

        # Combine tensors
        x = x * (1 - boolean_for_mask - boolean_for_null - boolean_for_padding).unsqueeze(-1) + mask_tensor + null_tensor + padding_tensor
        x = x + x_categorical_embeddings
        x = F.relu(self.dense2(x))
        x = self.layernorm(x)

        return x

class MLMPredictionHead(nn.Module):
    def __init__(self, vocab_size, embedding_dim, continuous_head_activation):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.dense_categorical = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.categorical_head = nn.Linear(self.embedding_dim, self.vocab_size)

        self.dense_continuous = nn.Linear(self.embedding_dim + self.vocab_size, self.embedding_dim + self.vocab_size)
        self.continuous_head = nn.Linear(self.embedding_dim + self.vocab_size, 1)

        # Activation function for continuous head
        if continuous_head_activation == 'relu':
            self.continuous_head_activation = nn.ReLU()
        elif continuous_head_activation == 'sigmoid':
            self.continuous_head_activation = nn.Sigmoid()
        elif continuous_head_activation == 'tanh':
            self.continuous_head_activation = nn.Tanh()
        elif continuous_head_activation == 'linear':
            self.continuous_head_activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported continuous head activation: {continuous_head_activation}")

    def forward(self, inputs):
        cat = F.relu(self.dense_categorical(inputs))
        categorical_logits = self.categorical_head(cat)
        categorical_prediction = F.softmax(categorical_logits, dim=-1)

        augmented_inputs = torch.cat([inputs, categorical_prediction], dim=-1)
        cont = F.relu(self.dense_continuous(augmented_inputs))
        continuous_prediction = self.continuous_head(cont)
        continuous_prediction = self.continuous_head_activation(continuous_prediction)

        return {
            "categorical_output": categorical_prediction,
            "continuous_output": continuous_prediction
        }


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, activation, feedforward_dim, dropout_rate, first_block=False):
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            activation,
            nn.Linear(feedforward_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, attn_mask=None):
        attn_output, _ = self.att(x, x, x, key_padding_mask=attn_mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class Labrador(nn.Module):
    def __init__(self, mask_token, null_token, vocab_size, embedding_dim, transformer_heads, num_blocks, transformer_feedforward_dim, include_head, continuous_head_activation='relu', dropout_rate=0.2, pad_token=102):
        super().__init__()
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.null_token = null_token
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.transformer_heads = transformer_heads
        self.num_blocks = num_blocks
        self.transformer_feedforward_dim = transformer_feedforward_dim
        self.include_head = include_head
        self.continuous_head_activation = continuous_head_activation
        self.dropout_rate = dropout_rate

        self.categorical_embedding_layer = nn.Embedding(num_embeddings=vocab_size + 3, embedding_dim=embedding_dim) # Add 3 for special tokens
        self.continuous_embedding_layer = ContinuousEmbedding(embedding_dim, pad_token, mask_token, null_token)
        self.projection_layer = nn.Linear(embedding_dim * 2, embedding_dim)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embedding_dim, num_heads=transformer_heads, activation=nn.ReLU(), feedforward_dim=transformer_feedforward_dim, dropout_rate=dropout_rate)
            for _ in range(num_blocks)
        ])

        self.head = MLMPredictionHead(vocab_size, embedding_dim, continuous_head_activation)

    def forward(self, categorical_input, continuous_input, attn_mask=None):
        x_categorical_embedding = self.categorical_embedding_layer(categorical_input)
        x_continuous = self.continuous_embedding_layer(continuous_input, x_categorical_embedding)
        x = torch.cat([x_categorical_embedding, x_continuous], dim=-1)
        x = self.projection_layer(x)


        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        if self.include_head:
            x = self.head(x)

        return x



"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContinuousEmbedding(nn.Module):
    def __init__(self, embedding_dim, pad_token, mask_token, null_token, mask_padding=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.null_token = null_token
        self.mask_padding = mask_padding

        # Special token embeddings
        self.special_token_embeddings = nn.Embedding(2, self.embedding_dim)

        # Linear projection for each lab value
        self.dense1 = nn.Linear(1, self.embedding_dim) # 1 for each lab value

        # Non-linear MLP
        self.dense2 = nn.Linear(self.embedding_dim, self.embedding_dim)

        # Layer normalization
        self.layernorm = nn.LayerNorm(self.embedding_dim)

    def forward(self, x_continuous, x_categorical_embeddings):
        # Expand dimensions if necessary
        if x_continuous.dim() == 1:
            x_continuous = x_continuous.unsqueeze(0)

        # Compute masks for special tokens
        mask_tensor = torch.full_like(x_continuous, self.mask_token, dtype=torch.float32)
        null_tensor = torch.full_like(x_continuous, self.null_token, dtype=torch.float32)
        boolean_for_mask = (x_continuous == mask_tensor).float()
        boolean_for_null = (x_continuous == null_tensor).float()

        # Project each lab value to embedding_dim vector
        x = self.dense1(x_continuous.unsqueeze(-1))

        # Get special token embeddings
        mask_embedding = self.special_token_embeddings(torch.tensor([0], device=x.device))
        null_embedding = self.special_token_embeddings(torch.tensor([1], device=x.device))

        # Apply masks to special token embeddings
        mask_tensor = mask_embedding * boolean_for_mask.unsqueeze(-1)
        null_tensor = null_embedding * boolean_for_null.unsqueeze(-1)

        # Combine tensors
        x = x * (1 - boolean_for_mask - boolean_for_null).unsqueeze(-1) + mask_tensor + null_tensor
        x = x + x_categorical_embeddings
        x = F.relu(self.dense2(x))
        x = self.layernorm(x)

        return x

class MLMPredictionHead(nn.Module):
    def __init__(self, vocab_size, embedding_dim, continuous_head_activation):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.dense_categorical = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.categorical_head = nn.Linear(self.embedding_dim, self.vocab_size)

        self.dense_continuous = nn.Linear(self.embedding_dim + self.vocab_size, self.embedding_dim + self.vocab_size)
        self.continuous_head = nn.Linear(self.embedding_dim + self.vocab_size, 1)

        # Activation function for continuous head
        if continuous_head_activation == 'relu':
            self.continuous_head_activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported continuous head activation: {continuous_head_activation}")

    def forward(self, inputs):
        cat = F.relu(self.dense_categorical(inputs))
        categorical_logits = self.categorical_head(cat)
        categorical_prediction = F.softmax(categorical_logits, dim=-1)

        augmented_inputs = torch.cat([inputs, categorical_prediction], dim=-1)
        cont = F.relu(self.dense_continuous(augmented_inputs))
        continuous_prediction = self.continuous_head(cont)
        continuous_prediction = self.continuous_head_activation(continuous_prediction)

        return {
            "categorical_output": categorical_prediction,
            "continuous_output": continuous_prediction
        }


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, activation, feedforward_dim, dropout_rate, first_block=False):
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            activation,
            nn.Linear(feedforward_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        attn_output, _ = self.att(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class Labrador(nn.Module):
    def __init__(self, mask_token, null_token, vocab_size, embedding_dim, transformer_heads, num_blocks, transformer_feedforward_dim, include_head, continuous_head_activation='relu', dropout_rate=0.2, pad_token=102):
        super().__init__()
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.null_token = null_token
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.transformer_heads = transformer_heads
        self.num_blocks = num_blocks
        self.transformer_feedforward_dim = transformer_feedforward_dim
        self.include_head = include_head
        self.continuous_head_activation = continuous_head_activation
        self.dropout_rate = dropout_rate

        self.categorical_embedding_layer = nn.Embedding(num_embeddings=vocab_size + 2, embedding_dim=embedding_dim)
        self.continuous_embedding_layer = ContinuousEmbedding(embedding_dim, pad_token, mask_token, null_token)
        self.projection_layer = nn.Linear(embedding_dim * 2, embedding_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embedding_dim, num_heads=transformer_heads, activation=nn.ReLU(), feedforward_dim=transformer_feedforward_dim, dropout_rate=dropout_rate)
            for _ in range(num_blocks)
        ])

        self.head = MLMPredictionHead(vocab_size, embedding_dim, continuous_head_activation)

    def forward(self, categorical_input, continuous_input):
        x_categorical_embedding = self.categorical_embedding_layer(categorical_input)
        x_continuous = self.continuous_embedding_layer(continuous_input, x_categorical_embedding)
        x = torch.cat([x_categorical_embedding, x_continuous], dim=-1)
        x = self.projection_layer(x)

        for block in self.blocks:
            x = block(x)

        if self.include_head:
            x = self.head(x)

        return x
"""