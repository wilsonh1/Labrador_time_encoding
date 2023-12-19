# Bert model architecture to process tuples (labid, value)

import torch
import torch.nn as nn

# Constants
vocab_size = 10000 # Number of lab values
max_seq_length = 128 # Sequence length
hidden_size = 256 # Hidden size
num_attention_heads = 4 # Number of attention heads in the transformer
num_transformer_layers = 1 # Number of transformer layers

# Input tokens (random integers)
lab_ids = torch.randint(0, vocab_size, (1, max_seq_length))
# Values are continuous number (Noramlized between 0 and 1)
lab_values = torch.rand((1, max_seq_length))

# Create the dictionary of lab values
lab_dict = {}
for i in range(lab_ids.shape[1]):
    lab_dict[lab_ids[0, i].item()] = lab_values[0, i].item()

# Token embeddings
embedding_layer = nn.Embedding(vocab_size, hidden_size)
token_embeddings = embedding_layer(input_tokens)

# Positional encodings
position_encodings = torch.arange(0, max_seq_length).unsqueeze(0)
position_encodings = embedding_layer(position_encodings)

# Transformer encoder layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_attention_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )
    
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        feed_forward_output = self.feed_forward(x)
        x = x + feed_forward_output
        return x

# BERT-like model
class BERT(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_attention_heads, num_transformer_layers):
        super(BERT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer_layers = nn.ModuleList([TransformerEncoderLayer(hidden_size, num_attention_heads) for _ in range(num_transformer_layers)])
    
    def forward(self, input_tokens):
        embeddings = self.embedding(input_tokens) + position_encodings
        for layer in self.transformer_layers:
            embeddings = layer(embeddings)
        return embeddings

# Instantiate the BERT model
bert_model = BERT(vocab_size, hidden_size, num_attention_heads, num_transformer_layers)

# Forward pass
output = bert_model(input_tokens)
print(output.shape)  # Should print torch.Size([1, max_seq_length, hidden_size])
