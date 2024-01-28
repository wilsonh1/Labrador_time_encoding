from transformers import BertTokenizer
import numpy as np
import torch

class CustomBertTokenizer(BertTokenizer):
    """
    Customized BERT tokenizer class with additional functionalities for handling special tokens.

    Parameters:
    -----------
    vocab_file_path : str
        The path to the vocabulary file.

    special_tokens : dict, optional
        A dictionary containing special tokens and their positions in the vocabulary. Default is None.

    **kwargs : additional keyword arguments
        Additional keyword arguments to be passed to the base class (BertTokenizer).

    Methods:
    --------
    insert_special_tokens(input_list, special_tokens):
        Inserts special tokens into a list at specified positions.

    create_vocab_file(vocab_list, vocab_file_path):
        Creates a vocabulary file from a list of tokens.

    create_bert_tokenizer(vocab_list, special_tokens, vocab_file_path='vocab.txt'):
        Creates a customized BERT tokenizer with modified vocabulary.

    Example Usage:
    --------------
    # Instantiate the tokenizer with custom special tokens
    tokenizer = CustomBertTokenizer(vocab_file_path='custom_vocab.txt', special_tokens={'[MASK]': 0})

    # Access additional methods
    modified_list = CustomBertTokenizer.insert_special_tokens(['apple', 'banana'], {'[MASK]': 'orange'})
    CustomBertTokenizer.create_vocab_file(['apple', 'banana', '[MASK]'], 'custom_vocab.txt')
    custom_tokenizer = CustomBertTokenizer.create_bert_tokenizer(['apple', 'banana'], {'[MASK]': 'orange'})
    """
    def __init__(self, vocab_file_path, special_tokens=None, **kwargs):
        """
        Initializes the CustomBertTokenizer.

        Parameters:
        -----------
        vocab_file_path : str
            The path to the vocabulary file.

        special_tokens : dict, optional
            A dictionary containing special tokens and their positions in the vocabulary. Default is None.

        **kwargs : additional keyword arguments
            Additional keyword arguments to be passed to the base class (BertTokenizer).
        """
        super().__init__(vocab_file_path, do_lower_case=False, **kwargs)
        if special_tokens:
            self.add_special_tokens(special_tokens)

    @staticmethod
    def insert_special_tokens(input_list, special_tokens):
        """
        Inserts special tokens into a list at specified positions.

        Parameters:
        -----------
        input_list : list
            The input list of tokens.

        special_tokens : dict
            A dictionary containing special tokens and their positions.

        Returns:
        --------
        list
            The modified list with inserted special tokens.
        """
        result_list = input_list.copy()
        for position, token in special_tokens.items():
            result_list.insert(position, token)
        return result_list

    @staticmethod
    def create_vocab_file(vocab_list, vocab_file_path):
        """
        Creates a vocabulary file from a list of tokens.

        Parameters:
        -----------
        vocab_list : list
            The list of tokens to be included in the vocabulary.

        vocab_file_path : str
            The path to the vocabulary file.
        """
        with open(vocab_file_path, 'w', encoding='utf-8') as vocab_file:
            for token in vocab_list:
                vocab_file.write(token + '\n')

    @classmethod
    def create_bert_tokenizer(cls, vocab_list, special_tokens, vocab_file_path='vocab.txt'):
        """
        Creates a customized BERT tokenizer with modified vocabulary.

        Parameters:
        -----------
        vocab_list : list
            The list of tokens for the modified vocabulary.

        special_tokens : dict
            A dictionary containing special tokens and their positions.

        vocab_file_path : str, optional
            The path to the vocabulary file. Default is 'vocab.txt'.

        Returns:
        --------
        CustomBertTokenizer
            The customized BERT tokenizer instance.
        """
        modified_vocab = cls.insert_special_tokens(vocab_list, special_tokens)
        cls.create_vocab_file(modified_vocab, vocab_file_path)
        return cls(vocab_file_path)


class LabradorTokenizer:
    """
    A tokenizer class for the Labrador model, handling both categorical and continuous data.
    
    This tokenizer supports tokenization of individual sequences as well as batches of sequences,
    incorporating special tokens for mask, null, and padding functionalities. It is designed to
    prepare data for input into a model that requires both categorical and continuous inputs, along
    with attention mechanisms that might necessitate masking and padding.

    Attributes:
        mask_token (int): The token ID used for masking.
        null_token (int): The token ID used for null values.
        pad_token (int): The token ID used for padding.
        special_tokens (list of str): A list containing the string representations of special tokens.
        vocab (dict): A mapping from token strings to their corresponding IDs.
        inverse_vocab (dict): A reverse mapping from token IDs back to their string representations.
    """
    def __init__(self):
        """
        Initializes the tokenizer with predefined special tokens and their IDs.
        """
        self.mask_token = -1
        self.null_token = -2
        self.pad_token = -2
        self.special_tokens = ["[MASK]", "[NULL]", "[PAD]"]
        self.vocab = {"[MASK]": self.mask_token, "[NULL]": self.null_token, "[PAD]": self.pad_token}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def train(self, categorical_data):
        """
        Expands the vocabulary by analyzing a dataset of categorical data.

        Args:
            categorical_data (list of str): A list of categorical data items to be tokenized.
        """
        count = 0
        for item in categorical_data:
            if item not in self.vocab:
                self.vocab[item] = count
                self.inverse_vocab[self.vocab[item]] = item
                count += 1
                
        # update the especial tokens positions to be the in the end of the vocab
        self.mask_token = count
        self.null_token = count + 1
        self.pad_token = count + 2

        self.vocab["[MASK]"] = self.mask_token
        self.vocab["[NULL]"] = self.null_token
        self.vocab["[PAD]"] = self.pad_token

    def tokenize(self, categorical_data, continuous_data, max_length, return_tensors=None):
        """
        Tokenizes and encodes a single sequence of categorical and continuous data, applying padding as necessary.

        Args:
            categorical_data (list of str): The categorical data sequence to tokenize.
            continuous_data (list of float): The continuous data sequence associated with the categorical data.
            max_length (int): The maximum length of the tokenized sequence, used for padding.

        Returns:
            dict: A dictionary containing tokenized and padded `input_ids`, `continuous` data, and an `attention_mask`.
        """
        categorical_tokens = [self.vocab.get(item, self.null_token) for item in categorical_data]
        

        # if "[MASK]", "[NULL]", or "[PAD]" in continuous_data, tokenize:
        continuous_tokens = [self.vocab.get(value, self.null_token) if isinstance(value, str) and value in self.special_tokens else float(value) for value in continuous_data]
        

        # Padding
        pad_length = max_length - len(categorical_tokens)
        categorical_tokens.extend([self.pad_token] * pad_length)
        continuous_tokens.extend([self.pad_token] * pad_length)

        # Output structure
        output = {
            "input_ids": categorical_tokens,
            "continuous": continuous_tokens,
            "attention_mask": [0 if token == self.pad_token else 1 for token in categorical_tokens]
        }
        
        if return_tensors:
            output = {key: torch.tensor(val, dtype=torch.long) if key == "input_ids" else torch.tensor(val, dtype=torch.float32) for key, val in output.items()}

        return output
    
    def tokenize_batch(self, categorical_data_batch, continuous_data_batch, max_length=None, return_tensors="np"):
        """
        Tokenizes and encodes a batch of sequences, handling both categorical and continuous data with padding.

        Args:
            categorical_data_batch (list of list of str): A batch of categorical data sequences.
            continuous_data_batch (list of list of float): A batch of continuous data sequences corresponding to the categorical data.
            max_length (int, optional): An optional maximum length to pad the sequences to. If not provided, the longest sequence in the batch is used.

        Returns:
            dict: A dictionary containing the batch's `input_ids`, `continuous` data, and `attention_mask` as numpy arrays.
        """
        batch_size = len(categorical_data_batch)
        max_seq_length = max_length if max_length is not None else max(len(seq) for seq in categorical_data_batch)

        # Initialize lists to hold tokenized sequences
        batch_input_ids = []
        batch_continuous_tokens = []
        batch_attention_masks = []

        for i in range(batch_size):
            # Tokenize each sequence in the batch
            output = self.tokenize(categorical_data_batch[i], continuous_data_batch[i], max_seq_length)

            # Append tokenized sequences to their respective lists
            batch_input_ids.append(output["input_ids"])
            batch_continuous_tokens.append(output["continuous"])
            batch_attention_masks.append(output["attention_mask"])

        # Convert lists to numpy arrays for easier handling and compatibility with PyTorch/TensorFlow
        batch_input_ids = np.array(batch_input_ids)
        batch_continuous_tokens = np.array(batch_continuous_tokens)
        batch_attention_masks = np.array(batch_attention_masks)

        # Output structure for the whole batch
        if return_tensors == "np":
            output_batch = {
                "input_ids": batch_input_ids,
                "continuous": batch_continuous_tokens,
                "attention_mask": batch_attention_masks
            }
        elif return_tensors == "pt":
            output_batch = {
                "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                "continuous": torch.tensor(batch_continuous_tokens, dtype=torch.float32),
                "attention_mask": torch.tensor(batch_attention_masks, dtype=torch.float32)
            }
        else:
            raise ValueError("return_tensors must be either 'np' or 'pt'.")

        return output_batch
    
    def get_special_tokens(self):
        """
        Retrieves the special tokens and their corresponding IDs.

        Returns:
            tuple: A tuple containing a list of special tokens and a list of their corresponding IDs.
        """
        return {token: self.vocab[token] for token in self.special_tokens}
    
    def vocab_size(self):
        """
        Retrieves the size of the vocabulary.

        Returns:
            int: The size of the vocabulary.
        """
        return len(self.vocab)

    def decode(self, tokens):
        """
        Converts a sequence of token IDs back into their string representations.

        Args:
            tokens (list of int): A list of token IDs to decode.

        Returns:
            list of str: The decoded string representations of the token IDs.
        """
        return [self.inverse_vocab.get(token, "[UNK]") for token in tokens]
    
    def get_masking_indices(self, length, total_length, masking_prob=0.15):
        """
        Retrieves the indices of tokens to mask, based on the masking probability.

        Args:
            length (int): The length of the sequence to mask.
            masking_prob (float, optional): The probability of masking a token. Default is 0.15.

        Returns:
            list of int: The indices of tokens to mask.
        """
        length = int(length)

        indices = np.random.rand(length) < masking_prob

        if indices.sum() == 0:
            idx = [np.random.randint(length)]
            return idx

        # Get the indices of the tokens to mask (positions where indices is True)
        indices = np.arange(length)[indices]
            
        return indices
        
        

    def mask_tokens(self, tokens, masking_prob=0.15):
        """
        Masks a sequence of tokens according to the specified masking probability.

        Args:
            tokens (list of int): A list of token IDs to mask.
            masking_prob (float, optional): The probability of masking a token. Default is 0.15.

        Returns:
            list of int: The masked token IDs.
        """
        masked_tokens = tokens.copy()
        if masked_tokens['attention_mask'].sum() < 2:
            return masked_tokens
        else:
            length_list = masked_tokens['attention_mask'].sum()
        
        # mask the categorical tokens
        indices = self.get_masking_indices(length_list, len(masked_tokens['input_ids']), masking_prob)
        masked_tokens['input_ids'][indices] = self.mask_token
        
        indices = self.get_masking_indices(length_list, len(masked_tokens['continuous']),masking_prob)
        masked_tokens['continuous'][indices] = self.mask_token
        return masked_tokens
