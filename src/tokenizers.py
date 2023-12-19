from transformers import BertTokenizer

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
