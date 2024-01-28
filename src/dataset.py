import torch

class LabValuesDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset class for handling laboratory values encoded using Hugging Face's tokenizers.

    Parameters:
    -----------
    encodings : transformers.tokenization_utils_base.BatchEncoding
        The batch encoding containing tokenized inputs (e.g., 'input_ids', 'attention_mask', etc.).

    Methods:
    --------
    __getitem__(idx):
        Retrieves the encoded values for a specific index in the dataset.

    __len__():
        Returns the total number of samples in the dataset.

    Example Usage:
    --------------
    # Instantiate the dataset
    dataset = LabValuesDataset(encodings)

    # Access a specific sample
    sample = dataset[0]

    # Get the total number of samples in the dataset
    num_samples = len(dataset)
    """
    def __init__(self, encodings):
        """
        Initializes the LabValuesDataset with the provided batch encoding.

        Parameters:
        -----------
        encodings : transformers.tokenization_utils_base.BatchEncoding
            The batch encoding containing tokenized inputs.
        """
        self.encodings = encodings

    def __getitem__(self, idx):
        """
        Retrieves the encoded values for a specific index in the dataset.

        Parameters:
        -----------
        idx : int
            The index of the sample to be retrieved.

        Returns:
        --------
        dict
            A dictionary containing tensor values for each input key in the batch encoding.
        """
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
        --------
        int
            The total number of samples in the dataset.
        """
        return len(self.encodings.input_ids)
    

class LabradorDataset(torch.utils.data.Dataset):
    """PyTorch Dataset class for handling continuous values and categorical ids.
    
    Parameters:
    -----------
    continuous : torch.Tensor
        A tensor containing the continuous values.
    categorical : torch.Tensor
        A tensor containing the categorical ids.
    tokenier : Tokenizer
        A pre-trained tokenizer for encoding the categorical and continuous values.
    max_len : int
        The maximum length of the encoded sequence.

    Args:
        torch (_type_): _description_
    """
    
    def __init__(self, continuous, categorical, tokenizer, max_len, masking_prob=0.15):
        self.continuous = continuous
        self.categorical = categorical
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.masking_prob = masking_prob
        self.mask_token = tokenizer.mask_token
        print(tokenizer.vocab)
        
    def __len__(self):
        return len(self.continuous)
    
    def __getitem__(self, idx):
        continuous = self.continuous[idx]
        categorical = self.categorical[idx]
        
        encoded = self.tokenizer.tokenize(
            categorical_data=categorical, 
            continuous_data=continuous, 
            max_length=self.max_len, 
            return_tensors='pt'
        )
        
        labels = {
            'labels_input_ids': encoded['input_ids'].clone().detach().flatten(),
            'labels_continuous': encoded['continuous'].clone().detach().flatten()
        }
        
        if self.masking_prob > 0:
            masked_values = self.tokenizer.mask_tokens(encoded, self.masking_prob)
            inputs = {
                'input_ids': masked_values['input_ids'].flatten(),
                'continuous': encoded['continuous'].flatten(),
                'attention_mask': masked_values['attention_mask'].flatten()
            }
        else:
            inputs = {
                'input_ids': encoded['input_ids'].flatten(),
                'continuous': encoded['continuous'].flatten(),
                'attention_mask': encoded['attention_mask'].flatten()
            }
        
        return {**inputs, **labels}