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