import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

from sklearn.preprocessing import StandardScaler

def log_transform_scale_and_bin(data, num_bins=None):

    # Handle Pandas DataFrame input
    if isinstance(data, pd.DataFrame):
        data = data.values
        
    # Apply logarithmic transformation with a small constant to avoid log(0)
    log_data = np.log(data + 1)

    # Apply standard scaling

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(log_data)


    # Apply binning if num_bins is specified

    if num_bins is not None:

        binned_data = pd.qcut(scaled_data.flatten(), num_bins, labels=False, duplicates='drop').reshape(scaled_data.shape)

        return binned_data

    return scaled_data

def preprocess_df(df, scaler=MinMaxScaler(), columns_to_scale=['Bic', 'Crt', 'Pot', 'Sod', 'Ure', 'Hgb', 'Plt', 'Wbc'], num_bins=10):
    """
    Preprocesses a DataFrame containing medical records, including sorting, sampling, pivoting, renaming columns, 
    dropping NaN values, and scaling selected columns.

    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame containing medical records.

    scaler : sklearn.preprocessing.Scaler, optional
        The scaler object to be used for column scaling. Default is sklearn.preprocessing.MinMaxScaler().

    columns_to_scale : list, optional
        A list of column names in the DataFrame to be scaled. Default is ['Bic', 'Crt', 'Pot', 'Sod', 'Ure', 'Hgb', 'Plt', 'Wbc'].

    Returns:
    --------
    mrl : pandas DataFrame
        The preprocessed DataFrame with scaled numerical columns.
    """
    mrl_sorted = df.sort_values(by=['subject_id', 'hadm_id', 'chartday', 'itemid', 'charthour'])
    mrl_sampled = mrl_sorted.groupby(['subject_id', 'hadm_id', 'chartday', 'itemid']).first().reset_index()
    mrl_full = mrl_sampled.pivot(index=['subject_id', 'hadm_id', 'chartday'], columns='itemid', values='valuenum').reset_index()
    mrl = mrl_full.dropna()
    mrl = mrl.rename(columns={50882: 'Bic', 50912: 'Crt', 50971: 'Pot', 50983: 'Sod', 51006: 'Ure', 51222: 'Hgb', 51265: 'Plt', 51301: 'Wbc'})
    columns_to_scale = ['Bic', 'Crt', 'Pot', 'Sod', 'Ure', 'Hgb', 'Plt', 'Wbc']
    
    
    if scaler == 'log':
        mrl[columns_to_scale] = log_transform_scale_and_bin(mrl[columns_to_scale], num_bins=num_bins)
    else:
        # Scale selected columns
        mrl[columns_to_scale] = scaler.fit_transform(mrl[columns_to_scale])
    
    return mrl



def random_train_test_split(data, train_percent=.8):
    """
    Randomly splits a dataset into training and test sets.

    Parameters:
    -----------
    data : list or array-like
        The input data to be split.

    train_percent : float, optional
        The percentage of data to be used for training. Default is 0.8.

    Returns:
    --------
    train : list
        The training subset of the input data.

    test : list
        The test subset of the input data.
    """
    # Randomly split data into training and test sets.
    np.random.seed(42)
    data = np.array(data)
    np.random.shuffle(data)
    train_size = int(len(data) * train_percent)
    train = data[:train_size]
    test = data[train_size:]
    # Convert to list
    train = train.tolist()
    test = test.tolist()
    return train, test



class TextEncoder:
    """
    TextEncoder class for encoding numerical values into a text representation.

    Methods:
    --------
    __init__():
        Initializes the TextEncoder instance and generates a mapping of integers to letters.

    int_to_letters(number):
        Converts a non-negative integer into a corresponding uppercase letter-based representation.

    scale_to_letter(value):
        Converts a scaled numerical value into a corresponding letter based on a predefined mapping.

    encode_text(df, columns_to_scale=['Bic', 'Crt', 'Pot', 'Sod', 'Ure', 'Hgb', 'Plt', 'Wbc']):
        Encodes numerical values in the specified columns of a DataFrame into a text representation.

    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame containing numerical values to be encoded.

    columns_to_scale : list, optional
        A list of column names in the DataFrame to encode. Default is ['Bic', 'Crt', 'Pot', 'Sod', 'Ure', 'Hgb', 'Plt', 'Wbc'].

    Returns:
    --------
    df : pandas DataFrame
        The input DataFrame with an additional 'nstr' column containing the encoded text.

    grouped_df : pandas DataFrame
        A DataFrame containing grouped encoded text lists based on the 'hadm_id' column.
    """
    def __init__(self, bins=None, Repetition_id=False, lab_id=False):
        """
        Initializes the TextEncoder instance and generates a mapping of integers to letters.
        """
        self.Repetition_id = Repetition_id
        self.lab_id = lab_id
        self.bins = bins
        if self.bins:
            # Generate letters for numbers from 0 to num_bins
            self.letters_mapping = {i: self.int_to_letters(i) for i in range(self.bins)}
        else:
            # Generate letters for numbers from 0 to 99
            self.letters_mapping = {i: self.int_to_letters(i) for i in range(101)}
            

    def int_to_letters(self, number):
        """
        Converts a non-negative integer into a corresponding uppercase letter-based representation.

        Parameters:
        -----------
        number : int
            The non-negative integer to be converted.

        Returns:
        --------
        str
            Uppercase letter-based representation of the input integer.
        """
        result = ""
        while number >= 0:
            number, remainder = divmod(number, 26)
            result = chr(65 + remainder) + result
            if number == 0:
                break
            number -= 1  # Adjust for 0-based indexing
        return result

    def scale_to_letter(self, value):
        """
        Converts a scaled numerical value into a corresponding letter based on a predefined mapping.

        Parameters:
        -----------
        value : float
            The scaled numerical value to be converted.

        Returns:
        --------
        str
            Uppercase letter based on the input scaled value.
        """
        if self.bins:
            return self.letters_mapping[int(value)]
        else:
            return self.letters_mapping[int(value * 100)]

    def encode_text(self, df, columns_to_scale=['Bic', 'Crt', 'Pot', 'Sod', 'Ure', 'Hgb', 'Plt', 'Wbc']):
        """
        Encodes numerical values in the specified columns of a DataFrame into a text representation.

        Parameters:
        -----------
        df : pandas DataFrame
            The input DataFrame containing numerical values to be encoded.

        columns_to_scale : list, optional
            A list of column names in the DataFrame to encode. Default is ['Bic', 'Crt', 'Pot', 'Sod', 'Ure', 'Hgb', 'Plt', 'Wbc'].

        Returns:
        --------
        df : pandas DataFrame
            The input DataFrame with an additional 'nstr' column containing the encoded text.

        grouped_df : pandas DataFrame
            A DataFrame containing grouped encoded text lists based on the 'hadm_id' column.
        """
        if self.Repetition_id:
            if self.lab_id:
                df['nstr'] = df[columns_to_scale].apply(
                    lambda row: ' '.join(f'{col} {col}{self.scale_to_letter(val)}' for col, val in zip(columns_to_scale, row)),
                    axis=1)
            else:
                df['nstr'] = df[columns_to_scale].apply(
                    lambda row: ' '.join(f'{col} {self.scale_to_letter(val)}' for col, val in zip(columns_to_scale, row)),
                    axis=1)
        else:
            if self.lab_id:
                df['nstr'] = df[columns_to_scale].apply(
                    lambda row: ' '.join(f'{col}{self.scale_to_letter(val)}' for col, val in zip(columns_to_scale, row)),
                    axis=1)
            else:
                df['nstr'] = df[columns_to_scale].apply(
                    lambda row: ' '.join(f'{self.scale_to_letter(val)}' for col, val in zip(columns_to_scale, row)),
                    axis=1)                
        grouped_df = df.groupby('hadm_id')['nstr'].apply(list).reset_index()

        return df, grouped_df

    
def randomly_mask_dataset(inputs, parcentage=0.20, CLS=101, SEP=102, PAD=0, MASK=103):
    """
    Randomly masks a percentage of non-special tokens in the input tensor.

    Parameters:
    -----------
    inputs : transformers.tokenization_utils_base.BatchEncoding
        The input batch encoding containing 'input_ids' tensor.

    percentage : float, optional
        The percentage of non-special tokens to be randomly masked. Default is 0.20.

    CLS : int, optional
        The token ID for the [CLS] special token. Default is 101.

    SEP : int, optional
        The token ID for the [SEP] special token. Default is 102.

    PAD : int, optional
        The token ID for the [PAD] special token. Default is 0.

    Returns:
    --------
    inputs : transformers.tokenization_utils_base.BatchEncoding
        The modified input batch encoding with randomly masked tokens.
    """
    # create random array of floats with equal dimensions to input_ids tensor
    rand = torch.rand(inputs.input_ids.shape)
    # Create mask array
    mask_arr = (rand < parcentage) * (inputs.input_ids != CLS) * (inputs.input_ids != SEP) * (inputs.input_ids != PAD)

    # Take the index of each masked token and replace with the token 103
    masked = []

    for i in range(inputs.input_ids.shape[0]):
        masked.append(torch.flatten(mask_arr[i].nonzero()).tolist())

    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, masked[i]] = MASK

    return inputs

def set_labels_features(train, test, parcentage=0.20, CLS=101, SEP=102, PAD=0, MASK=103):
    """
    Sets labels in the input tensors and randomly masks a percentage of non-special tokens.

    Parameters:
    -----------
    train : transformers.tokenization_utils_base.BatchEncoding
        The training batch encoding containing 'input_ids' tensor.

    test : transformers.tokenization_utils_base.BatchEncoding
        The testing batch encoding containing 'input_ids' tensor.

    percentage : float, optional
        The percentage of non-special tokens to be randomly masked. Default is 0.20.

    Returns:
    --------
    train : transformers.tokenization_utils_base.BatchEncoding
        The modified training batch encoding with labels and randomly masked tokens.

    test : transformers.tokenization_utils_base.BatchEncoding
        The modified testing batch encoding with labels and randomly masked tokens.
    """
    train['labels'] = train.input_ids.detach().clone()
    test['labels'] = test.input_ids.detach().clone()

    train = randomly_mask_dataset(train, parcentage=parcentage, CLS=CLS, SEP=SEP, PAD=PAD, MASK=MASK)
    test = randomly_mask_dataset(test, parcentage=parcentage, CLS=CLS, SEP=SEP, PAD=PAD, MASK=MASK)
    
    return train, test