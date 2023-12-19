import numpy as np
from transformers import BertForMaskedLM, BertTokenizer

def load_model(model_path="model/", tokenizer_path="tokenizer/"):
    """
    Loads a pre-trained BERT model and tokenizer from the specified paths.

    Parameters:
    -----------
    model_path : str, optional
        The path to the directory containing the saved BERT model. Default is "model/".

    tokenizer_path : str, optional
        The path to the directory containing the saved BERT tokenizer. Default is "tokenizer/".

    Returns:
    --------
    model : transformers.BertForMaskedLM
        The loaded BERT model for masked language modeling.

    tokenizer : transformers.BertTokenizer
        The loaded BERT tokenizer for tokenizing input texts.
    """
    # Load the saved model and tokenizer
    model = BertForMaskedLM.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    
    return model, tokenizer

def get_embeddings(model, tokenizer, texts):
    """
    Obtains word embeddings for input texts using a pre-trained BERT model.

    Parameters:
    -----------
    model : transformers.BertForMaskedLM
        The pre-trained BERT model for obtaining embeddings.

    tokenizer : transformers.BertTokenizer
        The pre-trained BERT tokenizer for tokenizing input texts.

    texts : str or List[str]
        The input text or a list of texts for which embeddings are to be obtained.

    Returns:
    --------
    embeddings : numpy.ndarray
        An array containing the word embeddings for the input texts, extracted from the last hidden state of the BERT model.
    """
    model.eval()
    
    # Tokenize and convert the input texts to tensors
    tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    
    # Forward pass to obtain the embeddings
    with torch.no_grad():
        embeddings = model(**tokens, output_hidden_states=True, return_dict=True)

    # Extract the embeddings from the last hidden state
    embeddings = np.array(embeddings.hidden_states[-1])

    return embeddings