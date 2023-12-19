import matplotlib.pyplot as plt
import torch
from transformers import AdamW

from tqdm import tqdm


def train_mlm(model, train_loader, test_loader, device, tokenizer, lr=5e-5, epochs=2, save=True, model_path="model/", tokenizer_path="tokenizer/"):
    """
    Trains a masked language model (MLM) using a specified DataLoader for training and evaluation.

    Parameters:
    -----------
    model : transformers.modeling_utils.PreTrainedModel
        The pre-trained model to be fine-tuned for MLM.

    train_loader : torch.utils.data.DataLoader
        DataLoader for training data.

    test_loader : torch.utils.data.DataLoader
        DataLoader for test/evaluation data.

    device : torch.device
        The device to which the model and data should be moved (e.g., "cuda" for GPU).

    lr : float, optional
        Learning rate for the optimizer. Default is 5e-5.

    epochs : int, optional
        Number of training epochs. Default is 2.

    save : bool, optional
        If True, save the trained model and tokenizer. Default is True.

    model_path : str, optional
        The path to save the trained model. Default is "model/".

    tokenizer_path : str, optional
        The path to save the tokenizer. Default is "tokenizer/".

    Returns:
    --------
    None
    """
    
    optim = AdamW(model.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loop = tqdm(train_loader, leave=True)
        for batch in train_loop:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optim.step()
            train_losses.append(loss.item())
            train_loop.set_description(f'Epoch {epoch}')
            train_loop.set_postfix(loss=loss.item())

        # Evaluation
        model.eval()
        test_loop = tqdm(test_loader, leave=True)
        for batch in test_loop:
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                test_losses.append(loss.item())
                test_loop.set_description(f'Epoch {epoch} (Eval)')
                test_loop.set_postfix(loss=loss.item())

    if save:
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(tokenizer_path)
    
    
    # Plotting
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()