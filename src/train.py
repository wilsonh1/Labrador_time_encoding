import matplotlib.pyplot as plt
import torch
from transformers import AdamW
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
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
    
    


def train_labrador(model, train_loader, val_loader, categorical_loss_fn, continuous_loss_fn, optimizer='Adam', num_epochs=2, device='cpu', save_model=False, model_path='labrador_model.pth'):
    train_losses_per_iter = []
    val_losses_per_iter = []
    train_losses_per_epoch = []
    val_losses_per_epoch = []

    model.to(device)
    
    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)    
    elif optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    else:
        raise ValueError("Please specify a valid optimizer (Adam or SGD)")
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        train_loop = tqdm(train_loader, leave=True)
        for batch in train_loop:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            continuous = batch['continuous'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels_input_ids = batch['labels_input_ids'].to(device)
            labels_continuous = batch['labels_continuous'].to(device)

            outputs = model(input_ids, continuous, attn_mask=attn_mask)
            
            masked_cat_indices = (input_ids == train_loader.dataset.tokenizer.mask_token).to(device)
            categorical_loss = categorical_loss_fn(outputs['categorical_output'][masked_cat_indices], labels_input_ids[masked_cat_indices])
            
            masked_cont_indices = (continuous == train_loader.dataset.tokenizer.mask_token).to(device)
            continuous_loss = continuous_loss_fn(outputs['continuous_output'][masked_cont_indices].squeeze(), labels_continuous[masked_cont_indices])
            continuous_loss = torch.sqrt(continuous_loss)

            loss = categorical_loss + continuous_loss
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_losses_per_iter.append(loss.item())
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses_per_epoch.append(avg_train_loss)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            val_loop = tqdm(val_loader, leave=True)
            for batch in val_loop:
                input_ids = batch['input_ids'].to(device)
                continuous = batch['continuous'].to(device)
                attn_mask = batch['attention_mask'].to(device)
                labels_input_ids = batch['labels_input_ids'].to(device)
                labels_continuous = batch['labels_continuous'].to(device)

                outputs = model(input_ids, continuous, attn_mask=attn_mask)

                masked_cat_indices = (input_ids == val_loader.dataset.tokenizer.mask_token).to(device)
                categorical_loss = categorical_loss_fn(outputs['categorical_output'][masked_cat_indices], labels_input_ids[masked_cat_indices])
                
                masked_cont_indices = (continuous == val_loader.dataset.tokenizer.mask_token).to(device)
                continuous_loss = continuous_loss_fn(outputs['continuous_output'][masked_cont_indices].squeeze(), labels_continuous[masked_cont_indices])
                continuous_loss = torch.sqrt(continuous_loss)

                val_loss = categorical_loss + continuous_loss
                total_val_loss += val_loss.item()
                val_losses_per_iter.append(val_loss.item())

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses_per_epoch.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")

        if save_model:
            torch.save(model.state_dict(), model_path)

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses_per_iter, label='Training Loss (Per Iteration)')
    #plt.plot(range(0, len(train_losses_per_iter), len(train_loader)), val_losses_per_iter, label='Validation Loss (Per Iteration)', linestyle='--')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses Per Iteration')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_losses_per_epoch, label='Training Loss (Per Epoch)')
    plt.plot(val_losses_per_epoch, label='Validation Loss (Per Epoch)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses Per Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model
