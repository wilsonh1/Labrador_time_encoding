import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def test_model(model, test_loader, device, labs_list):
    model.to(device)
    model.eval()

    metrics = {lab: {'rmse': [], 'mae': [], 'r2': []} for lab in labs_list}

    with torch.no_grad():
        for lab in labs_list:
            print(f'Evaluating {lab}: ')
            lab_token = test_loader.dataset.tokenizer.vocab[lab]

            preds = []
            true_vals = []
            count = 0

            for batch in tqdm(test_loader, leave=True):
                lab_idx = (batch['input_ids'] == lab_token)
                batch['continuous'][lab_idx] = torch.tensor(test_loader.dataset.tokenizer.mask_token, dtype=torch.float32, device=device)

                input_ids = batch['input_ids'].to(device)
                continuous = batch['continuous'].to(device)
                attn_mask = batch['attention_mask'].to(device)
                labels_continuous = batch['labels_continuous'].to(device)
                
                if count == 0:
                    #print(f'Input ids: {input_ids}')
                    #print(f'Continuous: {continuous}')
                    #print(f'Attn mask: {attn_mask}')
                    #print(f'Labels continuous: {labels_continuous}')
                    pass
                
                # If no 0s in attention mask, set it to None
                attn_mask_zero = (attn_mask == 0).any().item()
                if attn_mask_zero == 0:
                    # print('No 0s in attention mask')
                    attn_mask = None

                outputs = model(input_ids, continuous, attn_mask=attn_mask)
                continuous_output = outputs['continuous_output'].squeeze(-1)
                if count == 0:
                    #print(f'Continuous output: {continuous_output}')
                    pass

                masked_cont_indices = (continuous == test_loader.dataset.tokenizer.mask_token).to(device)
                batch_preds = continuous_output[masked_cont_indices]
                batch_labels = labels_continuous[masked_cont_indices].to(device)

                preds.extend(batch_preds.tolist())
                true_vals.extend(batch_labels.tolist())
                
                if count == 0:
                    print(f'Preds: {batch_preds.tolist()}')
                    print(f'True vals: {batch_labels.tolist()}')
                    count += 1

            rmse = np.sqrt(mean_squared_error(true_vals, preds))
            mae = mean_absolute_error(true_vals, preds)
            r2 = r2_score(true_vals, preds)

            metrics[lab]['rmse'].append(rmse)
            metrics[lab]['mae'].append(mae)
            metrics[lab]['r2'].append(r2)

            print(f'RMSE: {rmse:.3f}')
            print(f'MAE: {mae:.3f}')
            print(f'R2: {r2:.3f}')
            print('-------------------')

    return metrics
