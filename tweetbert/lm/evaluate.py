from transformers import PreTrainedModel
from torch.utils.data import DataLoader
import torch


def evaluate_model(
    model: PreTrainedModel, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    total_loss = 0
    for i, batch in enumerate(loader):

        for k, v in batch.items():
            batch[k] = v.to(device)

        outputs = model(**batch)
        loss = outputs[0]

        total_loss += loss.item()

    return total_loss / len(loader.dataset)
