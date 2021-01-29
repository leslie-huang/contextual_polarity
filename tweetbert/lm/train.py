from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
from typing import Dict
import logging


def train_step(
    model: PreTrainedModel, batched_input: Dict, device: torch.device
):
    model.train()
    for k, v in batched_input.items():
        batched_input[k] = v.to(device)

    # check device
    if batched_input["input_ids"].device.type != "cuda":
        logging.info(
            "Warning: inputs not on cuda. Is on {batched_input['input_ids'].device.type}"
        )

    # INPUT to model
    # GPT2 has labels param and BERT has masked_lm_labels param
    # both have input_ids, attention_mask
    output = model(**batched_input)

    # OUTPUT
    # output[0] is loss
    # output[1] is prediction scores (logits)

    return (output[0], output[1])


# model tracker utils


def is_new_smaller(old_value: float, new_value: float):
    if old_value is None:  # handle the first epoch
        old_value = new_value
    if new_value <= old_value:
        return True
    else:
        return False


def is_new_bigger(old_value: float, new_value: float):
    if old_value is None:
        old_value = new_value
    if new_value <= old_value:
        return False
    else:
        return True


class ModelTracker:
    def __init__(self, metric_name: str = "loss", objective: str = "minimize"):
        self.metric_name = metric_name
        self.objective = objective
        self.best_epoch = None
        self.best_value = None

        if self.objective == "minimize":
            self.comparison_func = is_new_smaller

        if self.objective == "maximize":
            self.comparison_func = is_new_bigger

    def update_tracker(
        self,
        new_value: float,
        epoch: int,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        model_output_dir: str,
    ):
        if self.comparison_func(
            old_value=self.best_value, new_value=new_value
        ):
            logging.info("Saving new best model")
            self.best_epoch = epoch
            self.best_value = new_value
            model.save_pretrained(model_output_dir)
            tokenizer.save_pretrained(model_output_dir)
