from transformers import PreTrainedModel, PreTrainedTokenizerFast
import torch
from tweetbert.extract_representations.token_to_word import (
    map_word_representations_from_subwords,
)


def get_contextual_word_representations(
    model: PreTrainedModel,
    batched_input: tuple,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int,
    device: torch.device,
):
    """
    Gets contextual representations for words
    from one batch of inputs
    Cones with the reconstructed (from RoBERTa not Spacy) tokens
    """
    model.eval()

    with torch.no_grad():
        tweets = [i for i in batched_input[0]]
        labels = (
            batched_input[1].cpu().numpy().tolist()
        )  # this can just be on cpu

        tokenization_dict = tokenizer.batch_encode_plus(
            tweets,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        for k, v in tokenization_dict.items():
            tokenization_dict[k] = v.to(device)

        last_hidden, _, _ = model(
            **tokenization_dict, output_hidden_states=True
        )

        reconstructed_tokens = map_word_representations_from_subwords(
            tokenization_dict["input_ids"],
            last_hidden,
            model_name="roberta",
            tokenizer=tokenizer,
        )

    return reconstructed_tokens, labels
