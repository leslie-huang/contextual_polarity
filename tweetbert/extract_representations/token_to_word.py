from transformers import PreTrainedTokenizerFast
import torch
from typing import List, Dict


def map_word_representations_from_subwords(
    input_ids_batched: torch.Tensor,
    last_hidden_state: List[torch.Tensor],
    model_name: str,
    tokenizer: PreTrainedTokenizerFast,
    word_boundary: str = "Ġ",
) -> List[Dict]:
    """
    Maps wordpiece tokens and their associated representations
    back to the source word
    Return List[Dict] where each element is a Dict for a document and contains:
    reconstructed_tokens: List[str] of the word in the doc
    token_representations: List[np.array] of the representation(s) for that word
    this list may be one element if the word is in the wordpiece vocabulary
    or it may be multiple elements if the word was tokenized into subwords
    """

    output = []

    for doc_id in range(input_ids_batched.size()[0]):
        reconstructed_tokens = []
        token_representations = []

        wordpiece_tokens = tokenizer.convert_ids_to_tokens(
            input_ids_batched[doc_id, :], skip_special_tokens=True
        )  # we automatically skip the padding tokens

        # ROBERTA: last_hidden_state is a LIST of len num_docs,
        # where each element is a document Tensor of size
        # (max_length, hidden_size)
        # GPT2: last_hidden_state is a Tensor of size
        # (num_docs, max_length, hidden_size)
        if model_name == "roberta":
            doc_representation = last_hidden_state[doc_id].cpu().numpy()
        if model_name == "gpt2":
            doc_representation = last_hidden_state[doc_id, :, :].cpu().numpy()

        # doc_representation has dimension (max_length, hidden_size)
        # and is a numpy array now

        for idx, token in enumerate(wordpiece_tokens):
            if token.startswith(word_boundary):
                # starting (or only) subword token of a word
                reconstructed_tokens.append(
                    token[1:]
                )  # 1: to exclude the word_boundary token
                token_representations.append([doc_representation[idx, :]])
            else:
                # if it doesn't start with the word boundary,
                # we need to combine it with the previous subword
                if reconstructed_tokens:
                    reconstructed_tokens[
                        -1
                    ] = f"{reconstructed_tokens[-1]}{token}"
                    token_representations[-1].append(
                        doc_representation[idx, :]
                    )

        output.append(
            {
                "reconstructed_tokens": reconstructed_tokens,
                "token_representations": token_representations,
            }
        )

    return output
