from typing import List, Optional
import numpy as np
import fasttext
import logging
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Token


def get_sentence_minus_word(
    input_sentence: str, target_word: str, nlp, matcher
) -> List:
    # to compute representation of sentence minus target word,
    # we need a vector where the target word has been deleted
    spacified_doc = nlp.make_doc(input_sentence)
    if len(spacified_doc) > 1:
        # this bit handles hashtags
        matches = matcher(spacified_doc)
        hashtags = []
        for match_id, start, end in matches:
            if spacified_doc.vocab.strings[match_id] == "HASHTAG":
                hashtags.append(spacified_doc[start:end])
        with spacified_doc.retokenize() as retokenizer:
            for span in hashtags:
                try:
                    retokenizer.merge(span)
                except IndexError:
                    logging.info(f"problem with {span}")
    tokens = [i.text for i in spacified_doc if (not i.is_space)]
    if target_word in tokens:
        result = [t for t in tokens if t != target_word]
        # this may be an empty list if the target word is the only word in the tweet
        return result
    else:
        return []


def fasttext_word_norm(
    token: str, model: fasttext.FastText
) -> Optional[np.array]:
    # divide each word vector by its L2 norm
    # and we will only sum over vectors with positive L2 norms

    raw_word_vec = model.get_word_vector(token)
    norm = np.linalg.norm(raw_word_vec)

    if norm > 0:
        return raw_word_vec / np.linalg.norm(raw_word_vec)
    else:
        # these Nones are dropped in get_fasttext_sentence_vector
        # so that we don't sum over ones with zero norm
        return None


def get_fasttext_sentence_vector(
    tokenized_sentence: List[str], model: fasttext.FastText
) -> np.array:
    # returns the sum over the vector norms of the context without the target word
    token_reps = [
        fasttext_word_norm(word, model) for word in tokenized_sentence
    ]
    # sum over vectors (we have skipped ones with negative norms)
    sentence_rep = np.sum([t for t in token_reps if t is not None], axis=0)

    return sentence_rep


def get_fasttext_sentence_minus_target(
    input_sentence: str,
    target_word: str,
    model: fasttext.FastText,
    nlp,
    matcher,
) -> np.array:
    tokenized_context = get_sentence_minus_word(
        input_sentence, target_word, nlp, matcher
    )
    if tokenized_context:
        context_rep = get_fasttext_sentence_vector(tokenized_context, model)
        target_word_rep = fasttext_word_norm(target_word, model)

        # subtract target word from sentence,
        # then take the average over num context words
        subtracted_rep = np.subtract(context_rep, target_word_rep)
        subtracted_rep /= len(tokenized_context)

        return subtracted_rep
    else:
        return None
