from typing import List
import logging


def spacify_texts(docs: List, nlp, matcher) -> List[List]:
    """
    Tokenizes list of texts into list of lists of tokens
    """

    spacy_tokenized_contexts = []
    for spacy_doc in nlp.pipe(docs, disable=["tagger", "parser", "ner"]):
        matches = matcher(spacy_doc)
        hashtags = []
        for match_id, start, end in matches:
            if spacy_doc.vocab.strings[match_id] == "HASHTAG":
                hashtags.append(spacy_doc[start:end])
        with spacy_doc.retokenize() as retokenizer:
            for span in hashtags:
                try:
                    retokenizer.merge(span)
                except IndexError as e:
                    logging.info(f"problem with {span}, {e}")

        spacy_tokenized_contexts.append(
            [t.text for t in spacy_doc if not t.is_space]
        )

    return spacy_tokenized_contexts
