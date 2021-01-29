import json
import spacy
from collections import Counter, OrderedDict
from itertools import chain
from sqlalchemy.engine.base import Engine
from typing import Optional, Dict
from spacy.matcher import Matcher
from spacy.tokens import Token
import logging


def filter_word_counts(all_counter: dict, vocab_filters: Dict, nlp, matcher):
    """
        Remove vocab words from the word counts Counter
        """
    with nlp.disable_pipes("tagger", "ner", "parser"):
        for k in list(all_counter.keys()):
            spacified_doc = nlp.make_doc(k)
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
                        logging.info(f"problem with merging {k}")
            if len(spacified_doc) > 1:
                logging.info(f"problem with {spacified_doc}")
                del all_counter[k]
            else:
                token = spacified_doc[0]
                if vocab_filters["drop_punct"] and token.is_punct:
                    try:
                        del all_counter[k]
                    except KeyError:
                        continue  # don't need to do the rest of these filters
                elif vocab_filters["drop_non_ascii"] and not token.is_ascii:
                    try:
                        del all_counter[k]
                    except KeyError:
                        continue  # don't need to do the rest of these filters
                elif vocab_filters["drop_numbers"] and token.is_digit:
                    try:
                        del all_counter[k]
                    except KeyError:
                        continue  # don't need to do the rest of these filters

    return OrderedDict(Counter(all_counter).most_common())


class BaseCorpus:
    """
    The full corpus with summary info about
    vocab frequencies
    """

    def __init__(
        self,
        db_engine: Engine,
        db_table: str,
        text_col: str = "text",
        class_var: str = "label",
        order_col: str = "id",
    ):

        self.db_engine = db_engine
        self.db_table = db_table
        self.text_col = text_col
        self.order_col = order_col
        self.class_var = class_var

        # Load the spacy NLP for use with this class
        nlp = spacy.load("en_core_web_sm")
        matcher = Matcher(nlp.vocab)
        matcher.add(
            "HASHTAG",
            None,
            [{"ORTH": "#"}, {"IS_ASCII": True, "IS_PUNCT": False}],
        )
        Token.set_extension("is_hashtag", default=False, force=True)
        self.nlp = nlp
        self.matcher = matcher

        result = self.db_engine.execute(
            f"SELECT COUNT(*) FROM {self.db_table}"
        )
        self.corpus_size = next(result)[0]

        self.vocab_filters = {}
        self.word_counts = None
        self.vocab_size = None
        self.word_doc_counts = None

        result = self.db_engine.execute(
            f"SELECT DISTINCT({class_var}) FROM {self.db_table}"
        )
        self.class_names = [i[0] for i in result]
        self.n_classes = len(self.class_names)

    def __len__(self):
        return self.corpus_size

    def compute_word_counts(
        self,
        drop_punct: bool = False,
        drop_numbers: bool = False,
        drop_non_ascii: bool = False,
        min_count: Optional[int] = None,
    ):
        """
        Spacy-tokenizes texts and stores the frequencies for
        all words in the vocab, after applying optional filters
        """

        self.vocab_filters["drop_punct"] = drop_punct
        self.vocab_filters["drop_numbers"] = drop_numbers
        self.vocab_filters["drop_non_ascii"] = drop_non_ascii

        all_docs = self.db_engine.execute(
            f"SELECT {self.text_col} FROM {self.db_table}"
        )

        all_tokenized = []
        with self.nlp.disable_pipes("tagger", "ner", "parser"):
            for doc in all_docs:
                spacified_doc = self.nlp.make_doc(doc[0])
                if len(spacified_doc) > 3:
                    # this bit handles hashtags
                    matches = self.matcher(spacified_doc)
                    hashtags = []
                    for match_id, start, end in matches:
                        if spacified_doc.vocab.strings[match_id] == "HASHTAG":
                            hashtags.append(spacified_doc[start:end])
                    with spacified_doc.retokenize() as retokenizer:
                        for span in hashtags:
                            try:
                                retokenizer.merge(span)
                            except IndexError:
                                logging.info(f"problem with {doc}")

                    all_tokenized.append(
                        [i.text for i in spacified_doc if (not i.is_space)]
                    )

        all_counter = Counter(list(chain.from_iterable(all_tokenized)))
        all_counter = OrderedDict(all_counter.most_common())

        if min_count:
            all_counter = OrderedDict(
                {k: v for k, v in all_counter.items() if v > min_count}
            )

        self.word_counts = filter_word_counts(
            all_counter, self.vocab_filters, self.nlp, self.matcher
        )

        self.vocab_size = len(self.word_counts)

    def get_corpus_top_words(
        self,
        num_words: Optional[int] = None,
        return_list: bool = True,
        drop_punct: bool = False,
        drop_numbers: bool = False,
        drop_non_ascii: bool = False,
        min_count: Optional[int] = None,
    ):
        """
        Returns all words and their counts if num_words is not specified,
        otherwise returns the most frequent words
        Returns a dictionary with the counts unless return_list is specified
        """
        if (
            not self.word_counts
            or drop_punct != self.vocab_filters["drop_punct"]
            or drop_numbers != self.vocab_filters["drop_numbers"]
            or drop_non_ascii != self.vocab_filters["drop_non_ascii"]
        ):
            print("Computing word counts")
            self.compute_word_counts(
                drop_punct, drop_numbers, drop_non_ascii, min_count
            )

        if not num_words:
            num_words = len(self.word_counts)

        if num_words > len(self.word_counts):
            print(
                f"{num_words} requested, but vocab contains {len(self.word_counts)}."
            )
            counts = dict(self.word_counts.most_common(len(self.word_counts)))
        else:
            counts = dict(self.word_counts.most_common(num_words))

        if return_list:
            return list(counts.keys())

        else:
            return counts  # returns dict

    def load_top_words_json(
        self, top_words_filepath: str, vocab_filters_filepath: str
    ):

        self.vocab_filters = vocab_filters_filepath
        with open(top_words_filepath, "r") as f:
            top_words = json.load(f)

        self.word_counts = Counter(top_words)
        self.vocab_size = len(self.word_counts)

    def save_top_words(
        self,
        top_words_filepath: str = "topwords.json",
        vocab_filters_filepath: str = "vocab_filters.json",
        min_count: Optional[int] = None,
    ):
        if not self.word_counts:
            self.compute_word_counts(min_count=min_count)
        print("Dumping top words to json")
        with open(top_words_filepath, "w") as f:
            json.dump(self.word_counts, f)
        print("Dumping vocab filter to json")
        with open(vocab_filters_filepath, "w") as f:
            json.dump(self.vocab_filters, f)

    def compute_word_doc_counts(
        self,
        drop_punct: bool = False,
        drop_numbers: bool = False,
        drop_non_ascii: bool = False,
        min_count: Optional[int] = None,
    ):
        """
        Spacy-tokenizes texts and stores the number of documents each word
        occurred in, after applying optional filters
        We limit ourselves to docs of at least 3 spacy tokens
        """

        self.vocab_filters["drop_punct"] = drop_punct
        self.vocab_filters["drop_numbers"] = drop_numbers
        self.vocab_filters["drop_non_ascii"] = drop_non_ascii

        all_docs = self.db_engine.execute(
            f"SELECT {self.text_col} FROM {self.db_table}"
        )

        all_tokenized = []
        with self.nlp.disable_pipes("tagger", "ner", "parser"):
            for doc in all_docs:
                spacified_doc = self.nlp.make_doc(doc[0])
                if len(spacified_doc) > 3:
                    # this bit handles hashtags
                    matches = self.matcher(spacified_doc)
                    hashtags = []
                    for match_id, start, end in matches:
                        if spacified_doc.vocab.strings[match_id] == "HASHTAG":
                            hashtags.append(spacified_doc[start:end])
                    with spacified_doc.retokenize() as retokenizer:
                        for span in hashtags:
                            try:
                                retokenizer.merge(span)
                            except IndexError:
                                print(doc)

                    all_tokenized.append(
                        (
                            set(
                                [
                                    i.text
                                    for i in spacified_doc
                                    if not i.is_space
                                ]
                            )
                        )
                    )

        all_counter = Counter(chain.from_iterable(all_tokenized))
        all_counter = OrderedDict(all_counter.most_common())

        if min_count:
            all_counter = OrderedDict(
                {k: v for k, v in all_counter.items() if v > min_count}
            )

        try:
            self.word_doc_counts = filter_word_counts(
                all_counter, self.vocab_filters, self.nlp, self.matcher
            )
        except Exception as e:
            self.word_doc_counts = all_counter
            logging.info(f"problem with filter_word_counts: {e}")
