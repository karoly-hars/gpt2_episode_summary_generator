import random
import torch
import json
from torch.utils.data import Dataset
from pytorch_transformers import GPT2Tokenizer


class EpisodeSummaryTokenizer(GPT2Tokenizer):
    """
    Tokenizer class for episode summary strings/text.
    The class inherits most of its functionality from the GPT2Tokenizer of pytorch_transformers.
    """

    def __init__(self, vocab_file, merges_file, max_num_words, errors='replace',
                 unk_token="<|endoftext|>", bos_token="<|endoftext|>", eos_token="<|endoftext|>", **kwargs):

        super(EpisodeSummaryTokenizer, self).__init__(
            vocab_file, merges_file, errors, unk_token, bos_token, eos_token, **kwargs
        )
        self.max_num_words = max_num_words

    def preprocess_text(self, text):
        """Pre-process (cut to size, tokenize) a text.

        :param text: String text
        :return: The tokenized and vectorized text (list of integers)
        """
        # if the number of words in the text is higher than the threshold,
        # try to cut the last sentences to make it shorter
        text = self._cut_text(text)

        # tokenize
        tokenized_text = self.convert_tokens_to_ids(
            self.tokenize("<|endoftext|> {} <|endoftext|>".format(text))
        )
        return tokenized_text

    @staticmethod
    def _is_sentence_end(word):
        """Decide, if a word is possibly the end a of a sentence.

        In the previous processing steps, the whitespaces are removed from before "." characters, and the text is
        split around the whitespaces, so a word is likely at the end of a sentence, if it ends with ".", "!" or "?".
        However, abbreviations form exception, since these often end in "."s.
        So we have have to filter out words that are possibly abbreviations, or just to be safe, any word that contains
        only dots and capital letters.

        :param word: A string
        :return: True or False
        """
        # common abbreviations in the English language
        abbrs = ['Dr.', 'Lt.', 'Mr.', 'Capt.', 'Cmdr.', 'Jr.', 'Ms.', 'Mrs.',
                 'Sgt.', 'Sr.', 'pt.', 'no.', 'Ltd.', 'inc.', 'Gov.' 'dept.',
                 'div.', 'est.', 'Cpl.', 'Corp.', 'Col.', 'Comdr.', 'Ave.',
                 'St.', 'Ser.', 'mt.', 'mts.', 'Assn.', 'Cdr.']

        if word.endswith('?') or word.endswith("!"):
            return True

        if word.endswith("."):
            # if it is a standalone "." character return True,
            # however this should not happen based on the previous processing steps
            if len(word) < 2:
                return True

            # filter words ending with uppercase characters and common abbreviations
            if not word[-2].isupper() and not any([word.lower() == abbr.lower() for abbr in abbrs]):
                return True

        return False

    def _cut_text(self, text):
        """Try to cut down a text to a given size.

        If the number of words in the text is shorter than the threshold, return it. If it is longer,
        try to break off some sentences, unless it is impossible without cutting in the middle of a sentence

        :param text: String text
        :return: A string summary or None
        """
        words = text.split()

        if len(words) <= self.max_num_words:
            return text

        # determine the indexes where a sentence ends
        sentence_end_idxs = [i + 1 for i in range(len(words)) if self._is_sentence_end(words[i])]

        # if sentence_end_idxs is [] it means that we were unable to identify any sentence ends,
        # in this case return the entire text, even if it is really long (this is highly unlikely)
        if not sentence_end_idxs:
            return text

        cut_idxs = [idx for idx in sentence_end_idxs if idx < self.max_num_words]

        # if we have not found a possible cut idx, cut after the first sentence
        if not cut_idxs:
            return " ".join(words[:sentence_end_idxs[0]])

        # or the first sentence is longer than our limit
        last_cut_idx = cut_idxs[-1]

        return " ".join(words[:last_cut_idx])

    def pad_batch_to_same_size(self, batch):
        """Given a batch of tokenized text (lists of integers), pad them to the same size.

        Find the length of the longest list in the batch,
        and pad all the sequences in the batch to this size by adding <|endoftext|> tokens to the end of the lists.

        :param batch: List of lists
        :return: Torch tensor created from the padded input batch.
        """
        block_size = len(max(batch, key=len))
        padded_batch = []

        for tokenized_text in batch:
            tokenized_text = tokenized_text + self.convert_tokens_to_ids(
                self.tokenize(" <|endoftext|>" * (block_size - len(tokenized_text)))
            )
            padded_batch.append(tokenized_text)

        return torch.tensor(padded_batch)


class EpisodeSummaryDataset(Dataset):
    """Episode Summary dataset."""

    def __init__(self, episode_summaries, tokenizer):
        """Initialize the EpisodeSummaryDataset object.

        :param episode_summaries: List of episode summaries (string)
        :param tokenizer: Tokenizer
        """
        self.tokenizer = tokenizer
        self.episode_summaries = episode_summaries

    @classmethod
    def create_datasets_from_jsons(cls, json_file_paths, tokenizer, val_split_ratio):
        """Parse the data from a list of JSON files, and create an EpisodeSummaryDataset object.

        :param json_file_paths: List of JSON file paths
        :param tokenizer: Tokenizer object
        :param val_split_ratio: The ratio between the size of our full dataset and the validation subset
        :return: Tuple of EpisodeSummaryDataset objects (train and val datasets)
        """
        episode_summaries = []
        for json_file_path in json_file_paths:
            with open(json_file_path, "r") as f:
                json_data = json.load(f)
            episode_summaries += [ep_data["episode_summary"] for ep_data in json_data]

        # break up episode summaries into train and val subsets
        random.shuffle(episode_summaries)
        train_ep_sums = episode_summaries[int(len(episode_summaries) * val_split_ratio):]
        val_ep_sums = episode_summaries[:int(len(episode_summaries) * val_split_ratio)]

        train_dataset = EpisodeSummaryDataset(train_ep_sums, tokenizer)
        val_dataset = EpisodeSummaryDataset(val_ep_sums, tokenizer)

        return train_dataset, val_dataset

    def __len__(self):
        return len(self.episode_summaries)

    def __getitem__(self, idx):
        return self.tokenizer.preprocess_text(self.episode_summaries[idx])
