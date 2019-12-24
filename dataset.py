import random
import torch
import json
from torch.utils.data import Dataset
from pytorch_transformers import GPT2Tokenizer


class EpisodeSummaryTokenizer(GPT2Tokenizer):
    """
    Tokenizer class for episode summary strings/text.

    The class inherits most of its functionality from the GPT2Tokenizer of pytorch_transformers.
    Except for the class attributes max_num_words and size_variance_handling. These are used to limit sequence sizes,
    and determine how to handle oversized text sequences:

    - max_num_words: The maximum number of words allowed in our episode summaries
    - size_variance_handling: A string, that describes how we want to handle texts with variable length.
        Options: - 'chop_at_sentence_end': Chop long texts to make sure that they contain <= words than max_num_words,
                                           but only chop at the end of a sentence.
                                           If that is not possible, return None instead of vectorizing the text.
                 - 'chop': Chop long texts to make sure that they contain <= words than max_num_words.
                           It is okay to chop after any word.
                 - 'ignore': Ignore size variance and tokenize all text without chopping.
    """

    def __init__(self, vocab_file, merges_file, max_num_words, size_variance_handling, errors='replace',
                 unk_token="<|endoftext|>", bos_token="<|endoftext|>", eos_token="<|endoftext|>", **kwargs):

        super(EpisodeSummaryTokenizer, self).__init__(
            vocab_file, merges_file, errors, unk_token, bos_token, eos_token, **kwargs
        )

        self.max_num_words = max_num_words

        # select how we handle sequences with different sizes
        self._look_up_dict = {
            "chop_at_sentence_end": self._chop_text_at_sentence_end,
            "chop": self._chop_text,
            "ignore": lambda x: x
        }
        self.size_var_handling_fnc = self._look_up_dict[size_variance_handling]

    def preprocess_text(self, text):
        """
        Pre-process (chop to size + tokenize) text.

        :param text: String text
        :return: The tokenized and vectorized text (list of integers)
        """
        # here define a bunch of different ways to handle size difference
        text = self.size_var_handling_fnc(text)

        if not text:
            return None

        tokenized_text = self.convert_tokens_to_ids(self.tokenize("<|endoftext|> {} <|endoftext|>".format(text)))
        return tokenized_text

    @staticmethod
    def _is_sentence_end(word):
        """
        Decide, if a word is possibly the end a of a sentence.

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

    def _chop_text_at_sentence_end(self, text):
        """
        Try to cut down a text to a given size.

        If the number of words in the text is shorter than the threshold, return it. If it is longer,
        try to break off some sentences to make it shorter.
        If it is impossible without cutting in the middle of a sentence, return None

        :param text: String text
        :return: A string summary or None
        """
        words = text.split()

        if len(words) <= self.max_num_words:
            return text

        # determine the indexes where a sentence ends
        sentence_end_idxs = [i + 1 for i in range(len(words)) if self._is_sentence_end(words[i])]
        # if sentence_end_idxs is [] it means that we were unable to identify any sentence ends,
        # in this case return None
        if not sentence_end_idxs:
            return None

        cut_idxs = [idx for idx in sentence_end_idxs if idx < self.max_num_words]
        # if we have not found a possible cut idx, return None
        if not cut_idxs:
            return None

        # or the first sentence is longer than our limit
        last_cut_idx = cut_idxs[-1]

        return " ".join(words[:last_cut_idx])

    def _chop_text(self, text):
        """
        Chop down a text to a size.

        :param text: String text
        :return: Chopped text, containing <= words than max_num_words
        """
        words = text.split()

        if len(words) <= self.max_num_words:
            return text

        return " ".join(words[:self.max_num_words])

    def pad_batch_to_same_size(self, batch):
        """
        Given a batch of tokenized text (lists of integers), pad them to the same size.

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

    def __init__(self, episode_summaries):
        """Initialize the EpisodeSummaryDataset object.

        :param episode_summaries: List of tokenized episode summaries
        """
        self.episode_summaries = episode_summaries
        self._max_seq_size = len(max(episode_summaries, key=len))

    def __len__(self):
        return len(self.episode_summaries)

    def __getitem__(self, idx):
        return self.episode_summaries[idx]


def create_datasets_from_jsons(json_file_paths, tokenizer, val_split_ratio):
    """
    Parse the data from a list of JSON files, and create EpisodeSummaryDataset objects for train/validation.

    :param json_file_paths: List of JSON file paths
    :param tokenizer: Tokenizer object
    :param val_split_ratio: The ratio between the size of our full dataset and the validation subset
    :return: Tuple of EpisodeSummaryDataset objects (train and val datasets)
    """

    print("Creating datasets:")
    episode_summaries = []
    for json_file_path in json_file_paths:
        with open(json_file_path, "r") as f:
            json_data = json.load(f)

        for ep_data in json_data:
            episode_summaries.append(ep_data["episode_summary"])

    tokenized_summaries = []
    for ep_sum in episode_summaries:
        tokenized_summary = tokenizer.preprocess_text(ep_sum)

        if tokenized_summary:
            tokenized_summaries.append(tokenized_summary)

    print("  Dropped {}/{} episode summaries during vectorization.".format(
        len(episode_summaries) - len(tokenized_summaries), len(episode_summaries)
    ))

    # break up episode summaries into train and val subsets
    random.shuffle(tokenized_summaries)
    train_ep_sums = tokenized_summaries[int(len(tokenized_summaries) * val_split_ratio):]
    val_ep_sums = tokenized_summaries[:int(len(tokenized_summaries) * val_split_ratio)]

    train_dataset = EpisodeSummaryDataset(train_ep_sums)
    val_dataset = EpisodeSummaryDataset(val_ep_sums)
    print("  Training set size: {}\n  Validation set size: {}".format(len(train_dataset), len(val_dataset)))

    return train_dataset, val_dataset


"""
json_paths = [
    "/home/broccoli_consumer/workspace/episode_summary_generator/ep_data/star_trek_imdb_ep_sums.json",
    "/home/broccoli_consumer/workspace/episode_summary_generator/ep_data/star_trek_wiki_ep_sums.json"
]

tokenizer = EpisodeSummaryTokenizer.from_pretrained("gpt2", max_num_words=96,
                                                    size_variance_handling="chop_at_sentence_end")


train_dataset, val_dataset = create_datasets_from_jsons(json_paths, tokenizer, 0.1)

print(len(train_dataset), len(val_dataset))
print(train_dataset._max_seq_size, val_dataset._max_seq_size)

from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4, collate_fn=tokenizer.pad_batch_to_same_size)
val_dataloader = DataLoader(val_dataset, batch_size=1, collate_fn=tokenizer.pad_batch_to_same_size)

print(len(train_dataloader), len(val_dataloader))
"""
