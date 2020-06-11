from helpers.singletons import logging
import re
import math
import copy

from collections import Counter
from itertools import chain

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.optimizer import Optimizer

from typing import List, Tuple, Dict, Any


class Word2Vec:
    def __init__(self,
                 separators: str,
                 size_window: int,
                 num_epochs: int,
                 learning_rate: float,
                 embedding_size: int,
                 seed: int = 42) -> None:
        # Default parameters
        self.train_batch_size = 16
        self.eval_batch_size = 64
        self.device = torch.device("cpu")
        self.unknown_token = 'UNKNOWN'
        self.num_unknown_occurrence = 0  # number of time the word considered as unknown appear

        self.separators = separators
        self.size_window = size_window
        self.epochs = num_epochs
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        self.seed = seed

        if self.seed != 0:
            torch.manual_seed(self.seed)

        self.voc_size: int = 0
        self.voc_counter: Counter = Counter()  # Count the occurrence of the vocabulary
        self.id2word: Dict[int, str] = dict()
        self.word2id: Dict[str, int] = dict()

        self.model: Any = None

    def train_model(self, train_data: List[str]) -> List[float]:
        """
        Train the model self.model with train_data.
        It will preprocess train_data, create a DataLoader and train a Word2VecModel model on multiple epochs.

        :param train_data: List of string representing the texts that will train the model.
        :return total_loss_values: List of loss of each training step.
        """
        data_preprocessed = self._data_preprocessing(data=train_data)

        train_dataset = Word2VecDataset(data_preprocessed)

        train_data_loader = DataLoader(dataset=train_dataset,
                                       batch_size=self.train_batch_size,
                                       shuffle=True)

        if self.model is None:
            self.model = Word2VecModel(self.embedding_size, self.voc_size).to(self.device)

        adam_optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        total_loss_values = list()
        for i in range(self.epochs):
            logging.logger.debug("Train epoch " + str(i))
            loss_values = train_loop(train_data_loader, self.model, adam_optimizer, self.device)
            total_loss_values.extend(loss_values)
        return total_loss_values

    def eval_model(self, eval_data: List[str], output_raw: bool) -> List[Tuple[Any, ...]]:
        """
        Evaluate eval_data with the model self.model.
        It will preprocess eval_data, create a DataLoader and evaluate the data with the model self.model.

        :param eval_data: List of string representing the texts that will be evaluated.
        :param output_raw: True to have eval_loop that output the raw values and False to have outputs in form of
        probabilities.
        :return: List of tuple. Each element within the tuple represent respectively;
            - center word index in text.
            - center word id.
            - context word index in text.
            - context word id.
            - text index in eval_data.
            - probability/raw output of the context word id given the center word id.
        """
        data_preprocessed = self._data_preprocessing(data=eval_data)

        eval_dataset = Word2VecDataset(data_preprocessed)

        eval_data_loader = torch.utils.data.DataLoader(eval_dataset,
                                                       batch_size=self.eval_batch_size,
                                                       shuffle=False)
        eval_outputs = eval_loop(eval_data_loader, self.model, self.device, output_raw)

        return eval_outputs

    def update_vocabulary_counter(self, data: List[str]) -> None:
        """
        Tokenize the data and update the vocabulary counter.

        :param data: List of string representing texts.
        """
        data_tokenized = self._tokenizer(data)
        self.voc_counter.update(chain.from_iterable(data_tokenized))

    def prepare_voc(self, max_voc_size: int = 6000, min_voc_occurrence: int = 1) -> None:
        """
        Create self.voc_size, self.word2id, self.id2word and self.num_unknown_occurrence.
        If vocabulary is bigger than max_voc_size, it removes the excedent vocabulary that has the smallest occurrence
        If the occurrence of one word in the vocabulary is smaller than min_voc_occurrence, it remove it from
        vocabulary.

        :param max_voc_size: Max size of the vocabulary.
        :param min_voc_occurrence: Minimum time a word has to appear in the dataset.
        """
        tmp_voc_dict = dict(self.voc_counter.most_common())

        num_unknown_occurrence = 0
        for num_occur in tmp_voc_dict.values():
            if num_occur < min_voc_occurrence:
                num_unknown_occurrence += num_occur
        self.num_unknown_occurrence = num_unknown_occurrence
        if min_voc_occurrence > 1:
            tmp_voc_dict = {k: v for k, v in tmp_voc_dict.items() if v >= min_voc_occurrence}
        tmp_voc_list = list(tmp_voc_dict)
        tmp_voc_size = len(tmp_voc_list)
        if tmp_voc_size > max_voc_size:
            tmp_voc_size = max_voc_size

        voc_list = tmp_voc_list[:tmp_voc_size] + [self.unknown_token]
        self.voc_size = tmp_voc_size + 1
        self.word2id = {w: idx for (idx, w) in enumerate(voc_list)}
        self.id2word = {idx: w for (idx, w) in enumerate(voc_list)}

    def _data_preprocessing(self, data: List[str]) -> List[Tuple[int, int, int, int, int]]:
        """
        Preprocess data.
        Tokenize data then convert the tokenized data for word2vec model input.

        :param data: list of string representing texts.
        :return: List of tuple. Each element within the tuple represent respectively;
            - center word index in text.
            - center word id in vocabulary.
            - context word index in text.
            - context word id in vocabulary.
            - text index in eval_data.
        """
        data_tokenized = self._tokenizer(data)
        model_inputs = self._tokenized_texts_to_model_inputs(data_tokenized)
        return model_inputs

    def _tokenizer(self, data: List[str]) -> List[List[str]]:
        """
        Tokenize the data by separating words in texts by the self.separators value.

        :param data: List of string representing texts.
        :return: List of list of string representing lists of words in a list of texts.
        """
        if self.separators == "":
            tokens = [list(x) for x in data]
        else:
            tokens = [re.split(self.separators, x) for x in data]
        return tokens

    def _tokenized_texts_to_model_inputs(self, tokenized_texts: List[List[str]]) \
            -> List[Tuple[int, int, int, int, int]]:
        """
        Convert tokenized text to word2vec model input format.

        :param tokenized_texts: List of list of string representing lists of words in a list of texts.
        :return: List of tuple. Each element within the tuple represent respectively;
            - center word index in text.
            - center word id in vocabulary.
            - context word index in text.
            - context word id in vocabulary.
            - text index in eval_data.
        """
        model_inputs = list()
        for text_idx, text in enumerate(tokenized_texts):
            for center_idx, center_word in enumerate(text):
                center_id = self._get_word_id(center_word)
                first_context_word_index = max(0, center_idx - self.size_window)
                last_context_word_index = min(center_idx + self.size_window + 1, len(text))
                for context_idx in range(first_context_word_index, last_context_word_index):
                    if center_idx != context_idx:
                        context_word = text[context_idx]
                        context_id = self._get_word_id(context_word)
                        model_inputs.append((center_idx, center_id, context_idx, context_id, text_idx))
        return model_inputs

    def _get_word_id(self, word: str):
        """
        Get word id from word.

        :param word: string representing a word.
        :return: vocabulary id corresponding to the word.
        """
        if word in self.word2id:
            return self.word2id[word]
        else:
            return self.word2id[self.unknown_token]

    def prob_model(self, train_data: List[str], output_log_prob: bool) -> List[Tuple[int, int, int, int, int, float]]:
        """
        Compute the true probability of each context word context_idx to appear given the center word center_idx.
        P(context_id|center_id)=(number of time context_id appears with center_id)/(number of time center_id appears)
        Return same output format than self.eval_model.

        :param train_data: List of string representing the phrases that will train the model
        :param output_log_prob: If true, convert probabilities to log of probabilities. Used for doing arithmetic mean
        instead of geometric mean.
        :return: List of tuple. Each element within the tuple represent respectively;
            - center word index in text.
            - center word id.
            - context word index in text.
            - context word id.
            - text index in eval_data.
            - probability/raw output of the context word id given the center word id.
        """
        model_inputs = self._data_preprocessing(data=train_data)

        # total time center_id appears with context_id
        center_id_to_context_id_to_count: Dict[int, Dict[int, float]] = dict()

        # total time center_id appears
        center_to_total_count = dict()

        # Computes total time center_id appears with context_id and total time center_id appears
        for _, center_id, _, context_id, _ in model_inputs:
            if center_id not in center_id_to_context_id_to_count:
                center_id_to_context_id_to_count[center_id] = dict()
                center_to_total_count[center_id] = 1
            if context_id not in center_id_to_context_id_to_count[center_id]:
                center_id_to_context_id_to_count[center_id][context_id] = 1
            center_id_to_context_id_to_count[center_id][context_id] += 1
            center_to_total_count[center_id] += 1

        # Computes probability context_id appear given center_id
        center_id_to_context_id_to_prob = copy.deepcopy(center_id_to_context_id_to_count)
        for center_id, context_count in center_id_to_context_id_to_count.items():
            for context_id, count in context_count.items():
                center_id_to_context_id_to_prob[center_id][context_id] = count/center_to_total_count[center_id]

        # Create and return eval_outputs
        eval_outputs: List[Tuple[int, int, int, int, int, float]] = list()
        for center_idx, center_id, context_idx, context_id, text_idx in model_inputs:
            prob = center_id_to_context_id_to_prob[center_id][context_id]
            if not output_log_prob:
                prob = math.log(prob)
            eval_outputs.append((center_idx, center_id, context_idx, context_id, text_idx, prob))
        return eval_outputs


class Word2VecModel(nn.Module):
    """Custom Word2Vec pytorch model."""

    def __init__(self, embedding_size: int, vocab_size: int):
        """
        Instantiates one nn.Embedding module and one nn.Linear modules and assign them as member variables.

        :param embedding_size: Size of the embedding vectors.
        :param vocab_size: Size of the vocabulary.
        """
        super(Word2VecModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)

    def forward(self, center_id=None):
        """
        Defines the computation.

        :param center_id: Tensor of size [batch_size] representing center_id.
        :return: raw tensor vector of size [batch_size][vocab_size] representing the context_id.
        """
        emb = self.embeddings(center_id)
        out = self.linear(emb)
        return out


class Word2VecDataset(Dataset):
    """Custom Word2Vec pytorch Dataset."""

    def __init__(self, model_inputs):
        self.model_inputs = model_inputs

    def __len__(self):
        return len(self.model_inputs)

    def __getitem__(self, item):
        center_idx, center_id, context_idx, context_id, text_idx = self.model_inputs[item]
        return {"center_idx": center_idx,
                "center_id": center_id,
                "context_idx": context_idx,
                "context_id": context_id,
                "text_idx": text_idx}


def loss_fn(outputs, targets):
    """Custom pytorch loss function."""

    return nn.CrossEntropyLoss()(outputs, targets)


def train_loop(data_loader: DataLoader, model: nn.Module, optimizer: Optimizer, device: torch.device) -> List[float]:
    """
    Train loop.
    Loop in model over input batches, compute loss and do back-propagation.

    :param data_loader: Pytorch DataLoader containing word2vec model inputs.
    :param model: Word2Vec pytorch model.
    :param optimizer: Pytorch Optimizer.
    :param device:
    :return: List of loss of each training step.
    """
    loss_values = list()
    model.train()
    for bi, d in enumerate(data_loader):
        center_id = d["center_id"]
        context_id = d["context_id"]

        center_id = center_id.to(device, dtype=torch.long)
        context_id = context_id.to(device, dtype=torch.long)

        optimizer.zero_grad()

        outputs = model(center_id=center_id)

        loss = loss_fn(outputs, context_id)
        loss_values.append(loss.item())

        loss.backward()

        optimizer.step()

    return loss_values


def eval_loop(data_loader: DataLoader,
              model: nn.Module,
              device: torch.device,
              output_raw: bool) -> List[Tuple[Any, ...]]:
    """
    Evaluation loop.
    Loop in model over batches.

    :param data_loader: Pytorch DataLoader containing word2vec model inputs.
    :param model: Word2Vec pytorch model.
    :param device:
    :param output_raw: True to have eval_loop that output the raw values and False to have outputs in form of
    probabilities.
    :return: List of tuple. Each element within the tuple represent respectively;
        - center word index in text.
        - center word id.
        - context word index in text.
        - context word id.
        - text index in eval_data.
        - probability/raw output of the context word id given the center word id.
    """
    list_center_idx = list()
    list_center_id = list()
    list_context_idx = list()
    list_context_id = list()
    list_text_idx = list()
    list_output_val = list()
    model.eval()

    for bi, d in enumerate(data_loader):
        center_idx = d["center_idx"]
        center_id = d["center_id"]
        context_idx = d["context_idx"]
        context_id = d["context_id"]
        text_idx = d["text_idx"]

        center_id = center_id.to(device, dtype=torch.long)
        context_id = context_id.to(device, dtype=torch.long)

        outputs = model(center_id=center_id)
        if not output_raw:
            outputs = nn.Softmax(dim=1)(outputs)  # shape: [batch_size, vocab_size]
        output_context = outputs[torch.arange(outputs.size(0)), context_id]  # shape: [batch_size]

        list_center_idx.extend(center_idx.tolist())
        list_center_id.extend(center_id.tolist())

        list_context_idx.extend(context_idx.tolist())
        list_context_id.extend(context_id.tolist())

        list_text_idx.extend(text_idx.tolist())

        list_output_val.extend(output_context.tolist())

    return list(zip(list_center_idx, list_center_id, list_context_idx, list_context_id, list_text_idx, list_output_val))
