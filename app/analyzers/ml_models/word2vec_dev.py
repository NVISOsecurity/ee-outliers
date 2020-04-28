from helpers.singletons import logging
import re

import copy

from collections import Counter, defaultdict
from itertools import chain

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.optimizer import Optimizer

from typing import List, Tuple, Dict, Optional


class Word2Vec:
    def __init__(self, separators: str, size_window: int) -> None:
        # Default parameters
        self.train_batch_size = 16
        self.eval_batch_size = 64
        self.embedding_size = 50
        self.epochs = 2  # TODO add early stopping option
        self.learning_rate = 0.01
        self.device = torch.device("cpu")  # TODO

        self.unknown_token = 'UNKNOWN'

        self.separators = separators
        self.size_window = size_window

        self.seed = 43  # Set in outliers.conf or let here as default? TODO
        torch.manual_seed(self.seed)

        self.voc_list: List = list()
        self.voc_size: int = 0
        self.voc_counter: Counter = Counter()  # Count the occurrence of the vocabulary
        self.idx2word: Dict[int, str] = dict()
        self.word2idx: Dict[str, int] = dict()

        self.model: Optional[nn.Module] = None

    def stat_model(self, train_data: List[str]) -> List[Tuple[int, int, int, float]]:
        center_context_text_idx_list = self._data_preprocessing(data=train_data)
        center_to_context_count: Dict[int, Dict[int, float]] = dict()
        center_total_count = dict()
        for center_idx, context_idx, text_idx in center_context_text_idx_list:
            if center_idx not in center_to_context_count.keys():
                center_to_context_count[center_idx] = dict()
                center_total_count[center_idx] = 1
            if context_idx not in center_to_context_count[center_idx].keys():
                center_to_context_count[center_idx][context_idx] = 1
            center_to_context_count[center_idx][context_idx] += 1
            center_total_count[center_idx] += 1
        center_context_prob = copy.deepcopy(center_to_context_count)
        for center_idx, context_count in center_context_prob.items():
            for context_idx, count in context_count.items():
                center_context_prob[center_idx][context_idx] = count/center_total_count[center_idx]

        center_context_text_prob_list: List[Tuple[int, int, int, float]] = list()
        for center_idx, context_idx, text_idx in center_context_text_idx_list:
            prob = center_context_prob[center_idx][context_idx]
            center_context_text_prob_list.append((center_idx, context_idx, text_idx, prob))
        return center_context_text_prob_list

    def train_model(self, train_data: List[str]) -> List[float]:
        """
        TODO
        :param train_data: List of string representing the phrases that will train the model
        :return total_loss_values: List of loss of each training steps
        """
        center_context_text_idx_list = self._data_preprocessing(data=train_data)

        train_dataset = Word2VecDataset(center_context_text_idx_list)

        train_data_loader = DataLoader(dataset=train_dataset,
                                       batch_size=self.train_batch_size,
                                       shuffle=True)

        if self.model is None:
            self.model = Word2VecModel(self.embedding_size, self.voc_size).to(self.device)

        adam_optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        total_loss_values = list()
        # TODO early stopping
        for i in range(self.epochs):
            logging.logger.debug("Train epoch " + str(i))
            loss_values = train_loop(train_data_loader, self.model, adam_optimizer, self.device)
            total_loss_values.extend(loss_values)
        return total_loss_values

    def eval_model(self, eval_data: List[str]) -> List[Tuple[int, int, int, float]]:
        """

        :param eval_data:
        :return:
        """
        center_context_text_idx_list = self._data_preprocessing(data=eval_data)
        # Create dataset
        eval_dataset = Word2VecDataset(center_context_text_idx_list)

        eval_data_loader = torch.utils.data.DataLoader(eval_dataset,
                                                       batch_size=self.eval_batch_size,
                                                       shuffle=False)
        center_context_text_prob_list = eval_loop(eval_data_loader, self.model, self.device)

        return center_context_text_prob_list

    def update_vocabulary_counter(self, data: List[str]) -> None:
        """

        :param data:
        """
        data_tokenized = self._tokenizer(data)
        self.voc_counter.update(chain.from_iterable(data_tokenized))

    # TODO add min_voc_occurrence in configuration of outlier.conf
    def prepare_voc(self, max_voc_size: int = 6000, min_voc_occurrence: int = 1) -> None:
        """

        :param max_voc_size:
        :param min_voc_occurrence:
        """
        tmp_voc_dict = dict(self.voc_counter)
        if min_voc_occurrence > 1:
            tmp_voc_dict = {k: v for k, v in tmp_voc_dict.items() if v >= min_voc_occurrence}
        tmp_voc_list = list(tmp_voc_dict)
        tmp_voc_size = len(tmp_voc_list)
        if tmp_voc_size > max_voc_size:
            tmp_voc_size = max_voc_size

        self.voc_list = tmp_voc_list[:tmp_voc_size] + [self.unknown_token]
        self.voc_size = tmp_voc_size + 1
        self.word2idx = {w: idx for (idx, w) in enumerate(self.voc_list)}
        self.idx2word = {idx: w for (idx, w) in enumerate(self.voc_list)}

    # def prepare_thresholds(self):
    #     center_context_text_idx_list = list()
    #     for center_idx in self.idx2word.keys():
    #         center_context_text_idx_list.append((center_idx, 0, 0))
    #
    #     dataset = Word2VecDataset(center_context_text_idx_list)
    #     data_loader = torch.utils.data.DataLoader(dataset,
    #                                               batch_size=self.eval_batch_size,
    #                                               shuffle=False)
    #     center_context_text_prob_list = eval_loop(data_loader, self.model, self.device)
    #
    #     thresholds = list()
    #
    #     # Loop from batch to batch
    #     for i, (center, context, text, probs) in enumerate(center_context_text_prob_list):
    #         # Loop within the batch
    #         for j in range(len(center)):
    #             center_idx = center[j].item()
    #             context_tensor = probs[j]
    #             mean_context = context_tensor.mean()
    #             median_context = context_tensor.median()
    #             std_probs = context_tensor.std()
    #             threshold = mean_context + std_probs
    #             thresholds.append(mean_context)
    #
    #     return thresholds

    def _data_preprocessing(self, data: List[str]) -> List[Tuple[int, int, int]]:
        data_tokenized = self._tokenizer(data)
        center_context_text_idx_list = self._texts_to_center_context_text_idx(data_tokenized)
        return center_context_text_idx_list

    def _tokenizer(self, data: List[str]) -> List[List[str]]:
        tokens = [re.split(self.separators, x) for x in data]
        return tokens

    def _texts_to_center_context_text_idx(self, tokenized_texts: List[List[str]]) -> List[Tuple[int, int, int]]:
        center_context_text_idx_list = list()
        for idx_phrase, phrase in enumerate(tokenized_texts):
            for i, center in enumerate(phrase):
                center_idx = self._get_token_idx(center)
                first_context_word_index = max(0, i - self.size_window)
                last_context_word_index = min(i + self.size_window + 1, len(phrase))
                for j in range(first_context_word_index, last_context_word_index):
                    if i != j:
                        context_idx = self._get_token_idx(phrase[j])
                        center_context_text_idx_list.append((center_idx, context_idx, idx_phrase))
        return center_context_text_idx_list

    def _get_token_idx(self, token: str):
        if token in self.word2idx:
            return self.word2idx[token]
        else:
            return self.word2idx[self.unknown_token]


class Word2VecModel(nn.Module):
    def __init__(self, embedding_size, vocab_size):
        super(Word2VecModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)

    def forward(self, center_idx=None):
        emb = self.embeddings(center_idx)
        out = self.linear(emb)
        return out


class Word2VecDataset(Dataset):
    def __init__(self, center_context_text_idx_list):
        self.center_context_text_idx_list = center_context_text_idx_list

    def __len__(self):
        return len(self.center_context_text_idx_list)

    def __getitem__(self, item):
        center_idx, context_idx, text_idx = self.center_context_text_idx_list[item]
        return {"center_idx": center_idx,
                "context_idx": context_idx,
                "text_idx": text_idx}


def loss_fn(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)


def train_loop(data_loader: DataLoader, model: nn.Module, optimizer: Optimizer, device: torch.device) -> List[float]:
    """

    :param data_loader:
    :param model:
    :param optimizer:
    :param device:
    :return:
    """
    loss_values = list()
    model.train()
    for bi, d in enumerate(data_loader):
        center_idx = d["center_idx"]
        context_idx = d["context_idx"]

        center_idx = center_idx.to(device, dtype=torch.long)
        context_idx = context_idx.to(device, dtype=torch.long)

        optimizer.zero_grad()

        outputs = model(center_idx=center_idx)
        loss = loss_fn(outputs, context_idx)
        loss_values.append(loss.item())

        loss.backward()

        optimizer.step()

    return loss_values


def eval_loop(data_loader: DataLoader, model: nn.Module, device: torch.device) -> List[Tuple[int, int, int, float]]:
    """
    Evaluation loop
    :param data_loader:
    :param model:
    :param device:
    :return:
    """
    total_center_idx = list()
    total_context_idx = list()
    total_text_idx = list()
    total_prob = list()
    model.eval()

    for bi, d in enumerate(data_loader):
        center_idx = d["center_idx"]
        context_idx = d["context_idx"]
        text_idx = d["text_idx"]

        center_idx = center_idx.to(device, dtype=torch.long)
        context_idx = context_idx.to(device, dtype=torch.long)

        outputs = model(center_idx=center_idx)
        all_prob = nn.Softmax(dim=1)(outputs)  # shape: [batch_size, vocab_size]

        prob = all_prob[torch.arange(all_prob.size(0)), context_idx]  # shape: [batch_size]

        total_center_idx.extend(center_idx.tolist())
        total_context_idx.extend(context_idx.tolist())
        total_text_idx.extend(text_idx.tolist())
        total_prob.extend(prob.tolist())

    return list(zip(total_center_idx, total_context_idx, total_text_idx, total_prob))
