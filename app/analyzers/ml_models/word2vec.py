import numpy as np
import tensorflow as tf
import math
import os
import json
import collections
import datetime
import time

from tensorflow.contrib.tensorboard.plugins import projector
from helpers.singletons import logging, settings

from typing import List, Tuple, Dict, Any, Union, Optional


class Word2Vec:
    def __init__(self, name: str) -> None:
        self.batch_size: int = 128
        self.embedding_size: int = 300  # Dimension of the embedding vector
        self.skip_window: int = 16  # How many context words to consider left and right of each target word
        self.num_sampled: float = self.batch_size / 2  # Number of negative examples to sample
        self.num_steps: int = settings.config.getint("machine_learning", "training_steps")
        self.save_every: float = self.num_steps / 100  # Save every N steps
        self.vocabulary_size: int

        # Debug flag
        self.use_test_data: bool = settings.config.getboolean("machine_learning", "word2vec_use_test_data")

        # Cached probabilities
        self.use_cache: str = settings.config.get("machine_learning", "word2vec_use_cache")
        self.all_probabilities_cache: Dict = dict()

        # Set up logging directory
        self.name :str = name
        self.model_name: str = self.name + "_word2vec"
        # important: need '' at end so it's treated as directory!
        self.models_dir: str = os.path.join(settings.config.get("machine_learning", "models_directory"),
                                            self.model_name, '')

        now: datetime.datetime = datetime.datetime.now()
        self.log_dir: str = os.path.join(self.models_dir, now.strftime("%Y-%m-%d %H:%M"), 'log')

        # File locations
        self.words_to_indices_filename: str = os.path.join(self.models_dir, self.model_name + "_words_to_indices.json")
        self.indices_to_words_filename: str = os.path.join(self.models_dir, self.model_name + "_indices_to_words.json")
        self.metadata_filename: str = os.path.join(self.models_dir, self.model_name + "_metadata.tsv")

        self.meta_graph_dir: str = os.path.join(self.models_dir, self.model_name)
        self.meta_graph_filename: str = os.path.join(self.models_dir, self.model_name + ".meta")

        # Create model directory if it does not exist
        if not os.path.exists(self.models_dir + "/"):
            os.makedirs(self.models_dir + "/", exist_ok=True)

    def train_model(self, init_sentences: List[Tuple]) -> None:
        sentences, words_to_indices, indices_to_words, words = flatten_and_build_indices(init_sentences)

        # Global position within sentences array
        sentence_index: int = 0
        vocabulary_size: int = len(set(words))  # Number of unique words in our vocabulary

        logging.logger.debug("number of training sentences: " + str(len(sentences)))
        logging.logger.debug("words: " + str(len(words)))
        logging.logger.debug("vocabulary size: " + str(vocabulary_size))

        graph: tf.Graph = tf.Graph()

        with graph.as_default():
            with tf.name_scope('inputs'):
                # Placeholders are structures for feeding input values
                train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
                train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

            # Ops and variables pinned to the CPU because of missing GPU implementation
            with tf.device('/cpu:0'):
                # Define embedding matrix variable
                # Variables are the parameters of the model that are being optimized
                with tf.name_scope('embeddings'):
                    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, self.embedding_size], -1.0, 1.0),
                                             name="embeddings")
                    # Take an input vector of integer indices,
                    # and “look up” these indices in the supplied embeddings tensor.
                    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

                # Construct the variables for the NCE loss
                with tf.name_scope('weights'):
                    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, self.embedding_size],
                                                                  stddev=1.0 / math.sqrt(self.embedding_size)),
                                              name="weights")
                with tf.name_scope('biases'):
                    nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name="biases")

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            with tf.name_scope('loss'):
                loss = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights=nce_weights,
                        biases=nce_biases,
                        labels=train_labels,
                        inputs=embed,
                        num_sampled=self.num_sampled,
                        num_classes=vocabulary_size))

            # Add the loss value as a scalar to summary.
            tf.summary.scalar('loss', loss)

            # Construct the SGD optimizer using a learning rate of 1.0.
            with tf.name_scope('optimizer'):
                optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True), name="norm")
            # normalized_embeddings = embeddings / norm

            # Merge all summaries.
            merged = tf.summary.merge_all()

            # Add variable initializer.
            init = tf.global_variables_initializer()

            # Add saver
            # Save only latest model
            saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)

        # BEGIN TRAINING
        logging.logger.info("training word2vec model using " + str(len(sentences)) + " samples")
        logging.init_ticker(total_steps=self.num_steps, desc=self.model_name + " - training word2vec model")

        with tf.Session(graph=graph) as session:
            # Open a writer to write summaries.
            writer = tf.summary.FileWriter(self.log_dir, session.graph)

            # We must initialize all variables before we use them.
            init.run()
            logging.logger.debug('Initialized all variables')
            logging.logger.debug(norm)

            average_loss: float = 0
            average_historical_loss: List[float] = list()

            step: int = 0
            while step < self.num_steps:
                logging.tick()

                batch_inputs, batch_labels, sentence_index = generate_batch(self.batch_size, self.skip_window,
                                                                            sentences, sentence_index)
                feed_dict: Dict = {train_inputs: batch_inputs, train_labels: batch_labels}

                # Define metadata variable.
                run_metadata = tf.RunMetadata()

                # We perform one update step by evaluating the optimizer op (including it in the list of returned
                # values for session.run())
                # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
                # Feed metadata variable to session for visualizing the graph in TensorBoard.
                _, summary, loss_val = session.run([optimizer, merged, loss], feed_dict=feed_dict,
                                                   run_metadata=run_metadata)
                average_loss += loss_val

                # Add returned summaries to writer in each step.
                writer.add_summary(summary, step)
                # Add metadata to visualize the graph for the last run.
                if step == (self.num_steps - 1):
                    writer.add_run_metadata(run_metadata, 'step%d' % step)

                if step % 1000 == 0:
                    if step > 0:
                        average_loss /= 1000
                        average_historical_loss.append(average_loss)
                    # The average loss is an estimate of the loss over the last 1000 steps.
                    logging.logger.info('average loss at step ' + str(step) + ': ' + str(average_loss))
                    average_loss = 0

                    # Check if historical loss is showing signs of improvement
                    if len(average_historical_loss) >= 10:
                        if np.std(average_historical_loss[-10:]) < 1:
                            logging.logger.info("loss seems to have stabilized, stopping training process")
                            step = self.num_steps - 1

                if step % self.save_every == 0:
                    saver.save(session, self.meta_graph_dir)

                step = step + 1

            # Save used embeddings together with the model
            with open(self.words_to_indices_filename, "w") as f:
                json.dump(words_to_indices, f)

            with open(self.indices_to_words_filename, "w") as f:
                json.dump(indices_to_words, f)

            # Write corresponding labels for the embeddings.
            with open(self.metadata_filename, 'w') as f:
                for i in range(vocabulary_size):
                    f.write(indices_to_words[i] + '\n')

            # Create a configuration for visualizing embeddings with the labels in TensorBoard.
            config = projector.ProjectorConfig()
            embedding_conf = config.embeddings.add()
            embedding_conf.tensor_name = embeddings.name
            embedding_conf.metadata_path = self.metadata_filename
            projector.visualize_embeddings(writer, config)

            writer.close()

    def is_trained(self) -> bool:
        try:
            with open(self.words_to_indices_filename, "r"):
                return True
        except FileNotFoundError:
            return False

    def evaluate_sentences(self, sentences: List) -> Optional[List[List[Union[int, float, np.float64]]]]:
        # Load mapping dicts saved while training the model
        try:
            with open(self.words_to_indices_filename, "r") as f:
                words_to_indices = json.load(f)
        except FileNotFoundError:
            logging.logger.warning(self.words_to_indices_filename + " not found, did you train the model before " +
                                   "running it?")
            return None

        try:
            with open(self.indices_to_words_filename, "r") as f:
                indices_to_words: Dict = json.load(f)
        except FileNotFoundError:
            logging.logger.warning(self.indices_to_words_filename + " not found, did you train the model before " +
                                   "running it?")
            return None

        graph = tf.Graph()
        with graph.as_default():
            saver = tf.train.import_meta_graph(self.meta_graph_filename)

        # DEBUG
        self.vocabulary_size = len(indices_to_words)

        with tf.Session(graph=graph) as sess:
            # appends the network defined in .meta file to the current graph
            saver.restore(sess, tf.train.latest_checkpoint(self.models_dir))

            embeddings = graph.get_tensor_by_name("embeddings/embeddings:0")
            norm = graph.get_tensor_by_name("norm:0")
            normalized_embeddings = embeddings / norm

            final_embeddings = normalized_embeddings.eval()

            weights = graph.get_tensor_by_name("weights/weights:0")
            biases = graph.get_tensor_by_name("biases/biases:0")

            sentences_probs: List = list()

            for sentence in sentences:
                tmp_probs: List[Union[int, float, np.float64]] = list()
                for word in sentence:
                    if word not in words_to_indices.keys():
                        logging.logger.debug("word " + word + " not known, skipping.")
                    else:
                        if self.use_cache:
                            if word in self.all_probabilities_cache.keys():
                                all_probabilities = self.all_probabilities_cache[word]
                            else:
                                all_probabilities = tf.nn.softmax(tf.nn.xw_plus_b(tf.expand_dims(
                                    final_embeddings[words_to_indices[word]], 0), tf.transpose(weights), biases)).eval()
                                self.all_probabilities_cache[word] = all_probabilities
                        else:
                            # For each word: Get the probabilities of all context words
                            all_probabilities = tf.nn.softmax(tf.nn.xw_plus_b(tf.expand_dims(
                                final_embeddings[words_to_indices[word]], 0), tf.transpose(weights), biases)).eval()

                        word_probs = all_probabilities[0]

                        for target_word in sentence:
                            if target_word not in words_to_indices.keys():
                                logging.logger.debug("word " + target_word + " not known, skipping.")
                            else:
                                target_word_prob = word_probs[words_to_indices[target_word]]
                                tmp_probs.append(target_word_prob)

                                if self.use_test_data:
                                    logging.logger.info("probability of seeing " + word + " in context of " +
                                                        target_word + " is " + str(target_word_prob))

                # In case we could not calculate any probability, due to all words being unknown, we return NaN
                if len(tmp_probs) == 0:
                    tmp_probs = np.nan
                else:
                    tmp_probs = np.mean(tmp_probs)

                if "sftp.exe" in sentence:
                    logging.logger.debug("Sentence:" + str(sentence) + str(tmp_probs))
                    time.sleep(5)
                else:
                    logging.logger.debug("Sentence:" + str(sentence) + str(tmp_probs))

                sentences_probs.append(tmp_probs)
            return sentences_probs


def print_most_matching_words(probs: Any, target_word: str, indices_to_words: Dict[str, str], top_n: int) -> None:
    top_n_indices = (np.argsort(probs)[-top_n:])
    top_n_indices = np.flip(top_n_indices, axis=0)

    logging.logger.info("-------------------------------")
    logging.logger.info("Probability of observing " + target_word + " around other words:")
    for i in top_n_indices:
        logging.logger.info(indices_to_words[str(i)] + " - " + str(probs[i]))


# Unique words must be mapped to integers and vice versa
# Function requires a flat list as input
###
# Dictionary maps words to integers
# Reversed_dictionary maps integers to words
###
def build_mappings(words: List) -> Tuple[Dict, Dict]:
    # Sort words by their count
    sort_by_count: List = []
    sort_by_count.extend(collections.Counter(words).most_common())
    dictionary: Dict = dict()
    for word, _ in sort_by_count:
        dictionary[word] = len(dictionary)
    reversed_dictionary: Dict = dict(zip(dictionary.values(), dictionary.keys()))

    return dictionary, reversed_dictionary


def flatten_and_build_indices(sentences: List[Tuple]) -> Tuple[List[List], Dict, Dict, List]:
    # Flatten list
    words: List = [item for sublist in sentences for item in sublist]
    words_to_indices, indices_to_words = build_mappings(words)

    logging.logger.info("example sentences: " + str(sentences[:3]))

    new_sentences: List[List] = [[words_to_indices[word] for word in sent] for sent in sentences]
    return new_sentences, words_to_indices, indices_to_words, words


# Get all skip gram pairs of a single sentence
def get_sentence_skipgrams_build(sentence: List, skip_window: int) -> Tuple[List, List]:
    sentence_targets: List = []
    sentence_labels: List = []
    for i in range(len(sentence)):  # i = target word
        context_indices = [j for j in range(max(0, (i - skip_window)), min(len(sentence), i + 1 + skip_window))
                           if j != i]
        for context_index in context_indices:
            sentence_targets.append(sentence[i])
            sentence_labels.append(sentence[context_index])
    return sentence_targets, sentence_labels


# The sentence_labels that this function returns are a list of lists
#    each inner list contains all the context words of the target word at the corresponding location in the target list
def get_sentence_skipgrams_restore(sentence: List, skip_window: int) -> Tuple[List, List]:
    sentence_targets: List = []
    sentence_labels: List = []
    for i in range(len(sentence)):  # i = target word
        context_indices: List = [j for j in range(max(0, (i - skip_window)), min(len(sentence), i + 1 + skip_window))
                           if j != i]
        sentence_targets.append(sentence[i])
        labels: List = []
        for context_index in context_indices:
            labels.append(sentence[context_index])
        sentence_labels.append(labels)
    return sentence_targets, sentence_labels


def generate_batch(batch_size: int, skip_window: int, sentences: List[List],
                   sentence_index: int) -> Tuple[np.ndarray, np.ndarray, int]:
    batch: List = []
    labels: List[List] = []
    while len(batch) < batch_size:
        cur_sentence: List = sentences[sentence_index]
        sentence_targets, sentence_labels = get_sentence_skipgrams_build(cur_sentence, skip_window)
        while len(batch) < batch_size and sentence_targets:
            batch.append(sentence_targets.pop(0))
            labels.append([sentence_labels.pop(0)])
        # When at the end of our sentences, start over
        sentence_index = (sentence_index + 1) % len(sentences)
    # Backtrack to avoid skipping words at the end of sentences
    sentence_index = max(0, (sentence_index - 1))
    # Model requires numpy arrays
    batch_out = np.asarray(batch, dtype=np.int32)
    labels_out = np.asarray(labels, dtype=np.int32)
    return batch_out, labels_out, sentence_index
