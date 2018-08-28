#utils.py

import tensorflow as tf
import numpy as np
from nltk import word_tokenize
import config

# color
# Usage: print bcolors.WARNING + "Warning: No active frommets remain. Continue?" + bcolors.ENDC
class bcolors:
    PINK = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def discounted_sum(num_list, discount):
    sum = 0.0
    for value in num_list[::-1]:
        sum = sum * discount + value
    return sum


def ground_truth_names2labels(label_names, labelers):
    labels = []
    for name, labeler in zip(label_names, labelers):
        labels.append(labeler.transform([name])[0])
    return labels


def id2token_str(id_list, id2token, unk):
    # id_list = np.minimum(id_list, unk)
    # return " ".join([id2token[_id] if _id != unk else "UNK" for _id in id_list])
    return " ".join([id2token[_id] for _id in id_list if _id in id2token])


def token2id_list(token_str, token2id, unk):
    tokens = word_tokenize(token_str)
    # return [token2id[token.lower()] for token in tokens if token.lower() in token2id]
    ids = [min(token2id.get(token.lower(), unk), unk) for token in tokens]
    return [_ for _ in ids if _ != unk]
    # to show the real unk
    # return np.minimum(ids, unk)


def label2name(label_list, label_encoders):
    names = []
    for subtask_idx, encoder in enumerate(label_encoders):
        if label_list[subtask_idx] is not None:
            names.append(encoder.inverse_transform([label_list[subtask_idx]])[0])
        else:
            names.append("None")
    return names


def bool_valid_args(received_args, FLAGS):
    """ Check whether the arguments are valid. """
    for arg in received_args:
        if arg[0] == "-":
            if arg[1] != "-":
                print("Invalid arg: missing bar.")
                return False
            if "=" in arg:
                real_arg = arg[2:arg.index("=")]
            else:
                real_arg = arg[2:]
            if real_arg not in FLAGS.__flags:
                print("Invalid arg: %s" % real_arg)
                return False

    return True


# def clipped_error(x):
#     return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)


def make_array(seqs, length=None):
    '''Make a 2D NumPy array from a list of strings or a list of 1D arrays/lists.
    Shape of result is len(seqs) x length of longest sequence.'''
    # padding
    if length is None:
        length = max(len(elem) for elem in seqs)
    array = np.full((len(seqs), length), 0, dtype=np.int32)

    for i, item in enumerate(seqs):
        if len(item) > length: # clipping
            item = item[:(length + 1)//2] + item[len(item) - length//2:]
        array[i, :len(item)] = item

    return array


def read_batch_state(states, instruction_length, user_answer_length, agent_level, bool_batch=True):
    """ Rearrange the states into a list of [bs, sent_len] or [bs]. """
    assert agent_level in {"low_level", "high_level"}
    if not bool_batch: # this is a single instance
        states = [states] # make it to a batch

    reshaped_states = []
    for idx in range(len(states[0])):
        content = [state[idx] for state in states]
        if agent_level == "low_level":
            if idx == 0:
                content = make_array(content, instruction_length)
            elif idx == 1:
                content = np.minimum(content, instruction_length)
            elif idx == 2:
                content = make_array(content, user_answer_length)
            elif idx == 3:
                content = np.minimum(content, user_answer_length)
            else:
                content = np.array(content)
        else:
            content = np.array(content)
        reshaped_states.append(content)

    return reshaped_states



