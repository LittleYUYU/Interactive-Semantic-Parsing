import numpy as np
import pickle
import random
import copy

import sys
reload(sys)
sys.setdefaultencoding('utf8')

from utils import *
import config

import pdb


class State(object):
    """
    A state consists of: initial instruction, current summarization vectors and user answers for each component.
    """
    def __init__(self, instruction, ground_truth, ground_truth_names):
        """
        Initialization.
        :param instruction: a list of ids, the initial natural language instruction.
        :param ground_truth: a list of numbers, the ground truth for trigger channel, trigger function,
            action channel and action function, respectively.
        :param ground_truth_names: a list of strings, the names of label.
        """
        self.reset(instruction, ground_truth, ground_truth_names)

    def update_evaluation(self):
        for subtask_index in range(4):
            self._evaluate_subtask(subtask_index)
        self._evaluate_state()

    def _evaluate_subtask(self, index):
        """ Return 1 if the subtask has been correctly predicted, and 0 otherwhise."""
        assert index < 4
        if self._cur_preds[index] is None or self._cur_preds[index] != self._ground_truth[index]:
            self._cur_corrects[index] = 0
            return 0
        else:
            self._cur_corrects[index] = 1
            return 1

    def _evaluate_state(self):
        """ Return and update overall accuracy of the state. """
        acc = sum(self._cur_corrects) * 1.0 / 4
        self._cur_acc = acc
        return acc

    def reset(self, instruction, ground_truth, ground_truth_names):
        """ Reset state."""
        self._instruction = instruction
        self._ground_truth = ground_truth
        self._ground_truth_names = ground_truth_names
        self._cur_vecs = [None] * 4
        self._cur_preds = [None] * 4
        self._user_answers = [[] for _ in range(4)]
        self._cur_corrects = [0] * 4
        self._cur_acc = 0.0
        self._cur_subtask = None # None: hasn't started; 0 ~ 3: the subtask the agent just worked on.
        self._cur_subtask_terminal = False
        self._accumu_reward = 0.0
        self._high_level_available_actions = [0, 1, 2, 3]
        self._low_level_available_actions = [None] * 4

    def print_state_readable(self, id2word, labeler_encoders, unk):
        print("*" * 10)
        print("State info:\nInstruction: %s\nGround_truth: %s" % (
            id2token_str(self._instruction, id2word, unk), str(label2name(self._ground_truth, labeler_encoders))))
        print("User answers: ")
        for idx in range(4):
            print("Subtask %d: %s" % (
                idx, id2token_str(self._user_answers[idx], id2word, unk)
                if self._user_answers[idx] is not None else "None"))
        print("Current predictions: %s" % str(label2name(self._cur_preds, labeler_encoders)))
        print("Current accumulate reward: %.3f" % self._accumu_reward)
        print("Current subtask: %s" % str(self._cur_subtask))
        print("Current subtask terminal: %d" % self._cur_subtask_terminal)
        print("*" * 10)

    def print_state(self):
        """ Print state info. """
        print("*" * 10)
        print("State info:\nInstruction: %s\nGround_truth: %s" % (
            self._instruction, str(self._ground_truth)))
        print("z vectors: ")
        for idx in range(4):
            print("z_%d (first 10 elements): %s" % (idx, str(self._cur_vecs[idx][:10])
            if self._cur_vecs[idx] is not None else "None"))
        print("User answers: ")
        for idx in range(4):
            print("Subtask %d: %s" % (
                idx, str(self._user_answers[idx])
                if self._user_answers[idx] is not None else "None"))
        print("Current predictions: %s" % str(self._cur_preds))
        print("Current accumulate reward: %.3f" % self._accumu_reward)
        print("Current subtask: %s" % str(self._cur_subtask))
        print("Current subtask terminal: %d" % self._cur_subtask_terminal)
        print("Current cross_subtask_constrants: %s" % str(self._low_level_available_actions))
        print("*" * 10)

    def get_cur_subtask(self):
        return self._cur_subtask

    def get_cur_subtask_terminal(self):
        return self._cur_subtask_terminal

    def get_instruction(self):
        return self._instruction

    def get_ground_truth(self):
        return self._ground_truth

    def get_ground_truth_names(self):
        return self._ground_truth_names

    def get_user_answer(self, subtask_index):
        assert subtask_index < 4
        return self._user_answers[subtask_index]

    def get_cur_acc(self):
        return self._cur_acc

    def get_cur_correctness(self, subtask_index):
        assert subtask_index < 4
        return self._cur_corrects[subtask_index]

    def get_cur_all_correctness(self):
        return self._cur_corrects

    def get_cur_vecs(self):
        return self._cur_vecs

    def get_cur_preds(self):
        return self._cur_preds

    def update_cur_preds(self, subtask_index, pred):
        assert subtask_index < 4
        self._cur_preds[subtask_index] = pred

    def update_cur_vecs(self, subtask_index, z):
        assert subtask_index < 4
        self._cur_vecs[subtask_index] = z

    def update_user_answers(self, index, new_ans):
        """
        Update with new user answer for the component specified by the index.
        :param index: index of the four components, between 0 ~ 3.
        :param new_ans: a list of ids, the received user answer.
        :return: None
        """
        assert index < 4
        self._user_answers[index].extend(new_ans)

    def bool_task_completed(self):
        """ Check whether the task is completed."""
        if self._cur_preds.count(None) == 0:
            return True
        else:
            return False

    def bool_subtask_completed(self, index):
        """Check whether the sub-task is completed."""
        assert index < 4
        if self._cur_preds[index] is not None:
            return True
        else:
            return False

    def get_accumu_reward(self):
        return self._accumu_reward

    def append_accumu_reward(self, reward):
        self._accumu_reward += reward

    def get_high_level_available_actions(self):
        return self._high_level_available_actions

    def update_high_level_available_actions(self):
        self._high_level_available_actions = [idx for idx, pred in enumerate(self._cur_preds) if pred is None]

    def get_low_level_available_actions(self, subtask_index):
        return self._low_level_available_actions[subtask_index]

    def update_low_level_available_actions(self, chnl_fn_constraints, subtask_idx, bool_correct_binding=False):
        """

        :param chnl_fn_constraints:
        :param subtask_index: The subtask that is completed.
        :param bool_correct_binding:
        :return:
        """
        bindings = {0:1, 1:0, 2:3, 3:2}
        binding_subtask_idx = bindings[subtask_idx]
        if self._cur_preds[binding_subtask_idx] is None:
            # bool_correct_binding: adding constraints only when the binding subtask is predicted correctly
            if (bool_correct_binding and self._cur_corrects[subtask_idx] == 1) or not bool_correct_binding:
                available_actions = list(chnl_fn_constraints[subtask_idx][self._cur_preds[subtask_idx]])
                if len(available_actions) > 1:
                    available_actions.append(config.ASK_USER_INDICES[binding_subtask_idx])
                self._low_level_available_actions[binding_subtask_idx] = available_actions


class UserSimulator(object):
    def __init__(self, answer_pool):
        self._answer_pool = answer_pool # a list of four dicts {channel/function: answer lists}.
        self._user_answer = None

    def get_current_user_answer(self):
        return self._user_answer

    def give_user_answer(self, state, subtask_index):
        if subtask_index == 0:
            item = state.get_ground_truth_names()[0]
        elif subtask_index == 1:
            item = "%s.%s" % (state.get_ground_truth_names()[0].lower().strip(),
                              state.get_ground_truth_names()[1].lower().strip())
        elif subtask_index == 2:
            item = state.get_ground_truth_names()[2]
        else:
            item = "%s.%s" % (state.get_ground_truth_names()[2].lower().strip(),
                              state.get_ground_truth_names()[3].lower().strip())

        if item in self._answer_pool[subtask_index]:
            self._user_answer = random.choice(self._answer_pool[subtask_index][item])
            return self._user_answer
        else:
            pdb.set_trace()
            raise Exception("Invalid function!")


class RealUser(object):
    def __init__(self, word2id, label_encoders, answer_pool, trigger_fn2desc, action_fn2desc):
        self._word2id = word2id
        self._id2word = {v:k for k,v in word2id.items()}
        self._unk_id = 12621
        self._label_encoders = label_encoders
        self._answer_pool = answer_pool
        self._trigger_fn2desc = trigger_fn2desc
        self._action_fn2desc = action_fn2desc
        self._user_answer = None

        self._tc_question = "\nQ: Which CHANNEL (e.g., gmail, facebook) should TRIGGER the action?" #What's the TRIGGER CHANNEL?
        self._tf_question = "\nQ: Could you describe more about the TRIGGER FUNCTION (i.e., the specific event that triggers the action)?"
        self._ac_question = "\nQ: Which CHANNEL (e.g., gmail, facebook) should ACT per your request?"#What's the ACTION CHANNEL?
        self._af_question = "\nQ: Could you describe more about the ACTION FUNCTION (i.e., the specific event that results from the trigger)?"
        self._questions = [self._tc_question, self._tf_question, self._ac_question, self._af_question]

    def print_question(self, question):
        return bcolors.PINK + bcolors.BOLD + question + bcolors.ENDC

    def print_candidate(self, candidate):
        return bcolors.BOLD + candidate + bcolors.ENDC

    def get_current_user_answer(self):
        return self._user_answer

    def give_user_answer(self, _, subtask_index):
        user_ans = raw_input(self.print_question(self._questions[subtask_index] + "\nA: "))
        self._user_answer = user_ans

        return token2id_list(user_ans.strip(), self._word2id, self._unk_id)
