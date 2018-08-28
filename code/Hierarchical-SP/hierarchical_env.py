import copy
import numpy as np
from utils import *

from environment import State, UserSimulator, RealUser

import pdb


class HierState(State):
    def __init__(self, instruction, ground_truth, ground_truth_names):
        super(HierState, self).__init__(instruction, ground_truth, ground_truth_names)

    def update_subtask_state(self, subtask_index, low_level_agent):
        fed_state = [self._instruction, len(self._instruction),
                     self._user_answers[subtask_index], len(self._user_answers[subtask_index])]
        fed_state.extend(self._cur_vecs[0:subtask_index])
        fed_state.extend(self._cur_vecs[subtask_index+1:])
        new_zi = low_level_agent.get_state_vector(read_batch_state(fed_state,
                                                  low_level_agent.config.instruction_length,
                                                  low_level_agent.config.user_answer_length,
                                                  "low_level", bool_batch=False))[0]
        self.update_cur_vecs(subtask_index, new_zi)

    def update_initial_state(self, low_agents):
        """ This function is to create the initial four z_i.
        When updating one subtask, z_i from others are zero. """
        for subtask_index in range(4):
            fed_state = [self._instruction, len(self._instruction), [], 0]
            fed_state.extend([np.zeros(low_agents[subtask_index].config.state_vector_dim, dtype=np.float32)] * 3)
            batched_fed_state = read_batch_state(fed_state, low_agents[subtask_index].config.instruction_length,
                                 low_agents[subtask_index].config.user_answer_length,
                                 "low_level", bool_batch=False)
            new_zi = low_agents[subtask_index].get_state_vector(batched_fed_state)[0]
            self.update_cur_vecs(subtask_index, new_zi)


class HierIFTTTEnvironment(object):
    def __init__(self, penalty_global_run, penalty_local_ask_user, max_global_run, max_local_run,
                 answer_pool, ask_user_indices, low_level_agents, chnl_fn_constraints=None,
                 source4real_user=None, bool_sibling_z=True, agent_type=None):
        """
        Set up the environment for the hierarchical MDP.
        :param penalty_global_run: a negative floating number, the penalty for global rounds.
        :param penalty_local_ask_user: a negative floating number, the penalty for asking users.
        :param max_global_run: the maximum number of global runs.
        :param max_local_run: the maximum number of local runs.
        :param answer_pool: a list of four pools, each is a dict of {item: a list of descriptions}.
        :param ask_user_indices: a dict of {subtask index: action index standing for asking users}.
        :param low_level_agents: a list of low-level agents.
        :param chnl_fn_constraints: a list of four dicts: trigger/action chnl2fns, trigger/action fn2chnls.
        :param source4real_user: used to activate realUser mode.
        :param bool_sibling_z: set to False to avoid updating z_{-i}.
        :param agent_type: a string in {"lam_rule", "lam_human", "hRL_fixedHigh", "hRL"}.
        """
        self._penalty_global_run = penalty_global_run
        self._penalty_local_ask_user = penalty_local_ask_user
        self._max_global_run = max_global_run
        self._max_local_run = max_local_run
        self._ask_user_indices = ask_user_indices
        self._global_run_count = 0  # counting the global runs
        self._local_run_count = [0] * 4  # counting the local runs
        self._ask_user_count = [0] * 4 # counting the asking user times
        self._passed_local_run_count = [0] * 4 # recording the passed local runs, in case some subtasks repeat
        self._select_ans_count = [0] * 4

        self._user_simulator = UserSimulator(answer_pool)
        if source4real_user:
            # print("\n%sReal user testing%s" % ("*"*20, "*"*20))
            token2id, label_encoders, trigger_fn2desc, action_fn2desc = source4real_user
            self._user_simulator = RealUser(token2id, label_encoders, answer_pool, trigger_fn2desc, action_fn2desc)

        self._state = HierState(None, None, None) # state
        self._last_high_level_state = None
        self._low_level_agents = low_level_agents

        self._chnl_fn_constraints = chnl_fn_constraints
        self._bool_sibling_z = bool_sibling_z

        self._agent_type = agent_type

    def get_global_run_count(self):
        return self._global_run_count

    def get_local_run_count(self):
        return self._local_run_count

    def get_all_local_run_count(self):
        return self._passed_local_run_count

    def get_ask_user_count(self):
        return self._ask_user_count

    def get_select_ans_count(self):
        return self._select_ans_count

    def get_state(self):
        return self._state

    def get_user_simulator(self):
        return self._user_simulator

    def reset(self, instruction, ground_truth, ground_truth_names,
              pseudo_ask_labels=None, user_study_answers=None, agent_type=None):
        """ Reset the environment with clean state. Must run before starting a new episode. """
        self._state.reset(instruction, ground_truth, ground_truth_names)
        # initialize state vectors
        if self._bool_sibling_z:
            self._state.update_initial_state(self._low_level_agents)
        else:
            for subtask_idx in range(4):
                self._state.update_cur_vecs(subtask_idx, np.zeros(
                    self._low_level_agents[subtask_idx].config.state_vector_dim))

        self._global_run_count = 0
        self._local_run_count = [0] * 4
        self._passed_local_run_count = [0] * 4
        self._ask_user_count = [0] * 4
        self._select_ans_count = [0] * 4

        self._last_high_level_state = copy.deepcopy(self._state)
        self._agent_type = agent_type

    def print_env(self, bool_readable=False, id2word=None, labeler_encoders=None, unk=None):
        """ Print the environment. """
        print("Environment info:")
        if bool_readable:
            assert id2word is not None and labeler_encoders is not None and unk is not None
            self._state.print_state_readable(id2word, labeler_encoders, unk)
        else:
            self._state.print_state()
        print("Others: global_run_count=%d, local_run_count=%s, passed_local_run_count=%s, "
              "ask_user_count=%s, user_select_ans_count=%s" % (
            self._global_run_count, str(self._local_run_count),
            str(self._passed_local_run_count), str(self._ask_user_count),
            str(self._select_ans_count)))

    def step(self, actions, zi, subtask_index, bool_new_global_run, bool_verbose=False,
             bool_correct_binding=False):
        """
        Performing one step and changing the state.
        :param actions: a list of top-ranked options.
        :param zi: the updated zi for the subtask.
        :param subtask_index: the index of the subtask.
        :param bool_new_global_run: set to True if the action takes place in a new global run.
        :param bool_verbose: set to True for printing info.
        :param bool_correct_binding: set to True for adding cross-subtask constraints
                ONLY when the binding subtask is predicted correctly.
        :return:
        """
        prev_low_level_state = copy.deepcopy(self._state)

        if bool_new_global_run: # reset global record
            self._local_run_count = [0] * 4
            self._last_high_level_state = copy.deepcopy(self._state)
            self._global_run_count += 1
            self._local_run_count[subtask_index] = 1 # start a new count for this subtask work
            self._passed_local_run_count[subtask_index] += 1 # record overall local runs
            self._state._cur_subtask = subtask_index
            self._state._cur_subtask_terminal = False
        else:
            self._local_run_count[subtask_index] += 1
            self._passed_local_run_count[subtask_index] += 1

        prev_high_level_state = self._last_high_level_state

        bool_ask_user = False
        ans = None

        if bool_verbose: print("Subtask %d" % subtask_index)

        action = actions[0]

        if self._agent_type == "lam_rule":
            (action, prob) = action
            if prob < config.LAM_RULE_THRESHOLD and self._ask_user_count[subtask_index] < config.MAX_LAM_RULE_ASK:
                action = self._ask_user_indices[subtask_index]

        if action == self._ask_user_indices[subtask_index]:
            bool_ask_user = True
            self._ask_user_count[subtask_index] += 1

            top3_actions = actions[1:4]
            if len(top3_actions) == 0:
                ans = self._user_simulator.give_user_answer(self._state, subtask_index)
                self._state.update_user_answers(subtask_index, ans)
                if bool_verbose:
                    print("Asking users...")
                    print("User answer: %s" % str(ans))

            else:
                raise Exception("Invalid action list!")

        if action != self._ask_user_indices[subtask_index]:  # not asking users: updating prediction
            if bool_verbose: print("Taking the action %d..." % action)
            self._state.update_cur_preds(subtask_index, action)
            self._state.update_evaluation()
            self._state.update_high_level_available_actions()
            if self._chnl_fn_constraints:
                self._state.update_low_level_available_actions(
                    self._chnl_fn_constraints, subtask_index, bool_correct_binding)

        if self._bool_sibling_z: self._state.update_cur_vecs(subtask_index, zi)
        new_state = self._state

        # check and update task completion
        bool_terminal_overall = self._state.bool_task_completed() or self._global_run_count == self._max_global_run
        self._state._cur_subtask_terminal = not bool_ask_user or\
                                            self._local_run_count[subtask_index] == self._max_local_run
        if self._state.get_cur_subtask_terminal() and bool_ask_user:
            # update state vector for the last received user answer
            self._state.update_subtask_state(subtask_index, self._low_level_agents[subtask_index])

        if not self._state.get_cur_subtask_terminal():
            bool_terminal_overall = False

        # internal reward: NOTE: the penalty is a positive number.
        penalty = self._penalty_local_ask_user
        internal_reward = 0.0
        if bool_ask_user:
            internal_reward = (-1) * penalty
        else:
            new_correctness = new_state.get_cur_correctness(subtask_index)
            internal_reward += (1.0 if new_correctness == 1 else -1.0)

        # append accumulate reward
        new_state.append_accumu_reward(internal_reward)

        # external reward is given only when a subtask terminates
        external_reward_mode2reward = {}
        if not self._state.get_cur_subtask_terminal():
            # external_reward = 0.0
            pass
        else:
            prev_accumu_reward = prev_high_level_state.get_accumu_reward()
            new_accumu_reward = new_state.get_accumu_reward()
            increment_accumu_reward = new_accumu_reward - prev_accumu_reward
            # external_reward_mode2reward[1] = increment_accumu_reward
            reward1 = increment_accumu_reward
            if prev_high_level_state.get_cur_preds()[subtask_index] is not None: # repeated attempt
                if prev_high_level_state.get_cur_correctness(subtask_index) == 0: # prev: wrong
                    reward1 += 1.0
                else:
                    reward1 -= 1.0
            external_reward_mode2reward[1] = reward1

        if bool_verbose:
            print("Stepping internal reward: %.3f" % (internal_reward))
            for mode in sorted(external_reward_mode2reward.keys()):
                print("Stepping external rewards for mode %d: %.3f." % (
                    mode, external_reward_mode2reward[mode]))
        return prev_low_level_state, prev_high_level_state, new_state, \
               (external_reward_mode2reward, internal_reward, internal_reward), \
               bool_terminal_overall, (bool_ask_user, ans)



