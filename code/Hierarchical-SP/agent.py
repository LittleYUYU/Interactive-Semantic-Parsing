import numpy as np
import random
import tensorflow as tf
import sys

from utils import *
from config import MINIMUM_EPSILON

import pdb


class Agent(object):
    def __init__(self, sess, config, agent_level):

        self.config = config

        self._agent_level = agent_level # high/low_level
        self._learning_rate = config.learning_rate
        self._num_actions = config.num_actions
        self._agent_scope = tf.get_variable_scope().name
        if "lam_rule_agent" in self._agent_scope:
            self._num_actions -= 1
        self._target_update_rate = config.target_update_rate
        self._state_vector_dim = config.state_vector_dim
        self._bool_action_mask = config.bool_action_mask  # action mask

        self._bool_vbaseline = (config.agent_mode == "REINFORCE-Vbaseline")
        if self._bool_vbaseline:
            self._v_learning_rate = config.v_learning_rate or self._learning_rate
        self._entropy_beta = config.entropy_beta
        self._bool_target_network = config.bool_target_network
        self._max_pi_time_step = config.max_pi_time_step
        self._max_v_time_step = config.max_v_time_step

        self._current_time_step = 0 # record the number of samples

        # construct the agent
        self._construct_graph()
        self.sess = sess

    def get_pi_learning_rate(self):
        return self._anneal_learning_rate(self._learning_rate, self._max_pi_time_step)

    def get_v_learning_rate(self):
        return self._anneal_learning_rate(self._v_learning_rate, self._max_v_time_step)

    def _anneal_learning_rate(self, init_lr, max_step):
        if max_step == 0:
            return init_lr
        else:
            return init_lr * max(max_step - self._current_time_step, 0) / max_step

    def _state_def(self):
        self._state = []
        raise NotImplementedError("Override me!")

    def _state_vector_network(self, state):
        raise NotImplementedError("Override me!")

    def _network_design(self):
        def _network(state):
            # state vector
            with tf.variable_scope("state_vector"):
                state_vector = self._state_vector_network(state)

            # policy network
            with tf.variable_scope("pi_network"):
                w_pi = tf.get_variable("weight", shape=[self._state_vector_dim, self._num_actions],
                                      initializer=tf.contrib.layers.xavier_initializer())
                if "lam_rule_agent" in tf.get_variable_scope().name:
                    pi_values = tf.nn.softmax(tf.matmul(state_vector, w_pi))
                else:
                    b_pi = tf.get_variable("bias", shape=[self._num_actions], initializer=tf.zeros_initializer())
                    pi_values = tf.nn.softmax(tf.matmul(state_vector, w_pi) + b_pi)

            # value network
            if self._bool_vbaseline:
                with tf.variable_scope("v_network"):
                    w_v = tf.get_variable("weight", shape=[self._state_vector_dim, 1],
                                          initializer=tf.contrib.layers.xavier_initializer())
                    b_v = tf.get_variable("bias", shape=[1], initializer=tf.zeros_initializer())
                    v_values = tf.matmul(state_vector, w_v) + b_v

                return state_vector, pi_values, tf.squeeze(v_values, axis=1) # [bs]
            else:
                return state_vector, pi_values, None

        with tf.variable_scope('network'):
            self._state_vector, self._pi_values, self._v_values = _network(self._state)
        if self._bool_target_network:
            with tf.variable_scope('target_network'):
                self._target_state_vector, self._target_pi_values, \
                self._target_v_values = _network(self._state)

    def _prepare_loss(self):
        # taken action (input for policy)
        self._picked_actions = tf.placeholder(shape=[None, 2], dtype=tf.int32)

        # temporary difference (R-V) (input for policy)
        self._td_targets = tf.placeholder(shape=[None], dtype=tf.float32)

        # avoid NaN with clipping when value in pi becomes zero
        log_pi = tf.log(tf.clip_by_value(self._pi_values, 1e-20, 1.0))

        # policy entropy
        entropy = -tf.reduce_sum(self._pi_values * log_pi, reduction_indices=1)

        # policy loss (output)  (Adding minus, because the original paper's objective function is for gradient ascent,
        # but we use gradient descent optimizer.)
        self.policy_loss = - tf.reduce_sum(
            tf.gather_nd(log_pi, self._picked_actions) * self._td_targets +
            entropy * self._entropy_beta)

        if self._bool_vbaseline:
            # R (input for value)
            self._returns = tf.placeholder(shape=[None], dtype=tf.float32)

            # value loss (output)
            # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
            self.value_loss = tf.nn.l2_loss(self._returns - self._v_values)

    def _prepare_policy_train_op(self):
        self.pi_learning_rate_input = tf.placeholder(dtype=tf.float32)
        self.optimizer = tf.train.RMSPropOptimizer(self.pi_learning_rate_input)
        grads_and_vars = self.optimizer.compute_gradients(
            self.policy_loss, tf.trainable_variables(scope=tf.get_variable_scope().name))
        grads = [gv[0] for gv in grads_and_vars]
        params = [gv[1] for gv in grads_and_vars]
        grads = tf.clip_by_global_norm(grads, 5.0)[0]
        clipped_grads_and_vars = zip(grads, params)
        self.policy_network_train_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        # self.policy_network_train_op = self.optimizer.minimize(
        #     self.policy_loss, var_list=tf.trainable_variables(scope=tf.get_variable_scope().name))

    def _prepare_value_train_op(self):
        self.v_learning_rate_input = tf.placeholder(dtype=tf.float32)
        self.optimizer = tf.train.RMSPropOptimizer(self.v_learning_rate_input)
        grads_and_vars = self.optimizer.compute_gradients(
            self.value_loss, tf.trainable_variables(scope=tf.get_variable_scope().name))
        grads = [gv[0] for gv in grads_and_vars]
        params = [gv[1] for gv in grads_and_vars]
        grads = tf.clip_by_global_norm(grads, 5.0)[0]
        clipped_grads_and_vars = zip(grads, params)
        self.value_network_train_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        # self.value_network_train_op = self.optimizer.minimize(
        #     self.value_loss, tf.trainable_variables(scope=tf.get_variable_scope().name))

    def _construct_graph(self):
        self._state_def()
        self._network_design()
        assert isinstance(self._state, list)
        # network params
        params = [t for t in tf.trainable_variables(scope=tf.get_variable_scope().name)
                  if t.name.startswith(tf.get_variable_scope().name + '/network')]
        params = sorted(params, key=lambda v: v.name)

        # update target network
        if self._bool_target_network:
            target_net_params = [t for t in tf.trainable_variables(scope=tf.get_variable_scope().name)
                                 if t.name.startswith(tf.get_variable_scope().name + '/target_network')]
            target_net_params = sorted(target_net_params, key=lambda v: v.name)
            self.update_target_network_op = []
            for p, tg_p in zip(params, target_net_params):
                self.update_target_network_op.append(
                    tg_p.assign(p * self._target_update_rate + tg_p * (1 - self._target_update_rate)))

        # loss
        self._prepare_loss()

        # gradient
        self._prepare_policy_train_op()
        if self._bool_vbaseline:
            self._prepare_value_train_op()

    def get_state_vector(self, state):
        assert len(state) == len(self._state)
        z_i = self.sess.run(self._state_vector, {key: value for key, value in zip(self._state, state)})
        return z_i

    def get_current_time_step(self):
        return self._current_time_step

    def set_current_time_step(self, step):
        self._current_time_step = step

    def update_policy_network(self, states, picked_actions, td_targets, learning_rate_input=None):
        feed_dict = {
            self._picked_actions: picked_actions,
            self._td_targets: td_targets,
            self.pi_learning_rate_input: learning_rate_input or self.get_pi_learning_rate()}
        feed_dict.update({key: value for key, value in zip(self._state, states)})

        self.sess.run(self.policy_network_train_op, feed_dict=feed_dict)

    def update_value_network(self, states, returns, learning_rate_input=None):
        feed_dict = {
            self._returns: returns,
            self.v_learning_rate_input: learning_rate_input or self.get_v_learning_rate()}
        feed_dict.update({key: value for key, value in zip(self._state, states)})

        self.sess.run(self.value_network_train_op, feed_dict=feed_dict)

    def update_target_network(self):
        self.sess.run(self.update_target_network_op)

    def update_params(self, states, picked_actions, td_targets, returns=None,
                      pi_learning_rate_input=None, v_learning_rate_input=None,
                      bool_update_pi_network=True, bool_update_v_network=True):
        if bool_update_pi_network:
            self.update_policy_network(states, picked_actions, td_targets, pi_learning_rate_input)
        if self._bool_vbaseline and bool_update_v_network and returns is not None:
            self.update_value_network(states, returns, v_learning_rate_input)
        if self._bool_target_network:
            self.update_target_network()

    def is_action_mask(self):
        return self._bool_action_mask

    def sample(self, state, available_actions=None, bool_greedy=False, num_top=1):
        """
        Sample an action following epsilon-greedy.action
        :param state: a list of [zi, booli]
        :param available_actions: a list of actions that haven't been chosen previously.
        :param bool_greedy: set to True for greedy action taken.
        :param num_top: the number of returned top ranked actions.
        :return: a list of actions, each is an action index between 0 ~ 3.
        """
        if "lam_rule_agent" in self._agent_scope:
            assert bool_greedy

        assert isinstance(state, list)
        state = read_batch_state(state, self.config.instruction_length if self._agent_level == "low_level" else None,
                                 self.config.user_answer_length if self._agent_level == "low_level" else None,
                                 self._agent_level, bool_batch=False)

        self._current_time_step += 1

        if self._bool_vbaseline:
            state_vector, pi_values, v_values = self.sess.run(
                [self._state_vector, self._pi_values, self._v_values],
                {key: value for key, value in zip(self._state, state)})
            v_value = v_values[0]

        else:
            state_vector, pi_values = self.sess.run([self._state_vector, self._pi_values],
                                          {key:value for key, value in zip(self._state,state)})
            v_value = 0.0

        if available_actions is not None:
            tmp = [pi_values[0][idx] + MINIMUM_EPSILON for idx in available_actions]
            available_pi_values = np.array(tmp) / sum(tmp)
        else:
            available_actions = range(self._num_actions)
            available_pi_values = pi_values[0]

        if bool_greedy:
            sorted_actions = np.argsort(available_pi_values)[::-1]
            if "lam_rule_agent" in self._agent_scope:
                actions = [(available_actions[idx], available_pi_values[idx]) for idx in sorted_actions[:num_top]]
            else:
                actions = [available_actions[idx] for idx in sorted_actions[:num_top]]
        else:
            actions_indices = np.random.choice(range(len(available_actions)), size=min(num_top, len(available_actions)),
                                       p=available_pi_values, replace=False)
            if "lam_rule_agent" in self._agent_scope:
                actions = [(available_actions[idx], available_pi_values[idx]) for idx in actions_indices]
            else:
                actions = [available_actions[idx] for idx in actions_indices]

        return state_vector[0], actions, v_value
