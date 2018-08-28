import tensorflow as tf
import random
import numpy as np
import sys
import config as config_const
from utils import *

from agent import Agent

import pdb


class LowLevelAgent(Agent):

    def __init__(self, config, sess, subtask_index, embedding=None):
        self._subtask_index = subtask_index
        self._embedding = embedding or tf.get_variable("embedding", shape=[config.vocab_size, config.dim])
        self._bool_attention = config.bool_attention

        super(LowLevelAgent, self).__init__(sess, config, "low_level")

        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                      scope=tf.get_variable_scope().name))

    def _state_def(self):
        instruction = tf.placeholder(shape=[None, self.config.instruction_length], dtype=tf.int32)
        instruction_actual_length = tf.placeholder(shape=[None], dtype=tf.int32)
        user_answer = tf.placeholder(shape=[None, self.config.user_answer_length], dtype=tf.int32)
        user_answer_actual_length = tf.placeholder(shape=[None], dtype=tf.int32)
        others_z1 = tf.placeholder(shape=[None, self._state_vector_dim], dtype=tf.float32)
        others_z2 = tf.placeholder(shape=[None, self._state_vector_dim], dtype=tf.float32)
        others_z3 = tf.placeholder(shape=[None, self._state_vector_dim], dtype=tf.float32)
        self._state = [instruction, instruction_actual_length, user_answer, user_answer_actual_length,
                       others_z1, others_z2, others_z3]

    def _state_vector_network(self, state):
        """ Learn a vector representation for the current subtask state. """
        instruction, instruction_actual_length, user_answer, user_answer_actual_length,\
        others_z1, others_z2, others_z3 = state

        def _latent_attention_model(instruction, instruction_actual_length):
            """ Latent Attention Model. Please refer to:
            https://papers.nips.cc/paper/6284-latent-attention-for-if-then-program-synthesis.pdf
            Args:
                instruction: the initial instruction, a 2d tensor of shape [bs, instruction_length].
                instruction_actual_length: a 1d tensor of shape [bs].
            Returns:
                oi: the output representation of the instruction, a 2d tensor of shape [bs, 2*dim].
            """
            instruction = tf.minimum(self.config.vocab_size - 1, instruction)
            embedded_inputs = tf.nn.embedding_lookup(self._embedding, instruction)  # [bs, instr_len, dim]

            # BDLSTM
            with tf.variable_scope("BDLSTM_cell", initializer=tf.contrib.layers.xavier_initializer()):
                bdlstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.dim)

            bdlstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(bdlstm_cell, bdlstm_cell, embedded_inputs,
                                                                sequence_length=instruction_actual_length,
                                                                time_major=False, dtype=tf.float32)
            bdlstm_outputs = tf.concat(list(bdlstm_outputs), 2)  # [bs, instr_len, z_dim]

            vec_size = self._state_vector_dim
            memory_size = self.config.instruction_length

            # Latent Attention Model
            # copy-paste from the author's github
            PREP = tf.get_variable("PREP", [1, vec_size])
            TA = tf.get_variable("TA", [memory_size, vec_size])
            m = bdlstm_outputs + TA

            # attention layer
            B = tf.get_variable("B", [vec_size, memory_size])
            m_t = tf.reshape(m, [-1, vec_size])
            d_t = tf.matmul(m_t, B)
            d_softmax = tf.nn.softmax(d_t)  # active attention
            d = tf.reshape(d_softmax, [-1, memory_size, memory_size])
            dotted_prep = tf.reduce_sum(bdlstm_outputs * PREP, 2)  # latent attention
            probs_prep = tf.nn.softmax(dotted_prep)  # [bs, memory_size].
            probs_prep_temp = tf.expand_dims(probs_prep, -1)

            # attention
            probs_temp = tf.matmul(d, probs_prep_temp)
            probs = tf.squeeze(probs_temp, axis=[2])
            output_probs = tf.nn.l2_normalize(probs, 1)

            # output representation
            probs_temp = tf.expand_dims(output_probs, 1)
            c_temp = tf.transpose(m, [0, 2, 1])
            oi = tf.reduce_sum(c_temp * probs_temp, 2)  # vec_size = 2 * dim

            return oi

        def _answer_understanding(user_ans, user_answer_actual_length, bool_attention=True):
            """ Module for user answer understanding.
            Args:
                user_ans: the user answers, a 2d tensor of shape [bs, user_answer_length].
                user_answer_actual_length: actual length of the user answer.
            Returns:
                od: answer representation, a 2d tensor of shape [bs, 2*dim].
                bool_empty: a 1d tensor of shape [bs]. """

            # empty
            bool_empty = tf.cast(tf.equal(user_answer_actual_length, 0), tf.float32)

            # embedding
            user_ans = tf.minimum(self.config.vocab_size - 1, user_ans)
            embedded_inputs = tf.stack([tf.nn.embedding_lookup(self._embedding, word)
                                        for word in tf.unstack(user_ans, self.config.user_answer_length, axis=1)],
                                       axis=0)

            # RNN
            with tf.variable_scope("rnn_cell", initializer=tf.contrib.layers.xavier_initializer()):
                rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.dim)

            outputs, last_states = tf.nn.bidirectional_dynamic_rnn(rnn_cell, rnn_cell, embedded_inputs,
                                                         sequence_length=user_answer_actual_length,
                                                         time_major=True, dtype=tf.float32)

            vec_size = self._state_vector_dim
            if bool_attention:
                h_ts = tf.unstack(tf.concat(list(outputs), 2), self.config.user_answer_length, axis=0)

                # attention
                v = tf.get_variable("v", shape=[vec_size, 1])
                unscaled_weights = tf.exp(tf.concat([tf.matmul(h_t, v) for h_t in h_ts], 1))  # [bs, user_answer_length]
                mask = tf.sequence_mask(user_answer_actual_length, self.config.user_answer_length, dtype=tf.float32)
                unscaled_weights = unscaled_weights * mask

                sum_up = tf.reshape(
                    tf.reduce_sum(unscaled_weights, 1) + bool_empty * config_const.MINIMUM_EPSILON,
                    shape=[-1, 1])
                attention_weights = unscaled_weights / sum_up  # in case zero denominator

                trans_h_ts = tf.stack(h_ts, axis=1)  # [bs, user_answer_length, vec_size]
                od = tf.reduce_sum(tf.expand_dims(attention_weights, -1) * trans_h_ts, axis=1)  # [bs, vec_size]
            else:
                v = tf.get_variable("v", shape=[vec_size, 1])
                last_forward_states = last_states[0].h
                last_backward_states = last_states[1].h
                od = tf.concat([last_forward_states, last_backward_states], axis=1)

            return od, bool_empty

        with tf.variable_scope("latent_attention_model"):
            o_i = _latent_attention_model(instruction, instruction_actual_length)
        with tf.variable_scope("answer_understanding"):
            o_d, bool_empty = _answer_understanding(user_answer, user_answer_actual_length, self._bool_attention)
        o = (1 - self.config.weight_user_answer) * o_i + self.config.weight_user_answer * o_d
        # scale for those who do not receive user answers
        scale = bool_empty * (- self.config.weight_user_answer) + 1  # [1, ..., 1, 1-w_d, 1, ..., 1]
        o = o / tf.reshape(scale, shape=[-1, 1])

        # concatenation
        others_z = [others_z1, others_z2, others_z3]
        z_summary = others_z[0:self._subtask_index] + [o] + others_z[self._subtask_index:]
        concatenation = tf.concat(z_summary, axis=1)  # [bs, 2*z_dim]

        # MLP
        with tf.variable_scope("MLP"):
            vec_size = self._state_vector_dim
            w_z = tf.get_variable("weight", shape=[vec_size * 4, vec_size],
                                  initializer=tf.contrib.layers.xavier_initializer())
            b_z = tf.get_variable("bias", shape=[vec_size], initializer=tf.zeros_initializer())
            z_i = tf.nn.tanh(tf.matmul(concatenation, w_z) + b_z)  # [bs, z_dim]

        return z_i


class HighLevelAgent(Agent):
    def __init__(self, config, sess):
        super(HighLevelAgent, self).__init__(sess, config, "high_level")

        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                      scope=tf.get_variable_scope().name))

    def _state_def(self):
        self._state = []
        for _ in range(4):  # z_1~z_4 and 4 booleans
            self._state.append(tf.placeholder(shape=[None, self._state_vector_dim], dtype=tf.float32))
            self._state.append(tf.placeholder(shape=[None], dtype=tf.float32))

    def _state_vector_network(self, state):
        concatenation = tf.concat([tf.expand_dims(item, -1)
                                   if len(item.get_shape()) == 1 else item for item in state], axis=1)

        with tf.variable_scope("MLP"):
            w_c = tf.get_variable("weight", shape=[4 * self._state_vector_dim + 4, self._state_vector_dim],
                                  initializer=tf.contrib.layers.xavier_initializer())
            b_c = tf.get_variable("bias", shape=[self._state_vector_dim], initializer=tf.zeros_initializer())
            c = tf.nn.tanh(tf.matmul(concatenation, w_c) + b_c)

        return c


class HierarchicalAgent(object):
    def __init__(self, config, sess):
        self.config = config

        with tf.variable_scope(self.config.scope):
            # define agents
            with tf.variable_scope("high_level_agent"):
                self.high_agent = HighLevelAgent(self.config.high_level_agent_config, sess)

            # shared word embedding
            embedding = None
            if self.config.bool_subtask_shared_vocab:
                with sess.as_default():
                    embedding = tf.get_variable("shared_embedding", shape=[config_const.OVERALL_VOCAB_SIZE,
                                                                       self.config.trigger_channel_config.dim])
            with tf.variable_scope("trigger_channel_agent"):
                tr_chnl_agent = LowLevelAgent(self.config.trigger_channel_config, sess, 0, embedding)

            with tf.variable_scope("trigger_function_agent"):
                tr_fn_agent = LowLevelAgent(self.config.trigger_function_config, sess, 1, embedding)

            with tf.variable_scope("action_channel_agent"):
                act_chnl_agent = LowLevelAgent(self.config.action_channel_config, sess, 2, embedding)

            with tf.variable_scope("action_function_agent"):
                act_fn_agent = LowLevelAgent(self.config.action_function_config, sess, 3, embedding)

            self.low_agents = [tr_chnl_agent, tr_fn_agent, act_chnl_agent, act_fn_agent]

        # pretraining with cross ent loss
        var_collections = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=self.config.scope + "/shared_embedding")
        param_section_names = ["trigger_channel_agent", "trigger_function_agent",
                               "action_channel_agent", "action_function_agent"]
        for _scope in param_section_names:
            var_collections.extend(
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                  scope=self.config.scope + "/" + _scope + "/network/state_vector"))
            var_collections.extend(
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                  scope=self.config.scope + "/" +_scope + "/network/pi_network"))
        self.pretrain_saver = tf.train.Saver(var_collections)
        _safe_scope = self.config.scope + ("/" if not self.config.scope.endswith("/") else "")
        self.params = tf.trainable_variables(scope=_safe_scope)
        self.saver = tf.train.Saver(var_list=self.params,max_to_keep=10)

        # print("Trainable params: ")
        # for params_i in self.params:
        #     print(params_i)
        # print("")
        sys.stdout.flush()

    def get_high_level_agent_state(self, env_state):
        """
        Get the state for high-level agent.
        :param env_state: a HierState instance.
        :return: a list.
        """
        cur_vecs = env_state.get_cur_vecs()  # a list of 4 vectors
        cur_bool_preds = [env_state.bool_subtask_completed(idx) for idx in range(4)]

        high_state = list()
        for subtask_idx in range(4):
            high_state.append(cur_vecs[subtask_idx])
            high_state.append(float(cur_bool_preds[subtask_idx]))
        return high_state

    def get_low_level_agent_state(self, env_state, subtask_idx):
        """
        Get the state for low-level agents.
        :param env_state: a HierState instance.
        :param subtask_idx: the subtask index.
        :return: a list.
        """
        assert subtask_idx < 4
        cur_vecs = env_state.get_cur_vecs()  # a list of 4 vectors
        low_state = [env_state.get_instruction(), len(env_state.get_instruction()),
                     env_state.get_user_answer(subtask_idx), len(env_state.get_user_answer(subtask_idx))]
        low_state.extend(cur_vecs[0:subtask_idx])
        low_state.extend(cur_vecs[subtask_idx + 1:])
        return low_state

    def sample(self, env, bool_greedy=False, num_top=1, max_ask=0):
        """
        Sample the next action.
        :param env: an environment instance.
        :param bool_greedy: set to True for greedy action choice.
        :param num_top: number of top ranked actions to sample.
        :return: action.
        """
        env_state = env.get_state()
        if env_state.get_cur_subtask_terminal() or\
            env_state.get_cur_subtask() is None:
            available_actions = range(4)
            if self.config.high_level_agent_config.bool_action_mask:
                available_actions = env_state.get_high_level_available_actions()

            if self.config.training_stage == 1: # fixed 0-1-2-3 order for high-level
                next_subtask_idx = min(available_actions)
                # next_subtask_idx = max(available_actions)
                high_v_value = 0.0
            else:
                high_state = self.get_high_level_agent_state(env_state)
                _, next_subtask_idx, high_v_value = self.high_agent.sample(high_state, available_actions, bool_greedy)
                next_subtask_idx = next_subtask_idx[0]
        else:
            next_subtask_idx = env_state.get_cur_subtask()
            high_v_value = 0.0

        # sample an action for this subtask
        low_agent = self.low_agents[next_subtask_idx]
        # low_state: instruction, user_answer, others_z1, others_z2, others_z3
        low_state = self.get_low_level_agent_state(env_state, next_subtask_idx)
        available_actions = range(low_agent._num_actions) #all possible actions
        if self.config.low_level_agent_configs[next_subtask_idx].bool_action_mask:
            available_actions = env_state.get_low_level_available_actions(next_subtask_idx)

        if max_ask and env.get_ask_user_count()[next_subtask_idx] == max_ask: # at most ask once
            available_actions.remove(config.ASK_USER_INDICES[next_subtask_idx])

        zi, next_actions, low_v_value = low_agent.sample(low_state, available_actions, bool_greedy, num_top)

        return next_actions, zi, next_subtask_idx, high_v_value, low_v_value

