# lam_rule agent

import sys
import config as config_const
from utils import *

from agent import Agent


class LamRuleAgent(Agent):

    def __init__(self, config, sess, subtask_index, embedding=None):
        self._subtask_index = subtask_index
        self._embedding = embedding or tf.get_variable("embedding", shape=[config.vocab_size, config.dim])
        self._bool_attention = config.bool_attention

        super(LamRuleAgent, self).__init__(sess, config, "low_level")

        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                  scope=tf.get_variable_scope().name))

    def _state_def(self):
        instruction = tf.placeholder(shape=[None, self.config.instruction_length], dtype=tf.int32)
        instruction_actual_length = tf.placeholder(shape=[None], dtype=tf.int32)
        self._state = [instruction, instruction_actual_length]

    def _state_vector_network(self, state):
        """ Learn a vector representation for the current subtask state. """
        instruction, instruction_actual_length = state

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

        with tf.variable_scope("latent_attention_model"):
            o_i = _latent_attention_model(instruction, instruction_actual_length)

        return o_i


class LamRuleAssembleAgent(object):
    def __init__(self, config, sess):
        self.config = config

        with tf.variable_scope(self.config.scope):
            # shared word embedding
            embedding = None
            if self.config.bool_subtask_shared_vocab:
                with sess.as_default():
                    embedding = tf.get_variable("shared_embedding", shape=[config_const.OVERALL_VOCAB_SIZE,
                                                                       self.config.trigger_channel_config.dim])
            with tf.variable_scope("trigger_channel_agent"):
                tr_chnl_agent = LamRuleAgent(self.config.trigger_channel_config, sess, 0, embedding)

            with tf.variable_scope("trigger_function_agent"):
                tr_fn_agent = LamRuleAgent(self.config.trigger_function_config, sess, 1, embedding)

            with tf.variable_scope("action_channel_agent"):
                act_chnl_agent = LamRuleAgent(self.config.action_channel_config, sess, 2, embedding)

            with tf.variable_scope("action_function_agent"):
                act_fn_agent = LamRuleAgent(self.config.action_function_config, sess, 3, embedding)

            self.low_agents = [tr_chnl_agent, tr_fn_agent, act_chnl_agent, act_fn_agent]

        # self.params = tf.trainable_variables(scope=self.config.scope)
        _safe_scope = self.config.scope + ("/" if not self.config.scope.endswith("/") else "")
        self.params = tf.trainable_variables(scope=_safe_scope)
        self.saver = tf.train.Saver(var_list=self.params, max_to_keep=5)
        # print("Trainable params: ")
        # for params_i in self.params:
        #     print(params_i)
        # print("")

        sys.stdout.flush()

    def get_low_level_agent_state(self, env_state, subtask_idx):
        """
        Get the state for low-level agents.
        :param env_state: a HierState instance.
        :param subtask_idx: the subtask index.
        :return: a list.
        """
        assert subtask_idx < 4
        updt_instruction = env_state.get_instruction() + env_state.get_user_answer(subtask_idx)
        actual_len = len(updt_instruction)
        low_state = [updt_instruction, actual_len]

        return low_state

    def sample(self, env, bool_greedy=True, num_top=1, max_ask=0):
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
            next_subtask_idx = min(available_actions)
        else:
            next_subtask_idx = env_state.get_cur_subtask()
        high_v_value = 0.0

        # sample an action for this subtask
        low_agent = self.low_agents[next_subtask_idx]
        # low_state: instruction, instruction_actual_len
        low_state = self.get_low_level_agent_state(env_state, next_subtask_idx)
        available_actions = None
        if self.config.low_level_agent_configs[next_subtask_idx].bool_action_mask:
            available_actions = env_state.get_low_level_available_actions(next_subtask_idx)
        zi, next_actions, low_v_value = low_agent.sample(low_state, available_actions, bool_greedy, num_top)

        return next_actions, zi, next_subtask_idx, high_v_value, low_v_value