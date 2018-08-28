from pprint import pprint
import numpy as np
import pickle

# constant configurations
# number of classes for each low-level agent
HIGH_LEVEL_NUM_ACTIONS = 4
TRIGGER_CHANNEL_NUM_ACTIONS = 252 # + AskUser
TRIGGER_FUNCTION_NUM_ACTIONS = 877
ACTION_CHANNEL_NUM_ACTIONS = 219
ACTION_FUNCTION_NUM_ACTIONS = 459
SHARED_NUM_ACTIONS = 1644 # + AskUser
# ask user indices
ASK_USER_INDICES = {0: TRIGGER_CHANNEL_NUM_ACTIONS - 1,
                    1: TRIGGER_FUNCTION_NUM_ACTIONS - 1,
                    2: ACTION_CHANNEL_NUM_ACTIONS - 1,
                    3: ACTION_FUNCTION_NUM_ACTIONS - 1} # the "AskUser" action idx

# vocab size
OVERALL_VOCAB_SIZE = 12622
SUBTASK_SHARED_VOCAB = True
TRIGGER_CHANNEL_VOCAB_SIZE = 0
TRIGGER_FUNCTION_VOCAB_SIZE = 0
ACTION_CHANNEL_VOCAB_SIZE = 0
ACTION_FUNCTION_VOCAB_SIZE = 0
# a minimum float number in calculation
MINIMUM_EPSILON = 1e-8

# default configs
LEARNING_RATE = 0.001
ITERS_PER_VALIDATION = 500 # evaluate the training performance per STEPS_PER_VALIDATION
DISCOUNT = 0.99 # Discount factor for future rewards.
BATCH_SIZE = 32
TARGET_UPDATE_RATE = 0.000 # Update the target/delayed network by this rate.
WEIGHT_USER_ANSWER = 0.5
DIM = 50 # same as LAM
ENTROPY_BETA = 0.01
V_LEARNING_RATE = 0.000
MAX_PI_TIME_STEP = 10 * 10**7
MAX_V_TIME_STEP = 10 * 10**7
# Environment setting
MAX_GLOBAL_RUN = 4
MAX_LOCAL_RUN = 5
PENALTY_GLOBAL_RUN = 0.0
PENALTY_LOCAL_ASK_USER = 0.3
BONUS_GLOBAL_CORRECT = 10.0

# LAM_RULE
LAM_RULE_THRESHOLD = 0.85
MAX_LAM_RULE_ASK = 1

# more default FLAGS setting
AGENT_MODE = "REINFORCE"
BOOL_TARGET_NETWORK = False


class BasicAgentConfig(object):
    def __init__(self, learning_rate, num_actions, agent_mode,
                 target_update_rate, batch_size, discount, dim,
                 state_vector_dim, bool_action_mask,
                 entropy_beta, bool_target_network, v_learning_rate,
                 max_pi_time_step, max_v_time_step):
        self.learning_rate = learning_rate
        self.num_actions = num_actions
        self.agent_mode = agent_mode
        self.target_update_rate = target_update_rate
        self.batch_size = batch_size
        self.discount = discount
        self.dim = dim
        self.bool_action_mask = bool_action_mask
        self.state_vector_dim = state_vector_dim
        self.entropy_beta = entropy_beta
        self.bool_target_network = bool_target_network
        self.v_learning_rate = v_learning_rate
        self.max_pi_time_step = max_pi_time_step
        self.max_v_time_step=max_v_time_step

    def __str__(self):
        return str(self.__dict__)


class LowLevelAgentConfig(BasicAgentConfig):
    def __init__(self, learning_rate, num_actions, agent_mode,
                 target_update_rate, batch_size, discount,
                 instruction_length, user_answer_length,
                 dim, bool_action_mask, weight_user_answer, vocab_size,
                 entropy_beta, bool_target_network, v_learning_rate,
                 max_pi_time_step, max_v_time_step, bool_attention):
        super(LowLevelAgentConfig, self).__init__(learning_rate, num_actions, agent_mode,
                 target_update_rate, batch_size, discount, dim, 2*dim, bool_action_mask,
                 entropy_beta, bool_target_network, v_learning_rate, max_pi_time_step, max_v_time_step)
        self.instruction_length = instruction_length
        self.user_answer_length = user_answer_length
        self.weight_user_answer = weight_user_answer
        self.vocab_size = vocab_size
        self.bool_attention = bool_attention


class HighLevelAgentConfig(BasicAgentConfig):
    def __init__(self, learning_rate, num_actions, agent_mode,
                 target_update_rate, batch_size, discount, dim, bool_action_mask,
                 entropy_beta, bool_target_network, v_learning_rate,
                 max_pi_time_step, max_v_time_step):
        super(HighLevelAgentConfig, self).__init__(learning_rate, num_actions, agent_mode,
                 target_update_rate, batch_size, discount, dim, 2*dim, bool_action_mask,
                 entropy_beta, bool_target_network, v_learning_rate, max_pi_time_step, max_v_time_step)


class HierAgentConfig(object):
    def __init__(self, FLAGS):
        self.bool_subtask_shared_vocab = FLAGS.bool_subtask_shared_vocab
        self.training_stage = FLAGS.training_stage
        self.bool_sibling_z = FLAGS.bool_sibling_z
        self.scope = FLAGS.scope

        self.high_level_agent_config = HighLevelAgentConfig(
            FLAGS.high_level_lr, HIGH_LEVEL_NUM_ACTIONS, AGENT_MODE,
            0.0, FLAGS.batch_size, FLAGS.discount, FLAGS.dim,
            FLAGS.high_level_action_mask, FLAGS.entropy_beta, False,
            0.0, MAX_PI_TIME_STEP, MAX_V_TIME_STEP
        )
        self.trigger_channel_config = LowLevelAgentConfig(
            FLAGS.low_level_lr,
            TRIGGER_CHANNEL_NUM_ACTIONS,
            AGENT_MODE,
            TARGET_UPDATE_RATE, FLAGS.batch_size, FLAGS.discount,
            FLAGS.instruction_length, FLAGS.trigger_channel_user_answer_length, FLAGS.dim,
            FLAGS.low_level_action_mask, FLAGS.weight_user_answer,
            OVERALL_VOCAB_SIZE if FLAGS.bool_subtask_shared_vocab else TRIGGER_CHANNEL_VOCAB_SIZE,
            FLAGS.entropy_beta, BOOL_TARGET_NETWORK,
            V_LEARNING_RATE, MAX_PI_TIME_STEP, MAX_V_TIME_STEP,
            FLAGS.bool_low_agent_attention
        )
        self.trigger_function_config = LowLevelAgentConfig(
            FLAGS.low_level_lr,
            TRIGGER_FUNCTION_NUM_ACTIONS,
            AGENT_MODE,
            TARGET_UPDATE_RATE, FLAGS.batch_size, FLAGS.discount,
            FLAGS.instruction_length, FLAGS.trigger_function_user_answer_length, FLAGS.dim,
            FLAGS.low_level_action_mask, FLAGS.weight_user_answer,
            OVERALL_VOCAB_SIZE if FLAGS.bool_subtask_shared_vocab else TRIGGER_FUNCTION_VOCAB_SIZE,
            FLAGS.entropy_beta, BOOL_TARGET_NETWORK,
            V_LEARNING_RATE, MAX_PI_TIME_STEP, MAX_V_TIME_STEP,
            FLAGS.bool_low_agent_attention
        )
        self.action_channel_config = LowLevelAgentConfig(
            FLAGS.low_level_lr,
            ACTION_CHANNEL_NUM_ACTIONS,
            AGENT_MODE,
            TARGET_UPDATE_RATE, FLAGS.batch_size, FLAGS.discount,
            FLAGS.instruction_length, FLAGS.action_channel_user_answer_length, FLAGS.dim,
            FLAGS.low_level_action_mask, FLAGS.weight_user_answer,
            OVERALL_VOCAB_SIZE if FLAGS.bool_subtask_shared_vocab else ACTION_CHANNEL_VOCAB_SIZE,
            FLAGS.entropy_beta, BOOL_TARGET_NETWORK,
            V_LEARNING_RATE, MAX_PI_TIME_STEP, MAX_V_TIME_STEP,
            FLAGS.bool_low_agent_attention
        )
        self.action_function_config = LowLevelAgentConfig(
            FLAGS.low_level_lr,
            ACTION_FUNCTION_NUM_ACTIONS,
            AGENT_MODE,
            TARGET_UPDATE_RATE, FLAGS.batch_size, FLAGS.discount,
            FLAGS.instruction_length, FLAGS.action_function_user_answer_length, FLAGS.dim,
            FLAGS.low_level_action_mask, FLAGS.weight_user_answer,
            OVERALL_VOCAB_SIZE if FLAGS.bool_subtask_shared_vocab else ACTION_FUNCTION_VOCAB_SIZE,
            FLAGS.entropy_beta, BOOL_TARGET_NETWORK,
            V_LEARNING_RATE, MAX_PI_TIME_STEP, MAX_V_TIME_STEP,
            FLAGS.bool_low_agent_attention
        )
        self.low_level_agent_configs = [self.trigger_channel_config, self.trigger_function_config,
                                        self.action_channel_config, self.action_function_config]

        # # print info
        # print("Hierarchical Agent Config:\nHigh level agent: %s" % str(self.high_level_agent_config))
        # for idx, cfg in enumerate(self.low_level_agent_configs):
        #     print("Low level agent %d: %s" % (idx, str(cfg)))
        # print("")
