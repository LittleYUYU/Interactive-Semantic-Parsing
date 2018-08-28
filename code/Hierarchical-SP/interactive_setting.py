# interactive_setting.py
# Setting for models

class hRL:
    penalty_global_run = 0.0
    penalty_local_ask_user = 0.3
    high_level_reward_mode = 1
    bonus_global_correct = 10.0
    penalty_local_ask_user_CI = 0.3
    low_level_reward_mode = 1
    low_level_action_mask = False #True
    bool_subtask_shared_vocab = True
    training_stage = 0
    high_level_lr = 0.001
    agent_mode = "REINFORCE"
    target_update_rate = 0.0
    batch_size = 0
    discount = 0.99
    dim = 50
    high_level_action_mask = True
    entropy_beta = 0.01
    bool_target_network = False
    high_level_lr_v = 0.0
    low_level_lr = 0.001
    low_level_lr_v = 0.0
    instruction_length = 25
    trigger_channel_user_answer_length = 10
    trigger_function_user_answer_length = 100
    action_channel_user_answer_length = 10
    action_function_user_answer_length = 100
    weight_user_answer = 0.5
    bool_sibling_z = True
    scope = "hierarchical_agent"
    bool_low_agent_attention = True
    agent_type = "hRL"


class hRL_fixedHigh:
    penalty_global_run = 0.0
    penalty_local_ask_user = 0.3
    high_level_reward_mode = 1
    bonus_global_correct = 10.0
    penalty_local_ask_user_CI = 0.3
    low_level_reward_mode = 1
    low_level_action_mask = False #True
    bool_subtask_shared_vocab = True
    training_stage = 1
    high_level_lr = 0.001
    agent_mode = "REINFORCE"
    target_update_rate = 0.0
    batch_size = 0
    discount = 0.99
    dim = 50
    high_level_action_mask = True
    entropy_beta = 0.01
    bool_target_network = False
    high_level_lr_v = 0.0
    low_level_lr = 0.001
    low_level_lr_v = 0.0
    instruction_length = 25
    trigger_channel_user_answer_length = 10
    trigger_function_user_answer_length = 100
    action_channel_user_answer_length = 10
    action_function_user_answer_length = 100
    weight_user_answer = 0.5
    bool_sibling_z = True
    scope = "hierarchical_agent_fixedHigh"
    bool_low_agent_attention = True
    agent_type = "hRL_fixedHigh"


class LAM_Sup:
    penalty_global_run = 0.0
    penalty_local_ask_user = 0.3
    high_level_reward_mode = 3
    bonus_global_correct = 10.0
    penalty_local_ask_user_CI = 0.3
    low_level_reward_mode = 1
    low_level_action_mask = False #True
    bool_subtask_shared_vocab = True
    training_stage = 1 #
    high_level_lr = 0.001
    agent_mode = "REINFORCE"
    target_update_rate = 0.0
    batch_size = 0
    discount = 0.99
    dim = 50
    high_level_action_mask = True
    entropy_beta = 0.01
    bool_target_network = False
    high_level_lr_v = 0.0
    low_level_lr = 0.001
    low_level_lr_v = 0.0
    instruction_length = 25
    trigger_channel_user_answer_length = 10
    trigger_function_user_answer_length = 100
    action_channel_user_answer_length = 10
    action_function_user_answer_length = 100
    weight_user_answer = 0.5
    bool_sibling_z = False
    scope = "baseline_agent"
    bool_low_agent_attention = True
    agent_type = "lam_human"


class LAM_Rule:
    penalty_global_run = 0.0
    penalty_local_ask_user = 0.3
    high_level_reward_mode = 3
    bonus_global_correct = 10.0
    penalty_local_ask_user_CI = 0.3
    low_level_reward_mode = 1
    low_level_action_mask = False  # True
    bool_subtask_shared_vocab = True
    training_stage = 1  #
    high_level_lr = 0.001
    agent_mode = "REINFORCE"
    target_update_rate = 0.0
    batch_size = 0
    discount = 0.99
    dim = 50
    high_level_action_mask = True
    entropy_beta = 0.01
    bool_target_network = False
    high_level_lr_v = 0.0
    low_level_lr = 0.001
    low_level_lr_v = 0.0
    instruction_length = 70
    # will not be used
    trigger_channel_user_answer_length = 10
    trigger_function_user_answer_length = 100
    action_channel_user_answer_length = 10
    action_function_user_answer_length = 100
    weight_user_answer = 0.5
    bool_sibling_z = False
    scope = "lam_rule_agent"
    bool_low_agent_attention = True
    agent_type = "lam_rule"


setting = {"hRL": hRL, "hRL_fixedHigh": hRL_fixedHigh, "lam_human": LAM_Sup, "lam_rule": LAM_Rule}
