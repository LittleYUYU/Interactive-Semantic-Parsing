"""
@author: Ziyu Yao
"""

import os, random, sys
import numpy as np
import tensorflow as tf
import pickle
import collections
import copy
import gc
import time

import config
from hierarchical_agent import HierarchicalAgent
from hierarchical_env import HierIFTTTEnvironment
from utils import *

import pdb

tf.app.flags.DEFINE_boolean("train", False, "Set to True if running to train.")
tf.app.flags.DEFINE_boolean("test", False, "Set to True if running to test.")
tf.app.flags.DEFINE_boolean("toy_data", False, "Set to True for running on toy data (debugging).")

# agent setting
tf.app.flags.DEFINE_string("agent_type", "hRL", "Support: hRL, BaselineAgent.")
tf.app.flags.DEFINE_float("high_level_lr", config.LEARNING_RATE, "Set learning rate for the high-level agent.")
tf.app.flags.DEFINE_float("low_level_lr", config.LEARNING_RATE, "Set learning rate for low-level agents.")
tf.app.flags.DEFINE_float("discount", config.DISCOUNT, "Set the discount for cumulative reward calculation.")
tf.app.flags.DEFINE_float("weight_user_answer", config.WEIGHT_USER_ANSWER, "Set the weight in 0~1 for user answers.")
tf.app.flags.DEFINE_float("entropy_beta", config.ENTROPY_BETA, "weights for entropy term in pi_loss.")
tf.app.flags.DEFINE_boolean("high_level_action_mask", True,
                            "Set to True for adding a mask when training the high-level agent.")
tf.app.flags.DEFINE_boolean("low_level_action_mask", False,
                            "Set to True for adding a mask to inconsistent selections given completed subtask.")
tf.app.flags.DEFINE_boolean("bool_sibling_z", True, "Set to False to disable sibling subtasks's affect.")
tf.app.flags.DEFINE_string("scope", "hierarchical_agent", "Set scope for the agent.")
tf.app.flags.DEFINE_boolean("bool_low_agent_attention", True,
                            "Set to False to close attention.")
# environment setting
tf.app.flags.DEFINE_float("penalty_global_run", config.PENALTY_GLOBAL_RUN, "Set the penalty for every global run.")
tf.app.flags.DEFINE_float("penalty_local_ask_user", config.PENALTY_LOCAL_ASK_USER,
                          "Set the penalty for asking users in local MDP.")
# data related
tf.app.flags.DEFINE_integer("instruction_length", 25, "Set the max instruction length.")
tf.app.flags.DEFINE_integer("trigger_channel_user_answer_length", 10,
                            "Set the max user answer length for trigger channel.")
tf.app.flags.DEFINE_integer("trigger_function_user_answer_length", 100,
                            "Set the max user answer length for trigger function.")
tf.app.flags.DEFINE_integer("action_channel_user_answer_length", 10,
                            "Set the max user answer length for action channel.")
tf.app.flags.DEFINE_integer("action_function_user_answer_length", 100,
                            "Set the max user answer length for action function.")
tf.app.flags.DEFINE_integer("dim", config.DIM, "Set vector dimension for word vectors. "
                                                   "The corresponding dimension of z vectors will be 2*dim.")
tf.app.flags.DEFINE_boolean("bool_subtask_shared_vocab", config.SUBTASK_SHARED_VOCAB,
                            "Set to False for separate word embedding for each subtask.")
tf.app.flags.DEFINE_boolean("bool_pretrained_crossent", True, "Set to True for params pretrained via cross ent.")

# others
tf.app.flags.DEFINE_integer("random_seed", 1, "Set an integer as the random seed.")
tf.app.flags.DEFINE_integer("max_iteration", -1, "Set number of instances for training.")
tf.app.flags.DEFINE_integer("batch_size", config.BATCH_SIZE, "Set the mini batch size.")
tf.app.flags.DEFINE_integer("iters_per_validation", config.ITERS_PER_VALIDATION, "Set the steps for each validation.")
tf.app.flags.DEFINE_integer("training_stage", 0,
                            "Set to 0: start or continue regular joint training."
                            "Set to 1: start or continue low_level agent pretraining. "
                            "  High_level agent will randomly(or in order) pick a subtask and will not be updated."
                            "Set to 2: need load trained params without high-level agent stats."
                            )
tf.app.flags.DEFINE_boolean("greedy_test", True, "Set to True for greedy test.")

FLAGS = tf.flags.FLAGS
print("Training stage: %d" % FLAGS.training_stage)
# check args
assert bool_valid_args(sys.argv, FLAGS)
# attribute
attr = "toy_" if FLAGS.toy_data else ""
attr += "lr{%.5f_%.5f}_penalty%.3f_wgtUser%.3f_hmask%d_pretrain%d" % (
    FLAGS.high_level_lr, FLAGS.low_level_lr,
    FLAGS.penalty_local_ask_user,
    FLAGS.weight_user_answer, FLAGS.high_level_action_mask,
    FLAGS.bool_pretrained_crossent)
if FLAGS.training_stage == 1:
    attr += "_fixedHigh"
if FLAGS.low_level_action_mask: # now, during training, the mask is applied only to correctly predicted binding subtask.
    attr += "_lmask1"
if not FLAGS.greedy_test:
    attr += "_sampleTest"
if not FLAGS.bool_low_agent_attention:
    attr += "_notLowAtt"
print("FLAGS attribute: %s" % attr)

# path setup
# data
data_path = os.path.join("../../data/lam/data_with_noisy_user_ans.pkl")
if FLAGS.toy_data:
    data_path = os.path.join("../../data/lam/toy_data_with_noisy_user_ans.pkl")
# checkpoint
checkpoint_dir = os.path.join("Log", FLAGS.agent_type, "REINFORCE", "checkpoint")
checkpoint_overall_path = os.path.join(checkpoint_dir, "ckpt_%s" % attr)
print("data path: %s\ncheckpoint path: %s\n" % (data_path, checkpoint_overall_path))

# pretrained agent
pretrain_path = os.path.join("Log", FLAGS.agent_type, "pretrain_agent", "hierarchical_agent") # for low-level agents only
baseline_agent_path = "baseline_agent"


def create_env(agent, user_answer, chnl_fn_constraints):
    env = HierIFTTTEnvironment(
        FLAGS.penalty_global_run, FLAGS.penalty_local_ask_user, config.MAX_GLOBAL_RUN,
        config.MAX_LOCAL_RUN, user_answer, config.ASK_USER_INDICES, agent.low_agents,
        chnl_fn_constraints=chnl_fn_constraints if FLAGS.low_level_action_mask else None,
        source4real_user=None, bool_sibling_z=agent.config.bool_sibling_z)

    return env


def create_agent(session, agent_type):
    assert agent_type in {"hRL", "baselineAgent"}
    config_setting = config.HierAgentConfig(FLAGS)
    if agent_type == "baselineAgent":
        config_setting.training_stage = 1
        config_setting.bool_sibling_z = False
        config_setting.scope = "baseline_agent"

    agent = HierarchicalAgent(config_setting, session)
    session.run(tf.variables_initializer(tf.global_variables(scope=config_setting.scope)))

    if agent_type == "baselineAgent":
        # load pretrained baseline agent
        print("Loading pretrained low-level agents from %s." % baseline_agent_path)
        ckpt_pretrain = tf.train.get_checkpoint_state(baseline_agent_path)
        assert ckpt_pretrain
        agent.pretrain_saver.restore(session, ckpt_pretrain.model_checkpoint_path)

        return agent, True

    # restore previous training results
    bool_found_existing = False
    # check existing params
    if not os.path.exists(checkpoint_overall_path):
        os.makedirs(checkpoint_overall_path)

    # restore the overall model
    ckpt = tf.train.get_checkpoint_state(checkpoint_overall_path)
    stats_file = os.path.join(checkpoint_overall_path, "stats.pkl")
    if os.path.exists(stats_file) and FLAGS.test:
        stats = pickle.load(open(stats_file))
        best_iter = stats["external_reward_mode2best_iter"][1]
        agent.saver.restore(session, os.path.join(checkpoint_overall_path, "ckpt-%d" % best_iter))
        print("Reading overall model parameters from %s." %
              os.path.join(checkpoint_overall_path, "ckpt-%d" % best_iter))
        bool_found_existing = True
    elif ckpt:
        agent.saver.restore(session, ckpt.model_checkpoint_path)
        print("Reading overall model parameters from %s." % ckpt.model_checkpoint_path)
        bool_found_existing = True
    else:
        assert not FLAGS.test
        print("Created overall model with fresh parameters.")

    # loading pretrained params
    if not bool_found_existing: # for fresh params
        # load params pretrained via cross ent
        if FLAGS.bool_pretrained_crossent:
            print("Loading pretrained low-level agents from %s." % pretrain_path)
            ckpt_pretrain = tf.train.get_checkpoint_state(pretrain_path)
            assert ckpt_pretrain
            agent.pretrain_saver.restore(session, ckpt_pretrain.model_checkpoint_path)

    return agent, bool_found_existing


def train():
    # load data
    print("Loading data...\n")
    data = pickle.load(open(data_path))
    train_data = data['train']
    user_answer = data['user_answers']
    chnl_fn_constraints = data.get('chnl_fn_constraints', None)

    # set the random seed
    tf.set_random_seed(FLAGS.random_seed)
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    if FLAGS.max_iteration == -1:
        random.shuffle(train_data)
        max_iteration = len(train_data)
    else:
        tmp = []
        while len(tmp) < FLAGS.max_iteration:
            random.shuffle(train_data)
            tmp.extend(train_data)
        train_data = tmp
        max_iteration = FLAGS.max_iteration
    print("Max iteration: %d" % max_iteration)

    with tf.Session() as sess:
        # set up
        print("Creating agent...")
        agent, bool_found_existing = create_agent(sess, FLAGS.agent_type)
        print("Creating environment...")
        env = create_env(agent, user_answer, chnl_fn_constraints)
        bool_baseline = False
        sys.stdout.flush()

        # training loop
        best_cf_acc = 0.0
        best_cf_acc_iter = 0
        best_averaged_internal_reward = 0.0
        best_averaged_low_level_iter = 0

        # best_external_reward = -1000.0
        # best_high_level_iter = 0
        external_reward_mode2best_reward = dict()
        external_reward_mode2best_iter = dict()

        start_iter = 0
        iter2performance = dict()
        train_iter2performance = dict()
        if bool_found_existing:
            if os.path.exists(os.path.join(checkpoint_overall_path, "stats.pkl")):
                print("Restoring stats.")
                stats = pickle.load(open(os.path.join(checkpoint_overall_path, "stats.pkl")))
                best_cf_acc = stats.get("best_cf_acc", 0.0) #stats["best_cf_acc"]
                best_cf_acc_iter = stats.get("best_cf_acc_iter", 0) #stats["best_cf_acc_iter"]
                best_averaged_internal_reward = stats.get("best_averaged_internal_reward", 0.0)
                best_averaged_low_level_iter = stats.get("best_averaged_low_level_iter", 0)
                if best_averaged_low_level_iter == 0:
                    best_averaged_internal_reward = 0.0

                external_reward_mode2best_reward = stats.get("external_reward_mode2best_reward", dict())
                external_reward_mode2best_iter = stats.get("external_reward_mode2best_iter", dict())

                start_iter = stats["iteration"] + 1
                agent_time_steps = stats["agent_current_time_steps"]
                for agent_idx in range(4): # restore low-level agent time step
                    agent.low_agents[agent_idx].set_current_time_step(agent_time_steps[agent_idx])
                if FLAGS.training_stage != 2: # at stage 2, we will train the high-level agent from scratch.
                    agent.high_agent.set_current_time_step(agent_time_steps[4])

            if os.path.exists(os.path.join(checkpoint_overall_path, "train_iter2performance.pkl")):
                print("Restoring performance.")
                train_iter2performance = pickle.load(open(os.path.join(checkpoint_overall_path, "train_iter2performance.pkl")))

            if os.path.exists(os.path.join(checkpoint_overall_path, "dev_iter2performance.pkl")):
                print("Restoring performance.")
                iter2performance = pickle.load(open(os.path.join(checkpoint_overall_path, "dev_iter2performance.pkl")))

        # records for this ITERS_PER_VALIDATION steps
        accuracy = []
        final_predictions = []
        # external_rewards = []
        external_reward_mode2reward_list = collections.defaultdict(list)
        internal_rewards_by_subtask = [[] for _ in range(4)]
        global_runs = []
        local_runs_by_subtask = [[] for _ in range(4)]
        ask_user_counts_by_subtask = [[] for _ in range(4)]
        user_select_ans_counts_by_subtask = [[] for _ in range(4)]
        correctness_by_subtask = [[] for _ in range(4)]
        high_level_orders = []

        # update record
        update_batch_by_item_by_subtask = [[[] for _ in range(4)] for _ in range(4)]
        update_batch_by_item_high_level = [[] for _ in range(4)]

        count_skipped_save = 0
        start_time = time.time()
        for iteration in range(start_iter, max_iteration):
            instruction = train_data[iteration]['ids']
            ground_truth = train_data[iteration]['labels']
            ground_truth_names = train_data[iteration]['label_names']
            env.reset(instruction, ground_truth, ground_truth_names)

            bool_verbose = (iteration % FLAGS.iters_per_validation < 3) or (iteration == max_iteration - 1)
            # bool_verbose = True
            if bool_verbose: print("Iteration %d..." % iteration)

            bool_terminal_overall = False
            bool_subtask_terminal = True
            # instance_external_reward = 0
            instance_external_reward_mode2reward = dict() # record a dict of high-level rewards
            instance_internal_reward = [0] * 4
            instance_high_level_order = []

            high_level_buffer = []
            low_level_buffer = []
            while not bool_terminal_overall:
                if bool_verbose:
                    print("%s\n%s" % ("==" * 10, "Before sampling:"))
                    env.print_env()

                next_actions, zi, next_subtask_idx, high_v_value, low_v_value = agent.sample(
                    env, num_top=1)
                prev_low_level_state, prev_high_level_state, new_state, \
                (external_reward_mode2reward, internal_reward, _), bool_terminal_overall, \
                (bool_ask_user, ans) = env.step(
                    next_actions, zi, next_subtask_idx, bool_subtask_terminal, bool_verbose=bool_verbose,
                    bool_correct_binding=True)
                if bool_verbose:
                    print("%s\n%s\n" % ("==" * 10, "After sampling and stepping:"))
                    env.print_env()

                # subtask terminal update
                bool_subtask_terminal = env.get_state().get_cur_subtask_terminal()

                baseline_reduce = 0.0
                # store the record for local MDP in buffer
                low_level_buffer.append((agent.get_low_level_agent_state(prev_low_level_state, next_subtask_idx),
                                         next_actions[0], internal_reward, low_v_value, baseline_reduce))

                if bool_subtask_terminal:
                    accumulate_internal_reward = 0.0
                    for step_idx in reversed(range(len(low_level_buffer))):
                        buffer_data = low_level_buffer[step_idx]
                        reward_reduce = 0.0
                        accumulate_internal_reward = buffer_data[2] + FLAGS.discount * accumulate_internal_reward
                        for item_idx, item in enumerate(
                                [buffer_data[0], buffer_data[1],
                                accumulate_internal_reward - reward_reduce,
                                accumulate_internal_reward]):
                            update_batch_by_item_by_subtask[next_subtask_idx][item_idx].append(item)

                    # clear buffer
                    low_level_buffer = []

                # record
                instance_internal_reward[next_subtask_idx] += internal_reward

                # when the subtask terminates, store the record for global MDP.
                if bool_subtask_terminal:
                    # record all types of high-level rewards
                    # instance_external_reward += external_reward
                    for mode, value in external_reward_mode2reward.items():
                        instance_external_reward_mode2reward[mode] = \
                            instance_external_reward_mode2reward.get(mode, 0.0) + value

                    external_reward = external_reward_mode2reward[1]
                    instance_high_level_order.append(next_subtask_idx)
                    if FLAGS.training_stage != 1:
                        # store in buffer
                        high_level_buffer.append(
                            (agent.get_high_level_agent_state(prev_high_level_state), next_subtask_idx,
                             external_reward, high_v_value))

                        if bool_terminal_overall:
                            accumulate_external_reward = 0
                            for step_idx in reversed(range(len(high_level_buffer))):
                                buffer_data = high_level_buffer[step_idx]
                                accumulate_external_reward = buffer_data[2] + FLAGS.discount * accumulate_external_reward
                                for item_idx, item in enumerate([buffer_data[0], buffer_data[1],
                                                     accumulate_external_reward - buffer_data[3],
                                                     accumulate_external_reward]):
                                    update_batch_by_item_high_level[item_idx].append(item)

                            # clear buffer
                            high_level_buffer = []

            # record
            accuracy.append(env.get_state().get_cur_acc())
            final_predictions.append(env.get_state().get_cur_preds())
            correctness = env.get_state().get_cur_all_correctness() # list of 4
            local_runs = env.get_all_local_run_count() # list of 4
            ask_user_count = env.get_ask_user_count() # list of 4
            user_select_ans_count = env.get_select_ans_count() # list of 4
            for idx in range(4):
                correctness_by_subtask[idx].append(correctness[idx])
                internal_rewards_by_subtask[idx].append(instance_internal_reward[idx])
                local_runs_by_subtask[idx].append(local_runs[idx])
                ask_user_counts_by_subtask[idx].append(ask_user_count[idx])
                user_select_ans_counts_by_subtask[idx].append(user_select_ans_count[idx])

            # external_rewards.append(instance_external_reward)
            for mode, value in instance_external_reward_mode2reward.items():
                external_reward_mode2reward_list[mode].append(value)
            global_runs.append(env.get_global_run_count())
            high_level_orders.append(instance_high_level_order)

            if iteration > 0 and iteration % 64 == 0: # update low-level agents
                for next_subtask_idx, update_batch_by_item in enumerate(update_batch_by_item_by_subtask):
                    # process batch data
                    states = read_batch_state(update_batch_by_item[0], FLAGS.instruction_length,
                                            agent.low_agents[next_subtask_idx].config.user_answer_length,
                                            "low_level", bool_batch=True)
                    picked_actions = zip(range(len(update_batch_by_item[1])), update_batch_by_item[1])
                    td_targets = np.array(update_batch_by_item[2])
                    returns = np.array(update_batch_by_item[3])
                    agent.low_agents[next_subtask_idx].update_params(states, picked_actions, td_targets, returns)

                update_batch_by_item_by_subtask = [[[] for _ in range(4)] for _ in range(4)] # clear

            if FLAGS.training_stage != 1 and iteration > 0 and iteration % 64 == 0:
                # process batch data
                tmp_time = time.time()
                states = read_batch_state(update_batch_by_item_high_level[0], None, None,
                                          "high_level", bool_batch=True)
                picked_actions = zip(range(len(update_batch_by_item_high_level[1])), update_batch_by_item_high_level[1])
                td_targets = np.array(update_batch_by_item_high_level[2])
                returns = np.array(update_batch_by_item_high_level[3])
                agent.high_agent.update_params(states, picked_actions, td_targets, returns)
                print("Elapsed time for high-level agent updation: %.3f." % (time.time() - tmp_time))
                print("High-level agent updated!")

                update_batch_by_item_high_level = [[] for _ in range(4)] # clear

            # evaluation
            if iteration % FLAGS.iters_per_validation == 0 or iteration == max_iteration - 1:
                evaluation_start_time = time.time()
                print("Evaluation on iteration %d, elapsed time %.3f:" % (
                    iteration, (evaluation_start_time - start_time)))
                print("Current step (4 low-level and the high-level agent): %s" % (str(
                    [_agent.get_current_time_step() for _agent in agent.low_agents + [agent.high_agent]])))

                print("Evaluation on Training data(this iteration interval):")
                cf_acc, channel_accuracy, averaged_acc, external_reward_mode2averaged_reward, \
                averaged_internal_reward_by_subtask, averaged_global_run, averaged_local_run_by_subtask, \
                averaged_overall_local_run, averaged_ask_user_count_by_subtask, \
                averaged_user_select_ans_count_by_subtask = _evaluate(
                    correctness_by_subtask, accuracy, external_reward_mode2reward_list, internal_rewards_by_subtask,
                    global_runs, local_runs_by_subtask, ask_user_counts_by_subtask, user_select_ans_counts_by_subtask)

                averaged_internal_reward = np.average(averaged_internal_reward_by_subtask)
                train_iter2performance[iteration] = [cf_acc, channel_accuracy, averaged_acc,
                                                     external_reward_mode2averaged_reward,
                                                     averaged_internal_reward_by_subtask,
                                                     averaged_global_run, averaged_local_run_by_subtask,
                                                     averaged_overall_local_run, averaged_ask_user_count_by_subtask,
                                                     averaged_internal_reward, averaged_user_select_ans_count_by_subtask]

                print("Evaluation on sampled Dev data:")
                cf_acc, channel_accuracy, averaged_acc, external_reward_mode2averaged_reward, \
                averaged_internal_reward_by_subtask, averaged_global_run, averaged_local_run_by_subtask, \
                averaged_overall_local_run, averaged_ask_user_count_by_subtask, \
                averaged_user_select_ans_count_by_subtask = \
                    _test_and_evaluate(agent, data, "sample_dev")

                averaged_internal_reward = np.average(averaged_internal_reward_by_subtask)
                iter2performance[iteration] = [cf_acc, channel_accuracy, averaged_acc,
                                               external_reward_mode2averaged_reward,
                                               averaged_internal_reward_by_subtask, averaged_global_run,
                                               averaged_local_run_by_subtask, averaged_overall_local_run,
                                               averaged_ask_user_count_by_subtask,
                                               averaged_internal_reward, averaged_user_select_ans_count_by_subtask]
                bool_should_save = False

                # overall acc
                if cf_acc > best_cf_acc:
                    best_cf_acc = cf_acc
                    best_cf_acc_iter = iteration
                    bool_should_save = True
                    print("Updated best overall acc: %.3f" % best_cf_acc)

                # # high-level agent
                # if averaged_external_reward > best_external_reward:
                #     best_external_reward = averaged_external_reward
                #     best_high_level_iter = iteration
                #     bool_should_save = True
                #     print("Updated best external reward: %.3f" % (best_external_reward))
                for mode, aver_value in external_reward_mode2averaged_reward.items():
                    if aver_value > external_reward_mode2best_reward.get(mode, -100.0):
                        external_reward_mode2best_reward[mode] = aver_value
                        external_reward_mode2best_iter[mode] = iteration
                        bool_should_save = True
                        print("Updated best external reward for mode %d: %.3f" % (mode, aver_value))

                # averaged low-level performance
                if averaged_internal_reward > best_averaged_internal_reward:
                    best_averaged_internal_reward = averaged_internal_reward
                    best_averaged_low_level_iter = iteration
                    bool_should_save = True
                    print("Updated best averaged internal reward: %.3f" % best_averaged_internal_reward)

                print("Current best: cf_acc=%.3f, external rewards=%s, averaged internal reward=%.3f." % (
                    best_cf_acc,
                    str(["mode %d: %.3f" % (mode, external_reward_mode2best_reward[mode])
                         for mode in sorted(external_reward_mode2best_reward.keys())]),
                    best_averaged_internal_reward
                ))

                if bool_should_save or iteration == max_iteration - 1 or iteration % 500000 == 0:
                    # save params
                    print("Save all params to %s." % checkpoint_overall_path)
                    agent.saver.save(sess, os.path.join(checkpoint_overall_path, "ckpt"),
                                     global_step=iteration, write_meta_graph=False)
                    with open(os.path.join(checkpoint_overall_path, "stats.pkl"), "wb") as f:
                        pickle.dump({
                                     "best_cf_acc": best_cf_acc,
                                     "best_cf_acc_iter": best_cf_acc_iter,
                                     "best_averaged_internal_reward": best_averaged_internal_reward,
                                     "best_external_reward": external_reward_mode2best_reward[1],
                                     "best_high_level_iter": external_reward_mode2best_iter[1],
                                     "external_reward_mode2best_reward": external_reward_mode2best_reward,
                                     "external_reward_mode2best_iter": external_reward_mode2best_iter,
                                     "best_averaged_low_level_iter": best_averaged_low_level_iter,
                                     "iteration": iteration,
                                     "agent_current_time_steps": [item.get_current_time_step()
                                                                 for item in agent.low_agents + [agent.high_agent]]},f)

                if bool_should_save or iteration == max_iteration - 1 or count_skipped_save == 50:
                    # save all records
                    print("Save records!")
                    with open(os.path.join(checkpoint_overall_path, "train_iter2performance.pkl"), "wb") as f:
                        pickle.dump(train_iter2performance, f)
                    with open(os.path.join(checkpoint_overall_path, "dev_iter2performance.pkl"), "wb") as f:
                        pickle.dump(iter2performance, f)
                    count_skipped_save = 1
                    print("count_skipped_save cleared!")
                else:
                    count_skipped_save += 1

                # clear records
                accuracy = []
                final_predictions = []
                # external_rewards = []
                external_reward_mode2reward_list = collections.defaultdict(list)
                internal_rewards_by_subtask = [[] for _ in range(4)]
                global_runs = []
                local_runs_by_subtask = [[] for _ in range(4)]
                ask_user_counts_by_subtask = [[] for _ in range(4)]
                user_select_ans_counts_by_subtask = [[] for _ in range(4)]
                correctness_by_subtask = [[] for _ in range(4)]
                high_level_orders = []
                start_time = time.time()
                gc.collect()
                print("Record cleared!")

            sys.stdout.flush()


def _eval_by_tag(data, accuracy, correctness_by_subtask, external_reward_mode2reward_list, internal_rewards_by_subtask,
                 global_runs, local_runs_by_subtask, ask_user_counts_by_subtask, user_select_ans_counts_by_subtask):
    """ Evaluation by tags. """
    accuracy_by_tag = collections.defaultdict(list)
    correctness_by_subtask_by_tag = collections.defaultdict(lambda: [[] for _ in range(4)])
    # external_rewards_by_tag = collections.defaultdict(list)
    external_reward_mode2reward_list_by_tag = collections.defaultdict(lambda: collections.defaultdict(list))
    internal_rewards_by_subtask_by_tag = collections.defaultdict(lambda: [[] for _ in range(4)])
    global_runs_by_tag = collections.defaultdict(list)
    local_runs_by_subtask_by_tag = collections.defaultdict(lambda: [[] for _ in range(4)])
    ask_user_counts_by_subtask_by_tag = collections.defaultdict(lambda: [[] for _ in range(4)])
    user_select_counts_by_subtask_by_tag = collections.defaultdict(lambda: [[] for _ in range(4)])

    list_of_instance_reward_mode2value = []
    for instance_idx in range(len(data)):
        list_of_instance_reward_mode2value.append(
            {mode:reward_list[instance_idx] for mode, reward_list in external_reward_mode2reward_list.items()})

    for instance_idx, (instance, acc, ext_r, g_run) in enumerate(
            zip(data, accuracy, list_of_instance_reward_mode2value, global_runs)):
        tags = instance.get("tags", ["Not-assigned"])
        correctness = [item[instance_idx] for item in correctness_by_subtask]
        inter_r = [item[instance_idx] for item in internal_rewards_by_subtask]
        local_runs = [item[instance_idx] for item in local_runs_by_subtask]
        ask_user_counts = [item[instance_idx] for item in ask_user_counts_by_subtask]
        user_select_counts = [item[instance_idx] for item in user_select_ans_counts_by_subtask]
        for tag in tags:
            accuracy_by_tag[tag].append(acc)
            # external_rewards_by_tag[tag].append(ext_r)
            for mode, value in ext_r.items():
                external_reward_mode2reward_list_by_tag[tag][mode].append(value)
            global_runs_by_tag[tag].append(g_run)
            for subtask_idx in range(4):
                correctness_by_subtask_by_tag[tag][subtask_idx].append(correctness[subtask_idx])
                internal_rewards_by_subtask_by_tag[tag][subtask_idx].append(inter_r[subtask_idx])
                local_runs_by_subtask_by_tag[tag][subtask_idx].append(local_runs[subtask_idx])
                ask_user_counts_by_subtask_by_tag[tag][subtask_idx].append(ask_user_counts[subtask_idx])
                user_select_counts_by_subtask_by_tag[tag][subtask_idx].append(user_select_counts[subtask_idx])

    # eval
    for tag, tagged_accuracy in accuracy_by_tag.items():
        print("%s\n%s" % ("="*10, "Tag %s:" % tag))
        _evaluate(correctness_by_subtask_by_tag[tag], tagged_accuracy, external_reward_mode2reward_list_by_tag[tag],
                  internal_rewards_by_subtask_by_tag[tag], global_runs_by_tag[tag],
                  local_runs_by_subtask_by_tag[tag], ask_user_counts_by_subtask_by_tag[tag],
                  user_select_counts_by_subtask_by_tag[tag], bool_easy_record=False)
        print("")


def _evaluate(correctness_by_subtask, accuracy, external_reward_mode2reward_list, internal_rewards_by_subtask,
              global_runs, local_runs_by_subtask, ask_user_counts_by_subtask, user_select_ans_counts_by_subtask,
              bool_easy_record=False):
    # averaged_external_reward = np.average(external_rewards)
    external_reward_mode2averaged_reward = {mode:np.average(reward_list)
                                            for mode, reward_list in external_reward_mode2reward_list.items()}
    averaged_internal_reward_by_subtask = [np.average(subtask_internal_rewards)
                                           for subtask_internal_rewards in internal_rewards_by_subtask]
    averaged_ask_user_count_by_subtask = [np.average(ask_user_counts)
                                          for ask_user_counts in ask_user_counts_by_subtask]
    averaged_user_select_ans_count_by_subtask = [np.average(user_select_ans_count)
                                                 for user_select_ans_count in user_select_ans_counts_by_subtask]
    averaged_acc = np.average(accuracy)
    averaged_global_run = np.average(global_runs)
    averaged_local_run_by_subtask = [np.average(local_runs) for local_runs in local_runs_by_subtask]
    averaged_overall_local_run = sum(averaged_local_run_by_subtask)

    # accuracy by subtask
    accuracy_by_subtask = [np.average(correctness) for correctness in correctness_by_subtask]

    # channel accuracy (tc+ac)
    channel_correctness = [np.all([t_c, a_c]) for t_c, a_c in zip(correctness_by_subtask[0], correctness_by_subtask[2])]
    channel_accuracy = np.average(channel_correctness)

    # overall accuracy (tc+tf+ac+af)
    cf_correctness = [np.all([t_c, t_f, a_c, a_f]) for t_c, t_f, a_c, a_f in
                           zip(correctness_by_subtask[0], correctness_by_subtask[1],
                               correctness_by_subtask[2], correctness_by_subtask[3])]
    cf_accuracy = np.average(cf_correctness) # final accuracy

    print("Evaluation result (averaged): "
          "CF_acc=%.3f, channel_acc=%.3f, overall_acc=%.3f, external_reward=%s, internal_reward=%s, global_run=%.3f, "
          "local_run_by_subtask=%s, overall_local_run=%.3f, ask_user_count_by_subtask=%s, "
          "user_select_ans_count_by_subtask=%s" % (
          cf_accuracy, channel_accuracy, averaged_acc,
          str(["mode %d: %.3f" % (mode, external_reward_mode2averaged_reward[mode])
               for mode in sorted(external_reward_mode2averaged_reward.keys())]),
          ",".join(["%.3f" % r for r in averaged_internal_reward_by_subtask]),
          averaged_global_run, ",".join(["%.3f" % run for run in averaged_local_run_by_subtask]),
          averaged_overall_local_run, ",".join(["%.3f" % count for count in averaged_ask_user_count_by_subtask]),
          ",".join(["%.3f" % count for count in averaged_user_select_ans_count_by_subtask])
          ))
    if bool_easy_record:
        print("# For easy record: %.3f\t%.3f\t%.3f\t%.3f" % (
            channel_accuracy, cf_accuracy, averaged_acc, sum(averaged_ask_user_count_by_subtask)))
    print("Acuracy by subtask: ")
    for idx in range(4):
        print("Subtask %d: %.3f" % (idx, accuracy_by_subtask[idx]))

    return cf_accuracy, channel_accuracy, averaged_acc, external_reward_mode2averaged_reward, \
           averaged_internal_reward_by_subtask, \
           averaged_global_run, averaged_local_run_by_subtask, averaged_overall_local_run, \
           averaged_ask_user_count_by_subtask, averaged_user_select_ans_count_by_subtask


def _test_and_evaluate(agent, data, data_type, savepath=None, bool_eval_by_tag=False,
                       bool_greedy_test=FLAGS.greedy_test):
    target_data = data[data_type]
    user_answer = data["user_answers"]
    chnl_fn_constraints = data.get('chnl_fn_constraints', None)

    final_predictions = []
    accuracy = []
    correctness_by_subtask = [[] for _ in range(4)]
    # external_rewards = []
    external_reward_mode2reward_list = collections.defaultdict(list)
    internal_rewards_by_subtask = [[] for _ in range(4)]
    local_runs_by_subtask = [[] for _ in range(4)]
    ask_user_counts_by_subtask = [[] for _ in range(4)]
    user_select_ans_counts_by_subtask = [[] for _ in range(4)]
    global_runs = []
    high_level_orders = []
    chosed_user_answers = []

    # create env
    env = create_env(agent, user_answer, chnl_fn_constraints)
    for data_idx, instance in enumerate(target_data):
        # if data_idx % 1000 == 0 or data_idx == len(target_data) - 1:
            # print("Index %d, elapsed time: %.3f" % (data_idx, (time.time() - start_time)))
            # sys.stdout.flush()
        instruction = instance['ids']
        ground_truth = instance['labels']
        ground_truth_names = instance['label_names']

        env.reset(instruction, ground_truth, ground_truth_names)
        instance_high_level_order = []
        instance_user_answer = []

        bool_terminal_overall = False
        bool_subtask_terminal = True
        # instance_external_reward = 0
        instance_external_reward_mode2reward = dict()
        instance_internal_reward = [0] * 4
        subtask_ans_list = []
        while not bool_terminal_overall:
            next_actions, zi, next_subtask_idx, high_v_value, low_v_value = agent.sample(
                env, bool_greedy=bool_greedy_test, num_top=1)
            prev_low_level_state, prev_high_level_state, new_state, \
            (external_reward_mode2reward, internal_reward, _), bool_terminal_overall, \
            (bool_ask_user, ans) = env.step(
                next_actions, zi, next_subtask_idx, bool_subtask_terminal, bool_verbose=False,
                bool_correct_binding=False)
            subtask_ans_list.append(ans)

            # subtask terminal update
            bool_subtask_terminal = env.get_state().get_cur_subtask_terminal()
            # accumulate rewards
            instance_internal_reward[next_subtask_idx] += internal_reward
            if bool_subtask_terminal:
                # instance_external_reward += external_reward
                for mode, value in external_reward_mode2reward.items():
                    instance_external_reward_mode2reward[mode] = instance_external_reward_mode2reward.get(mode, 0.0) + value
                instance_high_level_order.append(next_subtask_idx)
                instance_user_answer.append(subtask_ans_list)
                subtask_ans_list = []

        # save
        final_predictions.append(env.get_state().get_cur_preds())
        accuracy.append(env.get_state().get_cur_acc())
        correctness = env.get_state().get_cur_all_correctness()
        local_runs = env.get_all_local_run_count()
        ask_user_count = env.get_ask_user_count()
        user_select_ans_count = env.get_select_ans_count()
        for idx in range(4):
            correctness_by_subtask[idx].append(correctness[idx])
            internal_rewards_by_subtask[idx].append(instance_internal_reward[idx])
            local_runs_by_subtask[idx].append(local_runs[idx])
            ask_user_counts_by_subtask[idx].append(ask_user_count[idx])
            user_select_ans_counts_by_subtask[idx].append(user_select_ans_count[idx])

        # external_rewards.append(instance_external_reward)
        for mode, value in instance_external_reward_mode2reward.items():
            external_reward_mode2reward_list[mode].append(value)
        global_runs.append(env.get_global_run_count())
        high_level_orders.append(instance_high_level_order)
        chosed_user_answers.append(instance_user_answer)

    # eval
    final_accuracy, channel_accuracy, averaged_acc, external_reward_mode2averaged_reward, \
    averaged_internal_reward_by_subtask, averaged_global_run, averaged_local_run_by_subtask, \
    averaged_overall_local_run, averaged_ask_user_count_by_subtask, averaged_user_select_ans_count_by_subtask\
        = _evaluate(correctness_by_subtask, accuracy, external_reward_mode2reward_list, internal_rewards_by_subtask,
                    global_runs, local_runs_by_subtask, ask_user_counts_by_subtask, user_select_ans_counts_by_subtask,
                    bool_easy_record=True)

    if bool_eval_by_tag:
        _eval_by_tag(target_data, accuracy, correctness_by_subtask, external_reward_mode2reward_list,
                     internal_rewards_by_subtask, global_runs, local_runs_by_subtask,
                     ask_user_counts_by_subtask, user_select_ans_counts_by_subtask)

    if savepath:
        with open(savepath, "wb") as f:
            pickle.dump({
                "final_accuracy": final_accuracy, "channel_accuracy": channel_accuracy,
                "final_predictions": final_predictions,
                "correctness_by_subtask": correctness_by_subtask,
                "accuracy": accuracy,
                "external_reward_mode2averaged_reward": external_reward_mode2averaged_reward,
                "internal_rewards_by_subtask": internal_rewards_by_subtask,
                "global_runs": global_runs,
                "local_runs_by_subtask": local_runs_by_subtask,
                "ask_user_counts_by_subtask": ask_user_counts_by_subtask,
                "user_select_ans_counts_by_subtask": user_select_ans_counts_by_subtask,
                "high_level_orders": high_level_orders,
                "chosed_user_answer": chosed_user_answers
            }, f)

    return final_accuracy, channel_accuracy, averaged_acc, external_reward_mode2averaged_reward, \
           averaged_internal_reward_by_subtask, averaged_global_run, averaged_local_run_by_subtask, \
           averaged_overall_local_run, averaged_ask_user_count_by_subtask, \
           averaged_user_select_ans_count_by_subtask


def test():
    data = pickle.load(open("../../data/lam/toy_data_with_noisy_user_ans.pkl"))

    data_type = "test"
    bool_eval_by_tag = True if data_type == "test" else False

    with tf.Session() as sess:
        # set up
        print("Creating agent...\n")
        agent, _ = create_agent(sess, FLAGS.agent_type)

        for random_seed in range(1, 3):
            # set the random seed
            tf.set_random_seed(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)

            savepath = None

            _test_and_evaluate(agent, data, data_type, savepath=savepath,
                               bool_eval_by_tag=bool_eval_by_tag,
                               bool_greedy_test=True)


def main():
    if FLAGS.train:
        train()
    elif FLAGS.test:
        test()
    else:
        return


if __name__ == "__main__":
    main()
