# interactive_test.py
# Implementation for user study
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable warning msg from TF
import collections
import random
import tensorflow as tf
import argparse

from utils import *
import config
from lam_rule_agent import LamRuleAssembleAgent
from hierarchical_agent import HierarchicalAgent
from hierarchical_env import HierIFTTTEnvironment
from interactive_setting import *

import pdb

num_left = 0 # tracking remaining size


def print_header(remaining_size):
    task_notification = " Conversational IFTTT Program Synthesis "
    remain_notification = " Remaining: %d " % (remaining_size)
    print "=" * 50
    print (bcolors.BOLD + task_notification + bcolors.ENDC)
    print (bcolors.BOLD + remain_notification + bcolors.ENDC)
    print "=" * 50
    print ""


def print_instruction(text):
    print bcolors.BOLD + text + bcolors.ENDC


def create_env(agent, user_answer, chnl_fn_constraints, source4real_user, setting):
    env = HierIFTTTEnvironment(
        setting.penalty_global_run, setting.penalty_local_ask_user, config.MAX_GLOBAL_RUN,
        config.MAX_LOCAL_RUN, user_answer, config.ASK_USER_INDICES, agent.low_agents,
        chnl_fn_constraints=chnl_fn_constraints if setting.low_level_action_mask else None,
        source4real_user=source4real_user, bool_sibling_z=setting.bool_sibling_z,
        agent_type=setting.agent_type)

    return env


def create_agent(session, agent_type, setting):
    assert agent_type in {"hRL", "hRL_fixedHigh", "lam_human", "lam_rule"}
    config_setting = config.HierAgentConfig(setting)
    if agent_type == "lam_rule":
        agent = LamRuleAssembleAgent(config_setting, session)
    else:
        agent = HierarchicalAgent(config_setting, session)
    _safe_scope = setting.scope + ("/" if not setting.scope.endswith("/") else "")
    session.run(tf.variables_initializer(tf.global_variables(scope=_safe_scope)))

    if agent_type == "lam_human":
        # load pretrained baseline agent
        baseline_agent_path = "baseline_agent"
        # print("Loading pretrained low-level agents from %s." % baseline_agent_path)
        ckpt_pretrain = tf.train.get_checkpoint_state(baseline_agent_path)
        assert ckpt_pretrain
        agent.pretrain_saver.restore(session, ckpt_pretrain.model_checkpoint_path)
        return agent

    # restore previous training results
    checkpoint_file_path = "%s/ckpt" % agent_type
    agent.saver.restore(session, os.path.join(checkpoint_file_path))

    return agent


def _interactive_test(agent, agent_type, data, data_tag, setting, test_number=None,
                      bool_greedy_test=True):
    target_data = [item for item in data["test"] if data_tag in item["tags"]]
    user_answer = data["user_answers"]
    chnl_fn_constraints = data.get('chnl_fn_constraints', None)
    word2id = data['word_ids']
    trigger_fn2desc = data["trigger_fn2desc"]
    action_fn2desc = data["action_fn2desc"]

    if test_number:
        target_data = random.sample(target_data, test_number)
    else:
        random.shuffle(target_data)

    id2word = {v: k for k, v in word2id.items()}
    labeler_encoders = [data['labelers'][subtask]
                        for subtask in ["trigger_chans", "trigger_funcs", "action_chans", "action_funcs"]]

    final_predictions = []
    accuracy = []
    correctness_by_subtask = [[] for _ in range(4)]
    # external_rewards = []
    external_reward_mode2reward_list = collections.defaultdict(list)
    internal_rewards_by_subtask = [[] for _ in range(4)]
    local_runs_by_subtask = [[] for _ in range(4)]
    ask_user_counts_by_subtask = [[] for _ in range(4)]
    global_runs = []
    high_level_orders = []
    user_answer_records = []

    # create env
    env = create_env(agent, user_answer, chnl_fn_constraints,
                     (word2id, labeler_encoders, trigger_fn2desc, action_fn2desc), setting)
    for data_idx, instance in enumerate(target_data):
        os.system('clear')
        print_header(num_left)
        instruction = instance['ids']
        instruction_str = " ".join(instance['words'])
        ground_truth = instance['labels']
        ground_truth_names = instance['label_names']
        pseudo_ask_labels = instance.get('pseudo_ask_labels', None)
        bool_questions = False
        print_instruction("Please read the information below:")
        print(bcolors.RED + bcolors.BOLD + "IFTTT recipe description: " + bcolors.ENDC + instruction_str)
        print(bcolors.RED + bcolors.BOLD + "Ground-truth program (recipe): " + bcolors.ENDC + str(ground_truth_names))
        print("")

        print_instruction("Please read the following definitions of the program components:")
        # trigger
        tf_desc = trigger_fn2desc["%s.%s" % (ground_truth_names[0].lower(), ground_truth_names[1].lower())]
        tf_desc = tf_desc[0].upper() + tf_desc[1:]
        print("- \'%s\'" % ground_truth_names[0] + bcolors.BOLD + bcolors.BLUE + "(trigger channel name): " + bcolors.ENDC +
              "The service that you will use for trigger.")
        print("- \'%s\' " % ground_truth_names[1] + bcolors.BOLD + bcolors.BLUE + "(trigger function name): " + bcolors.ENDC +
            tf_desc)
        # action
        af_desc = action_fn2desc["%s.%s" % (ground_truth_names[2].lower(), ground_truth_names[3].lower())]
        af_desc = af_desc[0].upper() + af_desc[1:]
        print("- \'%s\' " % ground_truth_names[2] + bcolors.BOLD + bcolors.BLUE + "(action channel name): " + bcolors.ENDC +
              "The service that you will use for action.")
        print("- \'%s\' " % ground_truth_names[3] + bcolors.BOLD + bcolors.BLUE + "(action function name): " + bcolors.ENDC +
            af_desc)

        print(bcolors.BOLD + "Now, assuming you are the user who gave the above " + bcolors.RED +
              bcolors.BOLD + "recipe description " + bcolors.ENDC + bcolors.BOLD + "and had the " + bcolors.RED +
              "ground-truth program " + bcolors.ENDC + bcolors.BOLD +
              "as your goal. Please answer the prompted questions (if any): " + bcolors.ENDC)
        env.reset(instruction, ground_truth, ground_truth_names, pseudo_ask_labels, agent_type=setting.agent_type)
        instance_high_level_order = []
        real_user_answer = collections.defaultdict(list)

        bool_terminal_overall = False
        bool_subtask_terminal = True
        # instance_external_reward = 0
        instance_external_reward_mode2reward = dict()
        instance_internal_reward = [0] * 4
        while not bool_terminal_overall:
            # env.print_env(); pdb.set_trace()
            next_actions, zi, next_subtask_idx, high_v_value, low_v_value = agent.sample(
                env, bool_greedy=bool_greedy_test or agent_type == "lam_human", num_top=1, max_ask=1)
            prev_low_level_state, prev_high_level_state, new_state, \
            (external_reward_mode2reward, internal_reward, _), bool_terminal_overall, \
            (bool_ask_user, ans) = env.step(
                next_actions, zi, next_subtask_idx, bool_subtask_terminal, bool_verbose=False,
                bool_correct_binding=False)
            bool_questions = bool_questions or bool_ask_user

            # track user answers
            if bool_ask_user:
                real_user_answer[next_subtask_idx].append(env.get_user_simulator().get_current_user_answer())

            # subtask terminal update
            bool_subtask_terminal = env.get_state().get_cur_subtask_terminal()
            # accumulate rewards
            instance_internal_reward[next_subtask_idx] += internal_reward
            if bool_subtask_terminal:
                # instance_external_reward += external_reward
                for mode, value in external_reward_mode2reward.items():
                    instance_external_reward_mode2reward[mode] = instance_external_reward_mode2reward.get(
                        mode, 0.0) + value
                instance_high_level_order.append(next_subtask_idx)

        # if there's no question prompted
        if not bool_questions:
            print(bcolors.PINK + bcolors.BOLD + "\nThe agent decides not to ask questions." + bcolors.ENDC)

        # env.print_env(bool_readable=True, id2word=id2word, labeler_encoders=labeler_encoders, unk=12621)
        print(bcolors.RED + bcolors.BOLD + "\nPrediction: " + bcolors.ENDC +
              str(label2name(env.get_state().get_cur_preds(), labeler_encoders)))

        # save
        final_predictions.append(env.get_state().get_cur_preds())
        accuracy.append(env.get_state().get_cur_acc())
        correctness = env.get_state().get_cur_all_correctness()
        local_runs = env.get_all_local_run_count()
        ask_user_count = env.get_ask_user_count()
        for idx in range(4):
            correctness_by_subtask[idx].append(correctness[idx])
            internal_rewards_by_subtask[idx].append(instance_internal_reward[idx])
            local_runs_by_subtask[idx].append(local_runs[idx])
            ask_user_counts_by_subtask[idx].append(ask_user_count[idx])
        # external_rewards.append(instance_external_reward)
        for mode, value in instance_external_reward_mode2reward.items():
            external_reward_mode2reward_list[mode].append(value)
        global_runs.append(env.get_global_run_count())
        high_level_orders.append(instance_high_level_order)
        user_answer_records.append(real_user_answer)

    results = {
        "target_data": target_data,
        "user_answer_records": user_answer_records,
        "final_predictions": final_predictions,
        "correctness_by_subtask": correctness_by_subtask,
        "accuracy": accuracy,
        "external_reward_mode2reward_list": external_reward_mode2reward_list,
        "internal_rewards_by_subtask": internal_rewards_by_subtask,
        "global_runs": global_runs,
        "local_runs_by_subtask": local_runs_by_subtask,
        "ask_user_counts_by_subtask": ask_user_counts_by_subtask,
        "high_level_orders": high_level_orders
    }

    return results


def evaluation_single_person(results):
    print("=" * 50)

    agent_list = ["lam_rule", "lam_human", "hRL_fixedHigh", "hRL"]
    agent_type2stats = dict()

    for agent_type in agent_list:
        agent_results = results[agent_type]
        print("\nAgent: %s, num %d." % (agent_type, len(agent_results)))

        ask_user_counts = []
        channel_correct = []
        overall_correct = []
        correct = []
        for recipe in agent_results:
            recipe_ask_user_counts_by_subtask = np.array(recipe["ask_user_counts_by_subtask"])
            ask_user_counts.append(sum(recipe_ask_user_counts_by_subtask)[0])
            #pdb.set_trace()
            recipe_correctness_by_subtask = recipe["correctness_by_subtask"]
            channel_correct.append(np.all(recipe_correctness_by_subtask[0] + recipe_correctness_by_subtask[2]))
            overall_correct.append(np.all(recipe_correctness_by_subtask))
            correct.append(np.mean(recipe_correctness_by_subtask))

        agent_type2stats[agent_type] = {"average_asks": np.average(ask_user_counts),
                                        "channel_acc": np.average(channel_correct),
                                        "C+F acc": np.average(overall_correct),
                                        "overall acc": np.average(correct),
                                        "correctness": overall_correct,
                                        "ask_user_counts": ask_user_counts}
        print("Stats: ")
        measures = ["C+F acc", "overall acc", "channel_acc", "average_asks"]
        for key in measures:
            print key, agent_type2stats[agent_type][key]

        # for i in agent_type2stats[agent_type]["ask_user_counts"]:
        #     print int(i)
        # pdb.set_trace()

    print("=" * 50)

    return agent_type2stats


def evaluate_folder(folder_name, bool_combine):
    files = os.listdir(folder_name)
    agent_list = ["lam_rule", "lam_human", "hRL_fixedHigh", "hRL"]

    results = collections.defaultdict(list)
    for file in files:
        print file
        _results = pickle.load(open(folder_name + file))
        if bool_combine:
            for k in agent_list:
                results[k].extend(_results[k])
        else:
            evaluation_single_person(_results)

    if bool_combine:
        evaluation_single_person(results)


def get_lam_test_correctness(lam_test_results, test_data):
    results = []
    for subtask in ["trigger_chans", "trigger_funcs", "action_chans", "action_funcs"]:
        results.append(lam_test_results[subtask][0.0]['final_prediction'])
    combined_results = zip(results[0], results[1], results[2], results[3]) # a list of (pred0,...,pred3) tuples

    correctness = []
    for prediction, truth in zip(combined_results, test_data):
        true_labels = np.array(truth["labels"])
        pred_labels = np.array(list(prediction))
        correctness.append(np.all(true_labels == pred_labels))

    print("C+F acc: %.3f" % np.average(correctness))

    return correctness


def evaluate_lam(lam_test_correctness, test_ids, id2idx):
    indices = [id2idx[id] for id in test_ids]
    correctness = [lam_test_correctness[idx] for idx in indices]
    print("#=%d, #correct=%d, C+F acc=%.3f" % (len(correctness), correctness.count(True), np.average(correctness)))


def user_test(level, user_name):
    data = pickle.load(open("../../data/lam/toy_data_with_noisy_user_ans.pkl"))
    data["trigger_fn2desc"] = pickle.load(open("../../data/source/trigger_function2description.pkl"))
    data["action_fn2desc"] = pickle.load(open("../../data/source/action_function2description.pkl"))
    data_tag = level

    agent_list = ["lam_rule", "lam_human", "hRL_fixedHigh", "hRL"]
    dir = "interactive_test/noisy_user_ans/"
    if not os.path.exists(dir):
        os.makedirs(dir)

    num_todo = 80

    user = "%s_%s" % (user_name, data_tag)
    userpath = os.path.join(dir, "%s.pkl" % user)
    try:
        user_record = pickle.load(open(userpath))
        # user_todo = user_record["todo"]
        user_done = user_record["done"]
    except:
        user_record = collections.defaultdict(list)
        # user_todo = [20] * len(agent_list)
        # user_record["todo"] = user_todo
        user_done = [0] * len(agent_list)
        user_record["done"] = user_done
        pickle.dump(user_record, open(userpath, "wb"))

    global num_left
    # num_left = sum(user_todo)
    num_left = num_todo - sum(user_done)
    os.system("clear")
    print_header(num_left)

    with tf.Session() as sess:
        agent_instances = []
        if num_left:
            print("Creating agents...")
            for agent_type in agent_list:
                # print(agent_type)
                agent_instances.append(create_agent(sess, agent_type, setting[agent_type]))

        while num_left:
            # print("Randomly picking one agent...")
            # agent_idx = random.choice([agent_idx for agent_idx,num_todo in enumerate(user_todo) if num_todo])
            agent_idx = random.choice(range(len(agent_list)))
            # agent_idx = 2
            agent_type = agent_list[agent_idx]
            agent = agent_instances[agent_idx]
            # print("agent type:", agent_type)

            results = _interactive_test(agent, agent_type, data, data_tag, setting[agent_type],
                                        test_number=1, bool_greedy_test=True)

            print_instruction("\nSaving records...")
            user_record[agent_type].append(results)
            # user_todo[agent_list.index(agent_type)] -= 1
            # user_record["todo"] = user_todo
            user_done[agent_idx] += 1
            user_record["done"] = user_done
            pickle.dump(user_record, open(userpath, "wb"))
            # num_left = sum(user_todo)
            num_left = num_todo - sum(user_done)

            # detect end signal
            end_signal = raw_input(bcolors.GREEN + bcolors.BOLD +
                                   "\nNext? Press Enter for continue, Ctrl+C for exit." + bcolors.ENDC)
            if end_signal != "":
                return

        print(bcolors.YELLOW + bcolors.BOLD + "Congratulations! You have completed all your task!" + bcolors.ENDC)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--level', type=str)
    parser.add_argument('--user-name', type=str)
    parser.add_argument('--eval-all', action='store_true')

    args = parser.parse_args()

    if not args.eval and not args.eval_all:
        assert args.level and args.user_name
        assert args.level in {"VI-3", "VI-4"}
        user_test(args.level, args.user_name)
    else:
        if args.eval:
            evaluation_single_person(pickle.load(
                open("interactive_test/noisy_user_ans/%s_%s.pkl" % (args.user_name, args.level))))
        if args.eval_all:
            folder_name = "interactive_test/noisy_user_ans/"
            evaluate_folder(folder_name, True)

    # lam_test_results = pickle.load(open("lam_test.pkl"))
    # data = pickle.load(open("../../data/lam/toy_data_with_noisy_user_ans.pkl"))
    # lam_correctness = get_lam_test_correctness(lam_test_results, data['test'])
    # pickle.dump(lam_correctness, open("lam_test_correctness.pkl", "wb"))

    # lam_test_correctness = pickle.load(open("lam_test_correctness.pkl"))
    # test_ids = pickle.load(open("interactive_test/noisy_user_ans_total_recipe_ids.pkl"))
    # id2idx, _ = pickle.load(open("../../data/lam/test_data_id2idx_idx2id.pkl"))
    # evaluate_lam(lam_test_correctness, test_ids, id2idx)
    

if __name__ == "__main__":
    main()
