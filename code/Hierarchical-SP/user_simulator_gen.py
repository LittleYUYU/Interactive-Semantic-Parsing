# -*- coding: utf-8 -*-

import csv
import pickle
import operator
import pdb
import random
import collections
import re
from collections import Counter
from nltk import word_tokenize, pos_tag, wordpunct_tokenize
import numpy as np
import os
import copy

TITLES = ["id", "description", "triggerchannel", "trigger", "actionchannel", "action"]
TRIGGER_EXPRESS_TO_REMOVE = "this trigger fires"
ACTION_EXPRESS_TO_REMOVE = "this action will"


##############################
## Update: paraphrase (1):
## Simply taking the content after "if" or "then" of the same instruction
## into paraphrases of the corresponding function.
##############################
def clean_function_text(fn2text):
    """ Clean function name or description: removing TRIGGER_EXPRESS_TO_REMOVE and ACTION_EXPRESS_TO_REMOVE. """
    new_fn2text = dict()
    for fn, text in fn2text.items():
        old_text = text

        # remove template words
        if text.startswith(TRIGGER_EXPRESS_TO_REMOVE):
            text = text[len(TRIGGER_EXPRESS_TO_REMOVE)+1:]
        if text.startswith(ACTION_EXPRESS_TO_REMOVE):
            text = text[len(ACTION_EXPRESS_TO_REMOVE)+1:]

        # revise to first-person angle
        text = text.replace("your", "my")
        if "you" in text:
            tag_output = pos_tag(word_tokenize(text.decode('utf-8', 'ignore')))
            temp = []
            for idx, (word, tag) in enumerate(tag_output):
                if word != "you":
                    temp.append(word)
                else:
                    if idx+1 < len(tag_output) and tag_output[idx+1][1].startswith('VB'):
                        temp.append("I")
                    else:
                        temp.append("me")
            text = " ".join(temp)
        new_fn2text[fn] = text
        if text != old_text:
            try:
                print("Before: %s. After: %s." % (old_text, text))
            except:
                continue
    return new_fn2text


##############################
## Update: paraphrase (3):
## Simply taking the content after "if" or "then" of the same instruction
## into paraphrases of the corresponding function.
##############################
def extract_fn_user_descriptions(data):
    """ This function collects the paraphrases for trigger/action functions
    from user data, by extracting the if/then phrase. 
    Templates:
    (1) if <t>, (then) <a>
    (2) <a> every time/year/month/week/day/hour <t>
    (3) <a> when <t>
    (4) if <t> then <a>
    (5) <a> if <t>
    (6) when <t>, <a>
    """

    # trigger_fn2user_desc = collections.defaultdict(set)
    # action_fn2user_desc = collections.defaultdict(set)

    # a dict of {template : a dict of {fn:desc set}}
    tf_fn2desc_set_by_template = [collections.defaultdict(set) for _ in range(6)]
    af_fn2desc_set_by_template = [collections.defaultdict(set) for _ in range(6)]

    source_data = data['train'] + data['dev']

    for item in source_data:
        words = item['words']
        tc, tf, ac, af = item['label_names']
        template_id = None
        tf_desc = None
        af_desc = None

        if len(words) < 4:
            continue

        # check the templates one by one
        if words[0] == "if" and words.count("if") == 1: # template (1),(4)
            if "," in words: # template (1)
                template_id = 1
                tf_desc = words[1:words.index(",")]
                af_desc = words[words.index(",")+1:]
                if af_desc and af_desc[0] == "then": # remove the redundant "then"
                    af_desc = af_desc[1:]
            elif "then" in words: # template (4)
                template_id = 4
                tf_desc = words[1:words.index("then")]
                af_desc = words[words.index("then")+1:]
        elif words.count("if") == 1: # template (5)
            template_id = 5
            tf_desc = words[words.index("if")+1:]
            af_desc = words[:words.index("if")]
        elif "if" not in words: # others
            if words.count("when") == 1: # template (3),(6)
                if words[0] == "when" and "," in words:
                    template_id = 6
                    tf_desc = words[1:words.index(",")]
                    af_desc = words[words.index(",")+1:]
                elif words[0] != "when" and "," not in words:
                    template_id = 3
                    tf_desc = words[words.index("when")+1:]
                    af_desc = words[:words.index("when")]
            elif "when" not in words: # template (2)
                phrases = {"every time", "every year", "every month", "every week", "every day", "every hour"}
                picked_phrase = None
                picked_phrase_idx = None
                for word_idx in range(len(words)-2): # at least one token following the phrase
                    phrase = words[word_idx] + " " + words[word_idx+1]
                    if phrase in phrases:
                        picked_phrase = phrase
                        picked_phrase_idx = word_idx
                        break
                if picked_phrase and picked_phrase_idx: # idx must > 0
                    template_id = 2
                    tf_desc = words[picked_phrase_idx:]
                    af_desc = words[:picked_phrase_idx]

        if tf_desc and af_desc:
            tf_fn = "%s.%s" % (tc.lower().strip(), tf.lower().strip())
            af_fn = "%s.%s" % (ac.lower().strip(), af.lower().strip())
            tf_fn2desc_set_by_template[template_id-1][tf_fn].add(" ".join(tf_desc)) # NOTE: be sure to split
            af_fn2desc_set_by_template[template_id-1][af_fn].add(" ".join(af_desc))


    # print
    def _print_stats(src):
        for template_id, fn2desc_set in enumerate(src):
            print("Template %d: %d functions." % (template_id, len(fn2desc_set)))
            samples = random.sample(fn2desc_set.items(), 3)
            for fn, descs in samples:
                print("Function: %s:" % fn)
                for desc in descs:
                    print desc.encode('ascii', 'ignore')
                print("-"*10)
            print("")

    print("Stats of trigger function:")
    _print_stats(tf_fn2desc_set_by_template)
    print("Stats of action function:")
    _print_stats(af_fn2desc_set_by_template)

    return tf_fn2desc_set_by_template, af_fn2desc_set_by_template


def main():
    pass


if __name__ == "__main__":
    main()


