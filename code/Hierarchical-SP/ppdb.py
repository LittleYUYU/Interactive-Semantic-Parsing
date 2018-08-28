# This script collects paraphrases from PPDB
# (http://paraphrase.org/)

import pickle, pdb, random
from nltk import word_tokenize
from nltk.stem import SnowballStemmer

def add_to_dict_of_set(key, value, dict_set):
    if key in dict_set:
        dict_set[key].add(value)
    else:
        dict_set[key] = {value}

def clean_paraphrase(paraphrase_dict):
    stemmer = SnowballStemmer("english")
    paraphrase_dict_clean = dict()
    print("Size: %d" % len(paraphrase_dict))

    for phrase, paraphrases in paraphrase_dict.items():
        new_paraphrases = set()
        for paraphrase in paraphrases:
            if stemmer.stem(phrase) != stemmer.stem(paraphrase):
                new_paraphrases.add(paraphrase)
        if len(new_paraphrases):
            paraphrase_dict_clean[phrase] = new_paraphrases
    print("Size: %d" % len(paraphrase_dict_clean))
    return paraphrase_dict_clean

def collect_pairs_by_rel(filename, rel):
    """ Collect pairs from PPDB maintaining the specified relation. """
    stemmer = SnowballStemmer("english")

    with open(filename, "r") as f:
        data = f.readlines()

    phrase2paraphrase = dict()

    for item in data:
        item = item.strip()
        phrase = item.split('|||')[1].strip()
        paraphrase = item.split('|||')[2].strip()

        if stemmer.stem(phrase) == stemmer.stem(paraphrase):
            continue

        entailment = item.split('|||')[-1].strip()

        if entailment == rel:
            add_to_dict_of_set(phrase, paraphrase, phrase2paraphrase)
            add_to_dict_of_set(paraphrase, phrase, phrase2paraphrase)

    print("Size: %d" % len(phrase2paraphrase))
    return phrase2paraphrase

def gen_paraphrase_for_text(text, paraphrase_dict):
    """ This function replace several words/phrases in a sentence at once
    for generating paraphrases. """
    paraphrases = set()
    tokens = word_tokenize(text)
    replaced = []
    replacement = []
    token_idx = 0
    while token_idx < len(tokens):
        unigram = tokens[token_idx]
        if token_idx < len(tokens) - 1:
            bigram = tokens[token_idx] + " " + tokens[token_idx]
        else:
            bigram = None

        if bigram and bigram in paraphrase_dict:
            replaced.append(bigram)
            replacement.append(paraphrase_dict[bigram])
            token_idx += 1
        elif unigram in paraphrase_dict:
            replaced.append(unigram)
            replacement.append(paraphrase_dict[unigram])

        token_idx += 1

    # generate token possibilities
    num_paraphrases = min([len(replaced)] + [len(item) for item in replacement])
    for item_idx in range(len(replacement)):
        token_para = random.sample(replacement[item_idx], num_paraphrases)
        # if len(token_para) < num_paraphrases:
        #     token_para += [random.choice(token_para) for _ in range(num_paraphrases - len(token_para))]
        replacement[item_idx] = token_para

    # generate paraphrases
    for paraphrase_idx in range(num_paraphrases):
        new_text = text
        for token_replaced, token_para in zip(replaced, replacement):
            new_text = new_text.replace(token_replaced, token_para[paraphrase_idx])
        paraphrases.add(new_text)

    # try:
    #     print("Sentence: %s" % text)
    # except:
    #     pass
    # else:
    #     print("Paraphrases: ")
    #     for sent in paraphrases:
    #         print sent
    #     print("-"*10)

    return paraphrases


def gen_paraphrase(fn2text, paraphrase_dict):
    print("Original size: %d" % len(fn2text))
    fn2paraphrases = dict()
    for fn, text in fn2text.items():
        paraphrases = gen_paraphrase_for_text(text, paraphrase_dict)
        if len(paraphrases):
            fn2paraphrases[fn] = paraphrases

    print("Paraphras-able function size: %d" % len(fn2paraphrases))

    # stats
    for fn, paraphrases in random.sample(fn2paraphrases.items(), 10):
        print("Function: %s" % fn)
        print("Description: %s" % fn2text[fn])
        print("Paraphrases:")
        for para in paraphrases:
            print(para)
        print("-"*10)

    return fn2paraphrases

def main():
    savedir = "user_answers/"
    # # collect equivalent paraphrases
    # ppdb_file = "../../../../../Data/ppdb-2.0-s-all/data"
    # paraphrase_dict = collect_pairs_by_rel(ppdb_file, 'Equivalence')
    # pickle.dump(paraphrase_dict, open("equivalent_paraphrase_dict.pkl", "wb"))

    # # clean ppdb paraphrase data
    # paraphrase_dict = pickle.load(open("equivalent_paraphrase_dict.pkl"))
    # paraphrase_dict_clean = clean_paraphrase(paraphrase_dict)
    # pickle.dump(paraphrase_dict_clean, open("equivalent_paraphrase_dict_clean.pkl", "wb"))

    # # ppdb-based paraphrasing
    # paraphrase_dict = pickle.load(open(savedir + "equivalent_paraphrase_dict_clean.pkl"))
    # fn2text = pickle.load(open(savedir + "action_function2revised_description.pkl"))
    # fn2paraphrase = gen_paraphrase(fn2text, paraphrase_dict)
    # pickle.dump(fn2paraphrase, open(savedir + "action_function2revised_description_paraphrase.pkl", "wb"))


if __name__ == "__main__":
    main()