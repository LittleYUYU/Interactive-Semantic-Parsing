# Interactive Semantic Parsing for If-Then Recipes via Hierarchical Reinforcement Learning

## 1. Introduction
This repository contains source code and dataset for paper "[Interactive Semantic Parsing for If-Then Recipes via Hierarchical Reinforcement Learning](https://arxiv.org/pdf/1808.06740.pdf)" (AAAI'19). Please refer to Table 1 in the paper for an example.

## 2. Dataset
The processed dataset can be found:
- [Full data](data/lam/data.tar.gz) (compressed) for reconstructing the model.  
- [Toy data](data/lam/toy_data_with_noisy_user_ans.pkl) that contains a subset of the full training data, for quick model test.

Data format: [Python Pickle](https://docs.python.org/2/library/pickle.html) files. Please open with `pickle.load(open(filename))`.

Data source: [Training set from (Ur et al., CHI'16)](https://www.blaseur.com/papers/chi16-ifttt.pdf) and [Test set from (Quirk et al., ACL'15)](https://www.microsoft.com/en-us/research/project/language-to-code/).

## 3. Code
All source code is in `cd code/Hierarchical-SP`.

Requirements:
- Python 2.7
- Tensorflow >= 1.4.0

### 3.1 Agent training/testing
To train the HRL agent:
```
python run.py --train --training_stage=0
```

To train the HRL_fixedOrder agent:
```
python run.py --train --training_state=1
```

- For testing them on the test set, replace `--train` with `--test`.
- To quick test on the toy dataset, append `--toy_data`.

To interactively test the four agents {HRL, HRL_fixedOrder, LAM_sup, LAM_rule}:
```
python interactive_test.py --level='VI-3' --user-name=yourname
```

### 3.2 User simulator
Please refer to the paper appendix for more details.
The scripts for [PPDB paraphrasing](code/Hierarchical-SP/ppdb.py) and [collecting from user data or official function descriptions](code/Hierarchical-SP/user_simulator_gen.py) can be found.


## 4. Citation
Please kindly cite the following paper if you use the code or the dataset in this repo:
```
@inproceedings{yao2019interactive,
  title={Interactive Semantic Parsing for If-Then Recipes via Hierarchical Reinforcement Learning},
  author={Yao, Ziyu and Li, Xiujun and Gao, Jianfeng and Sadler, Brian and Sun, Huan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={2547--2554},
  year={2019}
}
```
