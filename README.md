# Multi-Turn Emotionally Engaging Dialog Model (MEED)

[![DOI](https://zenodo.org/badge/209135468.svg)](https://zenodo.org/badge/latestdoi/209135468)

Code for paper
> Yubo Xie, Ekaterina Svikhnushina, and Pearl Pu. A Multi-Turn Emotionally Engaging Dialog Model.
> IUI 2020 Workshop on User-Aware Conversational Agents. [PDF Link](https://arxiv.org/pdf/1908.07816.pdf).

## Environment
The project was developed using the following packages:
- TensorFlow 1.12.0
- NumPy 1.15.4
- spaCy 2.0.18

## Files
- `model.py`: implementation of the model;
- `attention.py`: implementation of the hierarchical attention mechanism;
- `attn_wrapper.py`: wrap the attention mechanism into the RNN cells;
- `train.py`: train the model on the training set;
- `validate.py`: evaluate the model on the validation set (choosing hyperparameters);
- `predict.py`: evaluate the model on the test set (predict the responses).

## Training Data
The training data (tokenized) can be found [here](https://drive.google.com/drive/folders/1mDV4nsN6x8Fw8j1SbLLp0mzLt3ytZWoR?usp=sharing).

## License
See the [LICENSE](LICENSE) file in the root repo folder for more details.
