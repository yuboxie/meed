# Multi-Turn Emotionally Engaging Dialog Model (MEED)

Code for paper
> Yubo Xie, Ekaterina Svikhnushina, and Pearl Pu. A Multi-Turn Emotionally Engaging Dialog Model.
> IUI 2020 Workshop on User-Aware Conversational Agents. [PDF Link](https://arxiv.org/pdf/1908.07816.pdf).

## Environment
The project was developed under the following environment:
- TensorFlow 1.12.0
- NumPy 1.15.4
- spaCy 2.0.18

## Files

List of files and their descriptions:

- `model.py`: implementation of the model;
- `attention.py`: implementation of the hierarchical attention mechanism;
- `attn_wrapper.py`: wrap the attention mechanism into the RNN cells;
- `train.py`: train the model on the training set;
- `validate.py`: evaluate the model on the validation set (choosing hyperparameters);
- `predict.py`: evaluate the model on the test set (predict the responses).
