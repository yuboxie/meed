import os
import pickle
import argparse
import numpy as np
from model import Options, MEED

# Parse the command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type = str, default = '../data/datasets/final/dailydialog',
                    help = 'the directory to the data')
parser.add_argument('--word_embeddings_path', type = str, default = '../data/datasets/final/word_embeddings_dd.npy',
                    help = 'the directory to the pre-trained word embeddings')
parser.add_argument('--num_epochs', type = int, default = 10,
                    help = 'the number of epochs to train the data')
parser.add_argument('--batch_size', type = int, default = 256,
                    help = 'the batch size')
parser.add_argument('--learning_rate', type = float, default = 0.001,
                    help = 'the learning rate')
parser.add_argument('--beam_width', type = int, default = 256,
                    help = 'the beam width when decoding')
parser.add_argument('--word_embed_size', type = int, default = 256,
                    help = 'the size of word embeddings')
parser.add_argument('--emot_input_layer_size', type = int, default = 256,
                    help = 'the size of emotion input layer')
parser.add_argument('--n_hidden_units_enc_s', type = int, default = 256,
                    help = 'the number of encoder hidden units (syntax)')
parser.add_argument('--n_hidden_units_enc_e', type = int, default = 256,
                    help = 'the number of encoder hidden units (emotion)')
parser.add_argument('--n_hidden_units_dec', type = int, default = 256,
                    help = 'the number of decoder hidden units')
parser.add_argument('--n_emot', type = int, default = 6,
                    help = 'number of emotion categories')
parser.add_argument('--word_level_attn_depth', type = int, default = 256,
                    help = 'word-level attention depth')
parser.add_argument('--uttr_level_attn_depth', type = int, default = 128,
                    help = 'uttrance-level attention depth')
parser.add_argument('--beta', type = float, default = 1.0,
                    help = 'trade-off between syntax and emotion')
parser.add_argument('--save_path', type = str, default = 'model_dailydialog',
                    help = 'the path to save the trained model to')
parser.add_argument('--restore_path', type = str, default = 'model_cornell',
                    help = 'the path to restore the trained model')
parser.add_argument('--restore_epoch', type = int, default = 4,
                    help = 'the epoch to restore')

args = parser.parse_args()

def read_data(data_path):
    def load_np_files(path):
        my_set = {}
        my_set['enc_input'] = np.load(os.path.join(path, 'enc_input.npy'))
        my_set['enc_input_e'] = np.load(os.path.join(path, 'enc_input_e.npy'))
        my_set['dec_input'] = np.load(os.path.join(path, 'dec_input.npy'))
        my_set['target'] = np.load(os.path.join(path, 'target.npy'))
        my_set['enc_input_len'] = np.load(os.path.join(path, 'enc_input_len.npy'))
        my_set['dec_input_len'] = np.load(os.path.join(path, 'dec_input_len.npy'))
        my_set['hist_len'] = np.load(os.path.join(path, 'hist_len.npy'))
        return my_set
    train_set = load_np_files(os.path.join(data_path, 'train'))
    valid_set = load_np_files(os.path.join(data_path, 'validation'))
    with open(os.path.join(data_path, '../token2id.pickle'), 'rb') as file:
        token2id = pickle.load(file)
    return train_set, valid_set, token2id

if __name__ == '__main__':
    train_set, valid_set, token2id = read_data(args.data_path)
    max_hist_len = train_set['enc_input'].shape[0]
    max_uttr_len = train_set['enc_input'].shape[2]

    word_embeddings = np.load(args.word_embeddings_path)

    options = Options(mode = 'TRAIN',
                      num_epochs = args.num_epochs,
                      batch_size = args.batch_size,
                      learning_rate = args.learning_rate,
                      beam_width = args.beam_width,
                      vocab_size = len(token2id),
                      max_hist_len = max_hist_len,
                      max_uttr_len = max_uttr_len,
                      go_index = token2id['<go>'],
                      eos_index = token2id['<eos>'],
                      word_embed_size = args.word_embed_size,
                      emot_input_layer_size = args.emot_input_layer_size,
                      n_hidden_units_enc_s = args.n_hidden_units_enc_s,
                      n_hidden_units_enc_e = args.n_hidden_units_enc_e,
                      n_hidden_units_dec = args.n_hidden_units_dec,
                      n_emot = args.n_emot,
                      word_level_attn_depth = args.word_level_attn_depth,
                      uttr_level_attn_depth = args.uttr_level_attn_depth,
                      beta = args.beta,
                      word_embeddings = word_embeddings)
    model = MEED(options)

    for var in model.tvars:
        print(var.name)

    if args.restore_epoch > 0:
        model.restore(os.path.join(args.restore_path, 'model_epoch_{:03d}.ckpt'.format(args.restore_epoch)))
    else:
        model.init_tf_vars()
    model.train(train_set, args.save_path, args.restore_epoch, valid_set = valid_set)
