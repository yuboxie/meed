import os
import spacy
import pickle
import argparse
import numpy as np
from model import Options, MEED

nlp = spacy.load('en_core_web_sm')

# Parse the command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type = str, default = '../data/datasets/final/dailydialog_test',
                    help = 'the directory to the data')
parser.add_argument('--word_embeddings_path', type = str, default = '../data/datasets/final/word_embeddings_dd.npy',
                    help = 'the directory to the pre-trained word embeddings')

parser.add_argument('--num_epochs', type = int, default = 10,
                    help = 'the number of epochs to train the data')
parser.add_argument('--batch_size', type = int, default = 1,
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

parser.add_argument('--restore_path', type = str, default = 'model_dailydialog',
                    help = 'the path to restore the trained model')
parser.add_argument('--restore_epoch', type = int, default = 8,
                    help = 'the epoch to restore')

parser.add_argument('--restore_path_r', type = str, default = 'model_dailydialog_r',
                    help = 'the path to restore the trained reversed model')
parser.add_argument('--restore_epoch_r', type = int, default = 7,
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
    test_set = load_np_files(data_path)
    with open(os.path.join(data_path, '../token2id.pickle'), 'rb') as file:
        token2id = pickle.load(file)
    with open(os.path.join(data_path, '../id2token.pickle'), 'rb') as file:
        id2token = pickle.load(file)
    with open(os.path.join(data_path, '../id2evec.pickle'), 'rb') as file:
        id2evec = pickle.load(file)
    return test_set, token2id, id2token, id2evec

def ids_to_sentence(ids, uttr_len, id2token):
    tokens = []
    if uttr_len is not None:
        for i in range(uttr_len):
            if id2token[ids[i]] != '<eos>' and id2token[ids[i]] != '<go>':
                tokens.append(id2token[ids[i]])
    else:
        i = 0
        while i < len(ids) and id2token[ids[i]] != '<eos>':
            tokens.append(id2token[ids[i]])
            i += 1
    return ' '.join(tokens)

def get_emotion_score(text, token2vad):
    n = 0
    score = 0.0
    doc = nlp(text)
    for t in doc:
        if not t.is_space:
            if t.lemma_ in token2vad:
                vad_diff = token2vad[t.lemma_] - np.array([5.0, 1.0, 5.0])
                score += np.linalg.norm(vad_diff)
                n += 1
    if n > 0:
        score = score / n
    return score

def build_emot_vec(pred_id, pred_len, id2evec):
    beam_width = pred_id.shape[0]
    n_emot = id2evec[0].shape[0]
    enc_input_e = np.zeros((beam_width, n_emot), np.float32)
    for i in range(beam_width):
        emot_vec = np.zeros((n_emot), np.float32)
        for j in range(pred_len[i] - 1):
            emot_vec += id2evec[pred_id[i,j]]
        enc_input_e[i,:] = emot_vec
    enc_input_e[enc_input_e > 0] = 1.0
    enc_input_e_part_sum = np.sum(enc_input_e[:,:n_emot-1], axis = 1)
    enc_input_e_part_sum[enc_input_e_part_sum > 0] = 1.0
    enc_input_e[:,-1] = 1.0 - enc_input_e_part_sum
    return enc_input_e

def build_reversed_data_set(n, enc_input, enc_input_e, enc_input_len, hist_len, pred_id, pred_len, id2evec, token2id):
    """
        n: how many utterances to take from enc_input, 0 <= n <= hist_len - 1
        enc_input: [max_hist_len, 1, max_uttr_len]
        enc_input_e: [1, max_hist_len, n_emot]
        enc_input_len: [max_hist_len, 1]
        hist_len: int scalar
        pred_id: [beam_width, max_time], max_time <= max_uttr_len + 1
        pred_len: [beam_width]
    """
    max_hist_len = enc_input.shape[0]
    max_uttr_len = enc_input.shape[2]
    n_emot = enc_input_e.shape[2]
    batch_size = pred_id.shape[0]  # batch_size == beam_width
    max_time = pred_id.shape[1]

    my_set = {}
    my_set['enc_input'] = np.zeros((max_hist_len, batch_size, max_uttr_len), np.int32)
    my_set['enc_input_e'] = np.zeros((batch_size, max_hist_len, n_emot), np.float32)
    my_set['dec_input'] = np.zeros((batch_size, max_uttr_len + 1), np.int32)
    my_set['target'] = np.zeros((batch_size, max_uttr_len + 1), np.int32)
    my_set['enc_input_len'] = np.zeros((max_hist_len, batch_size), np.int32)

    my_set['hist_len'] = np.tile(np.array([n + 1], np.int32), batch_size)

    # Set prediction as the first utterance in the history
    my_set['enc_input'][max_hist_len-n-1,:,:max_time-1] = pred_id[:,:-1]
    my_set['enc_input_len'][max_hist_len-n-1,:] = pred_len - 1  # exclude the <eos> token
    my_set['enc_input_e'][:,0,:] = build_emot_vec(pred_id, pred_len, id2evec)

    # Take the last n utterances from original enc_input
    for i in range(n):
        my_set['enc_input'][max_hist_len-n+i,:,:] = np.tile(enc_input[-(i+1),:,:], (batch_size, 1))
        my_set['enc_input_len'][max_hist_len-n+i,:] = np.tile(enc_input_len[-(i+1),:], batch_size)
        my_set['enc_input_e'][:,i+1,:] = np.tile(enc_input_e[:,hist_len-(i+1),:], (batch_size, 1))

    # Use the next utterance in history as response
    uttr_len = enc_input_len[-(n+1),0]
    my_set['dec_input'][:,0] = np.ones(batch_size, np.int32) * token2id['<go>']
    my_set['dec_input'][:,1:] = np.tile(enc_input[-(n+1),:,:], (batch_size, 1))
    my_set['target'][:,:uttr_len] = np.tile(enc_input[-(n+1),:,:uttr_len], (batch_size, 1))
    my_set['target'][:,uttr_len] = np.ones(batch_size, np.int32) * token2id['<eos>']
    my_set['dec_input_len'] = np.tile(np.array([uttr_len + 1], np.int32), batch_size)

    return my_set

if __name__ == '__main__':
    test_set, token2id, id2token, id2evec = read_data(args.data_path)
    max_hist_len = test_set['enc_input'].shape[0]
    max_uttr_len = test_set['enc_input'].shape[2]
    N = test_set['enc_input'].shape[1]

    word_embeddings = np.load(args.word_embeddings_path)

    options = Options(mode = 'PREDICT',
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
    model.restore(os.path.join(args.restore_path, 'model_epoch_{:03d}.ckpt'.format(args.restore_epoch)))

    prediction, scores, uttr_level_alignments, word_level_alignments, final_sequence_lengths = model.predict(
        test_set['enc_input'], test_set['enc_input_e'], test_set['enc_input_len'], test_set['hist_len'])

    print(prediction[0].shape)
    print(scores[0].shape)
    print(uttr_level_alignments[0].shape)
    print(word_level_alignments[0].shape)
    print(final_sequence_lengths[0].shape)

    pred_ids = []
    pred_scores = []
    pred_lens = []
    for i in range(N):
        n_batch = i // args.batch_size
        n_example = i % args.batch_size
        pred_id = np.transpose(prediction[n_batch][n_example,:,:])  # [beam_width, max_time]
        pred_score = np.transpose(scores[n_batch][n_example,:,:])  # [beam_width, max_time]
        pred_len = final_sequence_lengths[n_batch][n_example,:]  # [beam_width]
        pred_ids.append(pred_id)
        pred_scores.append(pred_score)
        pred_lens.append(pred_len)

    # get scores of the prediction from the normal dialog model
    del model
    options.mode = 'TRAIN'
    options.batch_size = options.beam_width
    model = MEED(options)
    model.restore(os.path.join(args.restore_path, 'model_epoch_{:03d}.ckpt'.format(args.restore_epoch)))

    scores = []
    for i in range(N):
        my_set = {}
        my_set['enc_input'] = np.tile(test_set['enc_input'][:,i:i+1,:], (1, options.batch_size, 1))
        my_set['enc_input_e'] = np.tile(test_set['enc_input_e'][i:i+1,:,:], (options.batch_size, 1, 1))
        my_set['dec_input'] = np.zeros((options.batch_size, options.max_uttr_len + 1), np.int32)
        my_set['dec_input'][:,:pred_ids[i].shape[1]] = np.concatenate((np.ones((options.batch_size, 1), np.int32) * token2id['<go>'], pred_ids[i][:,:-1]), axis = 1)
        my_set['target'] = np.zeros((options.batch_size, options.max_uttr_len + 1), np.int32)
        my_set['target'][:,:pred_ids[i].shape[1]] = pred_ids[i]
        my_set['enc_input_len'] = np.tile(test_set['enc_input_len'][:,i:i+1], (1, options.batch_size))
        my_set['dec_input_len'] = pred_lens[i]
        my_set['hist_len'] = np.tile(test_set['hist_len'][i:i+1], options.batch_size)
        loss_batch = model.validate_batch(my_set)
        scores.append(loss_batch * pred_lens[i])

    # get scores of the prediction from the reversed dialog model
    del model
    model = MEED(options)
    model.restore(os.path.join(args.restore_path_r, 'model_epoch_{:03d}.ckpt'.format(args.restore_epoch_r)))

    lambda_ = 0.75
    tau = 2.0

    scores_r = []
    for i in range(N):
        hist_len = test_set['hist_len'][i]
        score = np.zeros(options.beam_width, np.float32)
        my_lambda_ = lambda_
        for n in range(hist_len):
            my_set = build_reversed_data_set(n, test_set['enc_input'][:,i:i+1,:],
                test_set['enc_input_e'][i:i+1,:,:], test_set['enc_input_len'][:,i:i+1],
                test_set['hist_len'][i], pred_ids[i], pred_lens[i], id2evec, token2id)
            loss_batch = model.validate_batch(my_set)
            score += my_lambda_ * loss_batch * my_set['dec_input_len']
            my_lambda_ = my_lambda_ ** tau
        scores_r.append(score)

    with open('../data/corpus/token2vad.pickle', 'rb') as f:
        token2vad = pickle.load(f)
    scores_e = []
    responses = []
    for i in range(N):
        response = []
        score = []
        for j in range(args.beam_width):
            text = ids_to_sentence(pred_ids[i][j,:], None, id2token)
            response.append(text)
            score.append(get_emotion_score(text, token2vad))
        score = np.array(score, np.float32)
        print(i)
        scores_e.append(score)
        responses.append(response)

    pred_MEED = []
    pred_MEED_re = []
    pred_MEED_re_e = []
    f = open('pred_MEED.txt', 'w', encoding = 'utf-8')
    for i in range(N):
        new_scores = (1 - lambda_) * scores[i] + scores_r[i]
        new_scores_sorted = sorted(list(zip(range(options.beam_width), new_scores)), key = lambda x: x[1])

        new_scores_e = (1 - lambda_) * scores[i] + scores_r[i] + 0.2 * scores_e[i]
        new_scores_e_sorted = sorted(list(zip(range(options.beam_width), new_scores_e)), key = lambda x: x[1])

        len_h = test_set['hist_len'][i]
        f.write('HISTORY:\n')
        for j in range(len_h):
            uttr = ids_to_sentence(test_set['enc_input'][j+max_hist_len-len_h,i,:], test_set['enc_input_len'][j+max_hist_len-len_h,i], id2token)
            f.write('- {}\n'.format(uttr))
        label = ids_to_sentence(test_set['target'][i,:], test_set['dec_input_len'][i], id2token)
        f.write('(- {})\n'.format(label))

        response = responses[i][0]
        f.write('MEED: {}\n'.format(response))
        pred_MEED.append(response)

        idx, new_score = new_scores_sorted[0]
        response = responses[i][idx]
        f.write('MEED_RE: {}\n'.format(response))
        pred_MEED_re.append(response)

        idx, new_score = new_scores_e_sorted[0]
        response = responses[i][idx]
        f.write('MEED_RE_E: {}\n\n'.format(response))
        pred_MEED_re_e.append(response)
    f.close()
    with open('pred_MEED.pickle', 'wb') as f:
        pickle.dump(pred_MEED, f)
    with open('pred_MEED_re.pickle', 'wb') as f:
        pickle.dump(pred_MEED_re, f)
    with open('pred_MEED_re_e.pickle', 'wb') as f:
        pickle.dump(pred_MEED_re_e, f)
