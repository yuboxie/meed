import os
import numpy as np
import tensorflow as tf
from attention import UttrLevelAttentionMechanism
from attn_wrapper import MyAttentionWrapper
from tensorflow.contrib.seq2seq import BeamSearchDecoder, dynamic_decode, sequence_loss, tile_batch

class Options(object):
    '''Parameters used by the MEED model.'''
    def __init__(self, mode, num_epochs, batch_size, learning_rate, beam_width,
                 vocab_size, max_hist_len, max_uttr_len, go_index, eos_index,
                 word_embed_size, emot_input_layer_size, n_hidden_units_enc_s,
                 n_hidden_units_enc_e, n_hidden_units_dec, n_emot,
                 word_level_attn_depth, uttr_level_attn_depth,
                 beta, word_embeddings):
        super(Options, self).__init__()

        self.mode = mode
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beam_width = beam_width

        self.vocab_size = vocab_size
        self.max_hist_len = max_hist_len
        self.max_uttr_len = max_uttr_len
        self.go_index = go_index
        self.eos_index = eos_index

        self.word_embed_size = word_embed_size
        self.emot_input_layer_size = emot_input_layer_size
        self.n_hidden_units_enc_s = n_hidden_units_enc_s
        self.n_hidden_units_enc_e = n_hidden_units_enc_e
        self.n_hidden_units_dec = n_hidden_units_dec
        self.n_emot = n_emot

        self.word_level_attn_depth = word_level_attn_depth
        self.uttr_level_attn_depth = uttr_level_attn_depth

        self.beta = beta
        self.word_embeddings = word_embeddings

class MEED(object):
    '''Multi-turn emotionally engaging dialog generation.'''
    def __init__(self, options):
        super(MEED, self).__init__()
        self.options = options
        self.build_graph()
        self.session = tf.Session(graph = self.graph)

    def __del__(self):
        self.session.close()
        print('TensorFlow session is closed.')

    def build_graph(self):
        print('Building the TensorFlow graph...')
        opts = self.options

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.enc_input = tf.placeholder(tf.int32, shape = [opts.max_hist_len, opts.batch_size, opts.max_uttr_len])
            self.enc_input_e = tf.placeholder(tf.float32, shape = [opts.batch_size, opts.max_hist_len, opts.n_emot])
            self.dec_input = tf.placeholder(tf.int32, shape = [opts.batch_size, opts.max_uttr_len + 1])
            self.target = tf.placeholder(tf.int32, shape = [opts.batch_size, opts.max_uttr_len + 1])

            self.enc_input_len = tf.placeholder(tf.int32, shape = [opts.max_hist_len, opts.batch_size])
            self.dec_input_len = tf.placeholder(tf.int32, shape = [opts.batch_size])
            self.hist_len = tf.placeholder(tf.int32, shape = [opts.batch_size])

            with tf.variable_scope('embedding', reuse = tf.AUTO_REUSE):
                # word_embeddings = tf.Variable(tf.random_uniform([opts.vocab_size, opts.word_embed_size], -1.0, 1.0),
                #     name = 'word_embeddings')
                word_embeddings = tf.Variable(opts.word_embeddings, name = 'word_embeddings')
                enc_input_embed = tf.nn.embedding_lookup(word_embeddings, self.enc_input)
                dec_input_embed = tf.nn.embedding_lookup(word_embeddings, self.dec_input)

            with tf.variable_scope('word_level_encoding', reuse = tf.AUTO_REUSE):
                outputs_enc = []
                cell_fw = tf.nn.rnn_cell.GRUCell(opts.n_hidden_units_enc_s)
                cell_bw = tf.nn.rnn_cell.GRUCell(opts.n_hidden_units_enc_s)
                for i in range(opts.max_hist_len):
                    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                 inputs = enc_input_embed[i,:,:,:],
                                                                 sequence_length = self.enc_input_len[i,:],
                                                                 dtype = tf.float32)
                    outputs_enc.append(tf.concat(outputs, 2))
                outputs_enc = tf.stack(outputs_enc)

            with tf.variable_scope('emotion_encoding', reuse = tf.AUTO_REUSE):
                emot_input_layer = tf.layers.Dense(opts.emot_input_layer_size, activation = tf.sigmoid,
                    kernel_initializer = tf.truncated_normal_initializer(stddev = 0.1), name = 'emot_input_layer')
                enc_input_e = emot_input_layer(self.enc_input_e)

                cell_emot = tf.nn.rnn_cell.GRUCell(opts.n_hidden_units_enc_e)
                _, final_state = tf.nn.dynamic_rnn(cell_emot,
                    inputs = enc_input_e,
                    sequence_length = self.hist_len,
                    dtype = tf.float32)
                emot_vector = final_state * opts.beta

            if opts.mode == 'PREDICT':
                outputs_enc = tf.transpose(outputs_enc, perm = [1,0,2,3])
                outputs_enc = tile_batch(outputs_enc, multiplier = opts.beam_width)
                outputs_enc = tf.transpose(outputs_enc, perm = [1,0,2,3])
                tiled_enc_input_len = tile_batch(tf.transpose(self.enc_input_len), multiplier = opts.beam_width)
                tiled_enc_input_len = tf.transpose(tiled_enc_input_len)
                tiled_hist_len = tile_batch(self.hist_len, multiplier = opts.beam_width)
                tiled_emot_vector = tile_batch(emot_vector, multiplier = opts.beam_width)
            else:
                tiled_enc_input_len = self.enc_input_len
                tiled_hist_len = self.hist_len
                tiled_emot_vector = emot_vector

            with tf.variable_scope('decoding', reuse = tf.AUTO_REUSE) as vs:
                attn_mechanism = UttrLevelAttentionMechanism(word_level_num_units = opts.word_level_attn_depth,
                    uttr_level_num_units = opts.uttr_level_attn_depth,
                    n_hidden_units = opts.n_hidden_units_enc_s,
                    memory = outputs_enc,
                    memory_sequence_length = tiled_enc_input_len,
                    hist_length = tiled_hist_len)
                cell_dec = tf.nn.rnn_cell.GRUCell(opts.n_hidden_units_dec)
                cell_dec = MyAttentionWrapper(cell_dec, attn_mechanism, tiled_emot_vector)
                output_layer = tf.layers.Dense(units = opts.vocab_size - 1,
                    kernel_initializer = tf.truncated_normal_initializer(stddev = 0.1), name = 'output_layer')

                # Train
                if opts.mode == 'TRAIN':
                    outputs_dec, _ = tf.nn.dynamic_rnn(cell = cell_dec,
                        inputs = dec_input_embed,
                        sequence_length = self.dec_input_len,
                        initial_state = cell_dec.zero_state(opts.batch_size, tf.float32),
                        dtype = tf.float32,
                        scope = vs)
                    logits = output_layer.apply(outputs_dec)
                    weights = tf.sequence_mask(self.dec_input_len,
                        maxlen = opts.max_uttr_len + 1,
                        dtype = tf.float32)
                    self.loss = sequence_loss(logits, self.target, weights)
                    self.loss_batch = sequence_loss(logits, self.target, weights, average_across_batch = False)
                    self.optimizer = tf.train.AdamOptimizer(opts.learning_rate).minimize(self.loss)
                    self.init = tf.global_variables_initializer()

                # Predict
                if opts.mode == 'PREDICT':
                    start_tokens = tf.constant(opts.go_index, dtype = tf.int32, shape = [opts.batch_size])
                    bs_decoder = BeamSearchDecoder(cell = cell_dec, embedding = word_embeddings,
                        start_tokens = start_tokens,
                        end_token = opts.eos_index,
                        initial_state = cell_dec.zero_state(opts.batch_size * opts.beam_width, tf.float32),
                        beam_width = opts.beam_width,
                        output_layer = output_layer)
                    final_outputs, final_state, _ = dynamic_decode(bs_decoder,
                        impute_finished = False,
                        maximum_iterations = opts.max_uttr_len + 1,
                        scope = vs)
                    self.predicted_ids = final_outputs.predicted_ids
                    self.scores = final_outputs.beam_search_decoder_output.scores
                    self.uttr_level_alignments = final_state[0].alignment_history_ul.stack()
                    self.word_level_alignments = final_state[0].alignment_history_wl.stack()
                    self.final_sequence_lengths = final_state[3]

            self.tvars = tf.trainable_variables()
            self.saver = tf.train.Saver(max_to_keep = 100)

    def init_tf_vars(self):
        if self.options.mode == 'TRAIN':
            self.session.run(self.init)
            print('TensorFlow variables initialized.')

    def validate(self, valid_set):
        """Validate the model on the validation set.
        Args:
            valid_set: Dictionary containing:
                enc_input: Input to the word-level encoders (syntax).
                    Shaped `[max_hist_len, N, max_uttr_len]`.
                enc_input_e: Input to the word-level encoders (emotion).
                    Shaped `[N, max_hist_len, n_emot]`.
                dec_input: Input to the decoder. Shaped `[N, max_uttr_len]`.
                target: Targets, expected output of the decoder. Shaped `[N, max_uttr_len]`.
                enc_input_len: Lengths of the input to the word-level encoders. Shaped `[max_hist_len, N]`.
                dec_input_len: Lengths of the input to the decoder. Shaped `[N]`.
                hist_len: Lengths of the conversation history. Shaped `[N]`.
                (N should be a multiple of batch_size)
        Returns:
            perplexity: Perplexity on the validation set.
        """
        opts = self.options
        num_examples = valid_set['enc_input'].shape[1]
        num_batches = num_examples // opts.batch_size
        loss = 0.0
        for batch in range(num_batches):
            s = batch * opts.batch_size
            t = s + opts.batch_size
            feed_dict = {self.enc_input: valid_set['enc_input'][:,s:t,:],
                         self.enc_input_e: valid_set['enc_input_e'][s:t,:,:],
                         self.dec_input: valid_set['dec_input'][s:t,:],
                         self.target: valid_set['target'][s:t,:],
                         self.enc_input_len: valid_set['enc_input_len'][:,s:t],
                         self.dec_input_len: valid_set['dec_input_len'][s:t],
                         self.hist_len: valid_set['hist_len'][s:t]}
            loss_val = self.session.run(self.loss, feed_dict = feed_dict)
            loss += loss_val
        return np.exp(loss / num_batches)

    def validate_batch(self, valid_set):
        feed_dict = {self.enc_input: valid_set['enc_input'],
                     self.enc_input_e: valid_set['enc_input_e'],
                     self.dec_input: valid_set['dec_input'],
                     self.target: valid_set['target'],
                     self.enc_input_len: valid_set['enc_input_len'],
                     self.dec_input_len: valid_set['dec_input_len'],
                     self.hist_len: valid_set['hist_len']}
        loss_batch_val = self.session.run(self.loss_batch, feed_dict = feed_dict)
        return loss_batch_val

    def train(self, train_set, save_path, restore_epoch, valid_set = None):
        """Train the model.
        Args:
            train_set and valid_set: Dictionaries containing:
                enc_input: Input to the word-level encoders (syntax).
                    Shaped `[max_hist_len, N, max_uttr_len]`.
                enc_input_e: Input to the word-level encoders (emotion).
                    Shaped `[N, max_hist_len, n_emot]`.
                dec_input: Input to the decoder. Shaped `[N, max_uttr_len]`.
                target: Targets, expected output of the decoder. Shaped `[N, max_uttr_len]`.
                enc_input_len: Lengths of the input to the word-level encoders. Shaped `[max_hist_len, N]`.
                dec_input_len: Lengths of the input to the decoder. Shaped `[N]`.
                hist_len: Lengths of the conversation history. Shaped `[N]`.
        """
        print('Start to train the model...')
        opts = self.options

        num_examples = train_set['enc_input'].shape[1]
        num_batches = num_examples // opts.batch_size
        valid_ppl = [None]

        for epoch in range(opts.num_epochs):
            perm_indices = np.random.permutation(range(num_examples))
            for batch in range(num_batches):
                s = batch * opts.batch_size
                t = s + opts.batch_size
                batch_indices = perm_indices[s:t]
                feed_dict = {self.enc_input: train_set['enc_input'][:,batch_indices,:],
                             self.enc_input_e: train_set['enc_input_e'][batch_indices,:,:],
                             self.dec_input: train_set['dec_input'][batch_indices,:],
                             self.target: train_set['target'][batch_indices,:],
                             self.enc_input_len: train_set['enc_input_len'][:,batch_indices],
                             self.dec_input_len: train_set['dec_input_len'][batch_indices],
                             self.hist_len: train_set['hist_len'][batch_indices]}
                _, loss_val = self.session.run([self.optimizer, self.loss], feed_dict = feed_dict)
                print('Epoch {:03d}/{:03d}, valid ppl = {}, batch {:04d}/{:04d}, train loss = {}'.format(epoch + 1,
                    opts.num_epochs, valid_ppl[-1], batch + 1, num_batches, loss_val), flush = True)

            if valid_set is not None:
                ppl = self.validate(valid_set)
                valid_ppl.append(ppl)
            self.save(os.path.join(save_path, 'model_epoch_{:03d}.ckpt'.format(restore_epoch + epoch + 1)))

        if valid_set is not None:
            for epoch in range(opts.num_epochs):
                print('Epoch {:03d}, valid ppl = {}'.format(epoch + 1, valid_ppl[epoch + 1]))
            print('Beta = {}, lowest ppl = {}'.format(opts.beta, np.min(valid_ppl[1:])))

    def train_to_tune(self, train_set, valid_set, save_path):
        print('Start to train the model...')
        opts = self.options

        num_examples = train_set['enc_input'].shape[1]
        num_batches = num_examples // opts.batch_size
        valid_ppl = [None]

        for epoch in range(opts.num_epochs):
            perm_indices = np.random.permutation(range(num_examples))
            for batch in range(num_batches):
                s = batch * opts.batch_size
                t = s + opts.batch_size
                batch_indices = perm_indices[s:t]
                feed_dict = {self.enc_input: train_set['enc_input'][:,batch_indices,:],
                             self.enc_input_e: train_set['enc_input_e'][batch_indices,:,:],
                             self.dec_input: train_set['dec_input'][batch_indices,:],
                             self.target: train_set['target'][batch_indices,:],
                             self.enc_input_len: train_set['enc_input_len'][:,batch_indices],
                             self.dec_input_len: train_set['dec_input_len'][batch_indices],
                             self.hist_len: train_set['hist_len'][batch_indices]}
                _, loss_val = self.session.run([self.optimizer, self.loss], feed_dict = feed_dict)
                print('Epoch {:03d}/{:03d}, valid ppl = {}, batch {:04d}/{:04d}, train loss = {}'.format(epoch + 1,
                    opts.num_epochs, valid_ppl[-1], batch + 1, num_batches, loss_val), flush = True)

            ppl = self.validate(valid_set)
            valid_ppl.append(ppl)
            self.save(os.path.join(save_path, 'model_epoch_{:03d}.ckpt'.format(epoch + 1)))

        return valid_ppl[1:]

    def predict(self, enc_input, enc_input_e, enc_input_len, hist_len):
        """Predict the response based on the input.
        Args:
            enc_input: Input to the word-level encoders (syntax).
                Shaped `[max_hist_len, N, max_uttr_len]`.
            enc_input_e: Input to the word-level encoders (emotion).
                Shaped `[N, max_hist_len, n_emot]`.
            enc_input_len: Lengths of the input to the word-level encoders. Shaped `[max_hist_len, N]`.
            hist_len: Lengths of the conversation history. Shaped `[N]`.
            (N should be a multiple of batch_size)
        Returns:
            prediction: Predicted word indices. Shaped `[N, max_uttr_len, beam_width]`.
        """
        opts = self.options
        num_examples = enc_input.shape[1]
        num_batches = num_examples // opts.batch_size
        prediction = []
        scores = []
        uttr_level_alignments = []
        word_level_alignments = []
        final_sequence_lengths = []
        for batch in range(num_batches):
            s = batch * opts.batch_size
            t = s + opts.batch_size
            feed_dict = {self.enc_input: enc_input[:,s:t,:],
                         self.enc_input_e: enc_input_e[s:t,:,:],
                         self.enc_input_len: enc_input_len[:,s:t],
                         self.hist_len: hist_len[s:t]}
            p, s, u, w, fsl = self.session.run([self.predicted_ids, self.scores,
                self.uttr_level_alignments, self.word_level_alignments,
                self.final_sequence_lengths], feed_dict = feed_dict)
            prediction.append(p)
            scores.append(s)
            uttr_level_alignments.append(u)
            word_level_alignments.append(w)
            final_sequence_lengths.append(fsl)
        return prediction, scores, uttr_level_alignments, word_level_alignments, final_sequence_lengths

    def save(self, save_path):
        print('Saving the trained model to {}...'.format(save_path))
        self.saver.save(self.session, save_path)

    def restore(self, restore_path):
        print('Restoring a pre-trained model from {}...'.format(restore_path))
        self.saver.restore(self.session, restore_path)
