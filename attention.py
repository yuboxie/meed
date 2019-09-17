import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import AttentionMechanism, _BaseAttentionMechanism, _maybe_mask_score, _prepare_memory

_zero_state_tensors = rnn_cell_impl._zero_state_tensors

def _maybe_mask_score_no_check(score, memory_sequence_length, score_mask_value):
    if memory_sequence_length is None:
        return score
    score_mask = tf.sequence_mask(memory_sequence_length, maxlen = tf.shape(score)[1])
    score_mask_values = score_mask_value * tf.ones_like(score)
    return tf.where(score_mask, score, score_mask_values)

def _maybe_mask_score_softmax(score, memory_sequence_length):
    if memory_sequence_length is None:
        return score
    score_mask = tf.sequence_mask(memory_sequence_length, maxlen = tf.shape(score)[1])
    score_mask_values = 0.0 * tf.ones_like(score)
    return tf.where(score_mask, score, score_mask_values)

class WordLevelAttentionMechanism(_BaseAttentionMechanism):
    """Word-level attention mechanism."""
    def __init__(self,
                 attention_v,
                 dec_query_layer,
                 enc_query_layer,
                 memory_layer,
                 memory,
                 memory_sequence_length = None,
                 probability_fn = None,
                 score_mask_value = None,
                 check_inner_dims_defined = True):
        """Construct the word level attention mechanism.
        Args:
            attention_v: The attention v variable.
            dec_query_layer: Mapping layer for decoder's query.
            enc_query_layer: Mapping layer for utterance-level encoder's query.
            memory_layer: Mapping layer for memory.
            memory: The memory to query; the output of a bidirectional RNN.  This
                tensor should be shaped `[batch_size, max_uttr_len, 2*n_hidden_units]`.
            memory_sequence_length (optional): Sequence lengths for the batch entries
                in memory.  If provided, the memory tensor rows are masked with zeros
                for values past the respective sequence lengths.
            probability_fn: (optional) A `callable`.  Converts the score to
                probabilities.  The default is `tf.nn.softmax`. Other options include
                `tf.contrib.seq2seq.hardmax` and `tf.contrib.sparsemax.sparsemax`.
                Its signature should be: `probabilities = probability_fn(score)`.
            score_mask_value: (optional): The mask value for score before passing into
                `probability_fn`. The default is -inf. Only used if
                `memory_sequence_length` is not None.
            check_inner_dims_defined: Python boolean.  If `True`, the `memory`
                argument's shape is checked to ensure all but the two outermost
                dimensions are fully defined.
        """
        # super(WordLevelAttentionMechanism, self).__init__(
        #     query_layer = None,
        #     memory_layer = memory_layer,
        #     memory = memory,
        #     probability_fn = wrapped_probability_fn,
        #     memory_sequence_length = memory_sequence_length)

        # Use custom initialization due to the probable zero values in memory_sequence_length
        if probability_fn is None:
            probability_fn = tf.nn.softmax
        if (memory_layer is not None and not isinstance(memory_layer, tf.layers.Layer)):
            raise TypeError("memory_layer is not a Layer: %s" % type(memory_layer).__name__)
        self._query_layer = None
        self._memory_layer = memory_layer
        self.dtype = memory_layer.dtype
        if not callable(probability_fn):
            raise TypeError("probability_fn must be callable, saw type: %s" % type(probability_fn).__name__)
        if score_mask_value is None:
            score_mask_value = tf.dtypes.as_dtype(self._memory_layer.dtype).as_numpy_dtype(-np.inf)
        self._probability_fn = lambda score: _maybe_mask_score_softmax(
            probability_fn(_maybe_mask_score_no_check(score, memory_sequence_length, score_mask_value)),
            memory_sequence_length)
        with tf.name_scope(None, "BaseAttentionMechanismInit", tf.contrib.framework.nest.flatten(memory)):
            self._values = _prepare_memory(
                memory, memory_sequence_length,
                check_inner_dims_defined = check_inner_dims_defined)
            self._keys = (
                self.memory_layer(self._values) if self.memory_layer
                else self._values)
            self._batch_size = (self._keys.shape[0].value or tf.shape(self._keys)[0])
            self._alignments_size = (self._keys.shape[1].value or tf.shape(self._keys)[1])

        # Extra initialization
        self._dec_query_layer = dec_query_layer
        self._enc_query_layer = enc_query_layer
        self._attention_v = attention_v

    def __call__(self, dec_query, enc_query):
        """Score the query based on the keys and values.
        Args:
            dec_query: Tensor of dtype matching `self.values` and shape
                `[batch_size, query_depth]`.
            enc_query: Tensor of dtype matching `self.values` and shape
                `[batch_size, query_depth]`.
        Returns:
            alignments: Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size]` (`alignments_size` is memory's
                `max_time`).
        """
        with tf.variable_scope('word_level_attention'):
            processed_dec_query = self._dec_query_layer(dec_query)
            processed_enc_query = self._enc_query_layer(enc_query)
            processed_dec_query = tf.expand_dims(processed_dec_query, 1)
            processed_enc_query = tf.expand_dims(processed_enc_query, 1)
            # v = tf.get_variable('attention_v', [self._num_units], dtype = tf.float32)
            score = tf.reduce_sum(self._attention_v * tf.tanh(self._keys + processed_dec_query + processed_enc_query), [2])
            alignments = self._probability_fn(score)
        return alignments

class UttrLevelAttentionMechanism(AttentionMechanism):
    """Utterance-level attention mechanism."""
    def __init__(self,
                 word_level_num_units,
                 uttr_level_num_units,
                 n_hidden_units,
                 memory,
                 memory_sequence_length,
                 hist_length):
        """Construct the utterance level attention mechanism.
        Args:
            word_level_num_units: Word level attention depth.
            uttr_level_num_units: Utterance level attention depth.
            n_hidden_units: Number of hidden units for utterance-level encoder.
            memory: The memory to query; the output of the bidirectional RNNs.  This
                tensor should be shaped `[max_hist_len, batch_size, max_uttr_len, 2*n_hidden_units]`.
            memory_sequence_length: Sequence lengths for the batch entries in memory.
                Shaped `[max_hist_len, batch_size]`.
            hist_length: Lengths for the utterances in history. Shaped `[batch_size]`.
        """
        self._query_layer = tf.layers.Dense(uttr_level_num_units,
            name = 'uttr_level_query_layer',
            use_bias = False,
            dtype = tf.float32)
        self._memory_layer = tf.layers.Dense(uttr_level_num_units,
            name = 'uttr_level_memory_layer',
            use_bias = False,
            dtype = tf.float32)
        self._uttr_enc_cell = tf.nn.rnn_cell.GRUCell(n_hidden_units)
        self._uttr_level_num_units = uttr_level_num_units

        word_level_dec_query_layer = tf.layers.Dense(word_level_num_units,
            name = 'word_level_dec_query_layer',
            use_bias = False,
            dtype = tf.float32)
        word_level_enc_query_layer = tf.layers.Dense(word_level_num_units,
            name = 'word_level_enc_query_layer',
            use_bias = False,
            dtype = tf.float32)
        word_level_memory_layer = tf.layers.Dense(word_level_num_units,
            name = 'word_level_memory_layer',
            use_bias = False,
            dtype = tf.float32)

        self._attention_v_ul = tf.Variable(tf.truncated_normal([uttr_level_num_units], stddev = 0.1), name = 'attention_v_ul')
        self._attention_v_wl = tf.Variable(tf.truncated_normal([word_level_num_units], stddev = 0.1), name = 'attention_v_wl')

        self._word_level_attns = []
        for i in range(memory.shape[0].value):
            self._word_level_attns.append(WordLevelAttentionMechanism(
                attention_v = self._attention_v_wl,
                dec_query_layer = word_level_dec_query_layer,
                enc_query_layer = word_level_enc_query_layer,
                memory_layer = word_level_memory_layer,
                memory = memory[i,:,:,:],
                memory_sequence_length = memory_sequence_length[i,:]))

        self.dtype = tf.float32

        self._memory = memory
        self._hist_length = hist_length
        self._batch_size = memory.shape[1].value
        self._alignments_size = memory.shape[0].value

        self._alignments_w_size = self._alignments_size * self._word_level_attns[0].alignments_size

        score_mask_value = tf.as_dtype(self.dtype).as_numpy_dtype(-np.inf)
        self._probability_fn = lambda score: tf.nn.softmax(_maybe_mask_score(score, self._hist_length, score_mask_value))

        # self._values changes each time self.__call__() is called
        self._values = tf.zeros([self._batch_size, self._alignments_size, n_hidden_units])

    def __call__(self, query, state):
        """Score the query based on the keys and values.
        Args:
            query: Tensor of dtype matching `self.values` and shape
                `[batch_size, query_depth]`.
            state: Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size]`
                (`alignments_size` is memory's `max_time`).
        Returns:
            alignments: Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size]` (`alignments_size` is memory's
                `max_time`).
        """
        with tf.variable_scope('uttr_level_encoding'):
            uttr_enc_outputs = []
            uttr_enc_state = self._uttr_enc_cell.zero_state(self._batch_size, self.dtype)
            alignments_w = []
            for i in reversed(range(self._alignments_size)):
                word_alignments = self._word_level_attns[i](dec_query = query, enc_query = uttr_enc_state)
                alignments_w.append(word_alignments)
                word_alignments = tf.expand_dims(word_alignments, 1)
                uttr_enc_input = tf.matmul(word_alignments, self._word_level_attns[i].values)
                uttr_enc_input = tf.squeeze(uttr_enc_input, [1])
                _, uttr_enc_state = self._uttr_enc_cell(uttr_enc_input, uttr_enc_state)
                uttr_enc_outputs.append(uttr_enc_state)
            uttr_enc_outputs = tf.transpose(tf.stack(uttr_enc_outputs), perm = [1, 0, 2])  # [batch_size, max_hist_len, n_hidden_units]

            alignments_w = tf.transpose(tf.stack(alignments_w), perm = [1, 0, 2])  # [batch_size, max_hist_len, max_uttr_len]
            alignments_w = tf.reshape(alignments_w, [self._batch_size, self._alignments_w_size])

        with tf.variable_scope('uttr_level_attention'):
            mask = tf.sequence_mask(self._hist_length, maxlen = self._alignments_size, dtype = self.dtype)
            mask = tf.expand_dims(mask, 2)
            self._values = uttr_enc_outputs * mask

            processed_query = self._query_layer(query)
            processed_query = tf.expand_dims(processed_query, 1)
            keys = self._memory_layer(self._values)
            # v = tf.get_variable('attention_v', [self._uttr_level_num_units], dtype = self.dtype)
            score = tf.reduce_sum(self._attention_v_ul * tf.tanh(keys + processed_query), [2])
            alignments = self._probability_fn(score)
            next_state = alignments

        return alignments, next_state, alignments_w

    @property
    def values(self):
        return self._values

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def alignments_size(self):
        return self._alignments_size

    @property
    def alignments_w_size(self):
        return self._alignments_w_size

    @property
    def state_size(self):
        return self._alignments_size

    def initial_alignments(self, batch_size, dtype):
        max_time = self._alignments_size
        return _zero_state_tensors(max_time, batch_size, dtype)

    def initial_alignments_w(self, batch_size, dtype):
        max_time = self._alignments_w_size
        return _zero_state_tensors(max_time, batch_size, dtype)

    def initial_state(self, batch_size, dtype):
        return self.initial_alignments(batch_size, dtype)
