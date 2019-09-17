import collections
import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import AttentionWrapper

_zero_state_tensors = rnn_cell_impl._zero_state_tensors

class MyAttentionWrapperState(collections.namedtuple('MyAttentionWrapperState',
                             ('cell_state', 'attention', 'time', 'alignments',
                              'alignment_history_ul', 'alignment_history_wl',
                              'attention_state'))):
    def clone(self, **kwargs):
        def with_same_shape(old, new):
            if isinstance(old, tf.Tensor) and isinstance(new, tf.Tensor):
                return tf.contrib.framework.with_same_shape(old, new)
            return new
        return tf.contrib.framework.nest.map_structure(
            with_same_shape,
            self,
            super(MyAttentionWrapperState, self)._replace(**kwargs))

def _compute_attention(attention_mechanism, cell_output, attention_state,
                       attention_layer):
    alignments, next_attention_state, alignments_w = attention_mechanism(
        cell_output, state = attention_state)

    expanded_alignments = tf.expand_dims(alignments, 1)
    context = tf.matmul(expanded_alignments, attention_mechanism.values)
    context = tf.squeeze(context, [1])

    if attention_layer is not None:
        attention = attention_layer(tf.concat([cell_output, context], 1))
    else:
        attention = context

    return attention, alignments, next_attention_state, alignments_w

class MyAttentionWrapper(AttentionWrapper):
    """Custom AttentionWrapper."""
    def __init__(self,
                 cell,
                 attention_mechanism,
                 emotion_vector,
                 attention_layer_size = None,
                 alignment_history = True,
                 cell_input_fn = None,
                 output_attention = False,
                 initial_cell_state = None,
                 name = None,
                 attention_layer = None):
        super().__init__(cell, attention_mechanism,
            attention_layer_size = attention_layer_size,
            alignment_history = alignment_history,
            cell_input_fn = cell_input_fn,
            output_attention = output_attention,
            initial_cell_state = initial_cell_state,
            name = name,
            attention_layer = attention_layer)
        self._emotion_vector = emotion_vector

    @property
    def output_size(self):
        return self._cell.output_size + self._emotion_vector.shape[1].value

    @property
    def state_size(self):
        """The `state_size` property of `MyAttentionWrapper`.
        Returns:
          A `MyAttentionWrapperState` tuple containing shapes used by this object.
        """
        return MyAttentionWrapperState(
            cell_state = self._cell.state_size,
            time = tf.TensorShape([]),
            attention = self._attention_layer_size,
            alignments = self._item_or_tuple(
                a.alignments_size for a in self._attention_mechanisms),
            attention_state = self._item_or_tuple(
                a.state_size for a in self._attention_mechanisms),
            alignment_history_ul = self._item_or_tuple(
                a.alignments_size if self._alignment_history else ()
                for a in self._attention_mechanisms),
            alignment_history_wl = self._item_or_tuple(
                a.alignments_w_size if self._alignment_history else ()
                for a in self._attention_mechanisms))

    def zero_state(self, batch_size, dtype):
        """Return an initial (zero) state tuple for this `MyAttentionWrapper`.
        **NOTE** Please see the initializer documentation for details of how
        to call `zero_state` if using a `MyAttentionWrapper` with a
        `BeamSearchDecoder`.
        Args:
          batch_size: `0D` integer tensor: the batch size.
          dtype: The internal state data type.
        Returns:
          An `MyAttentionWrapperState` tuple containing zeroed out tensors and,
          possibly, empty `TensorArray` objects.
        Raises:
          ValueError: (or, possibly at runtime, InvalidArgument), if
            `batch_size` does not match the output size of the encoder passed
            to the wrapper object at initialization time.
        """
        with tf.name_scope(type(self).__name__ + "ZeroState", values = [batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            error_message = (
                "When calling zero_state of MyAttentionWrapper %s: " % self._base_name +
                "Non-matching batch sizes between the memory "
                "(encoder output) and the requested batch size.  Are you using "
                "the BeamSearchDecoder?  If so, make sure your encoder output has "
                "been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
                "the batch_size= argument passed to zero_state is "
                "batch_size * beam_width.")
            with tf.control_dependencies(self._batch_size_checks(batch_size, error_message)):
                cell_state = tf.contrib.framework.nest.map_structure(
                    lambda s: tf.identity(s, name = "checked_cell_state"),
                    cell_state)
            initial_alignments = []
            initial_alignments_w = []
            all_attentions = []
            initial_alignment_history_ul = [
                tf.TensorArray(
                    dtype,
                    size = 0,
                    dynamic_size = True,
                    element_shape = a.initial_alignments(batch_size, dtype).shape)
                if self._alignment_history else ()
                for a in self._attention_mechanisms]
            initial_alignment_history_wl = [
                tf.TensorArray(
                    dtype,
                    size = 0,
                    dynamic_size = True,
                    element_shape = a.initial_alignments_w(batch_size, dtype).shape)
                if self._alignment_history else ()
                for a in self._attention_mechanisms]
            for i, attention_mechanism in enumerate(self._attention_mechanisms):
                attention, alignments, _, alignments_w = _compute_attention(
                    attention_mechanism, cell_state, None, None)
                initial_alignment_history_ul[i] = initial_alignment_history_ul[i].write(
                    0, alignments) if self._alignment_history else ()
                initial_alignment_history_wl[i] = initial_alignment_history_wl[i].write(
                    0, alignments_w) if self._alignment_history else ()
                initial_alignments.append(alignments)
                initial_alignments_w.append(alignments_w)
                all_attentions.append(attention)
            attention = tf.concat(all_attentions, 1)
            return MyAttentionWrapperState(
                cell_state = cell_state,
                time = tf.zeros([], dtype = tf.int32) + 1,
                attention = attention,
                alignments = self._item_or_tuple(initial_alignments),
                attention_state = self._item_or_tuple(
                    attention_mechanism.initial_state(batch_size, dtype)
                    for attention_mechanism in self._attention_mechanisms),
                alignment_history_ul = self._item_or_tuple(initial_alignment_history_ul),
                alignment_history_wl = self._item_or_tuple(initial_alignment_history_wl))

    def call(self, inputs, state):
        if not isinstance(state, MyAttentionWrapperState):
            raise TypeError("Expected state to be instance of MyAttentionWrapperState. "
                            "Received type %s instead."  % type(state))

        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        cell_batch_size = (
            cell_output.shape[0].value or tf.shape(cell_output)[0])
        error_message = (
            "When applying MyAttentionWrapper %s: " % self.name +
            "Non-matching batch sizes between the memory "
            "(encoder output) and the query (decoder output).  Are you using "
            "the BeamSearchDecoder?  You may need to tile your memory input via "
            "the tf.contrib.seq2seq.tile_batch function with argument "
            "multiple=beam_width.")
        with tf.control_dependencies(self._batch_size_checks(cell_batch_size, error_message)):
            cell_output = tf.identity(
                cell_output, name = "checked_cell_output")

        if self._is_multi:
            previous_attention_state = state.attention_state
            previous_alignment_history_ul = state.alignment_history_ul
            previous_alignment_history_wl = state.alignment_history_wl
        else:
            previous_attention_state = [state.attention_state]
            previous_alignment_history_ul = [state.alignment_history_ul]
            previous_alignment_history_wl = [state.alignment_history_wl]

        all_alignments = []
        all_attentions = []
        all_attention_states = []
        maybe_all_histories_ul = []
        maybe_all_histories_wl = []
        for i, attention_mechanism in enumerate(self._attention_mechanisms):
            attention, alignments, next_attention_state, alignments_w = _compute_attention(
                attention_mechanism, cell_output, previous_attention_state[i],
                self._attention_layers[i] if self._attention_layers else None)
            alignment_history_ul = previous_alignment_history_ul[i].write(
                state.time, alignments) if self._alignment_history else ()
            alignment_history_wl = previous_alignment_history_wl[i].write(
                state.time, alignments_w) if self._alignment_history else ()

            all_attention_states.append(next_attention_state)
            all_alignments.append(alignments)
            all_attentions.append(attention)
            maybe_all_histories_ul.append(alignment_history_ul)
            maybe_all_histories_wl.append(alignment_history_wl)

        attention = tf.concat(all_attentions, 1)
        next_state = MyAttentionWrapperState(
            time = state.time + 1,
            cell_state = next_cell_state,
            attention = attention,
            attention_state = self._item_or_tuple(all_attention_states),
            alignments = self._item_or_tuple(all_alignments),
            alignment_history_ul = self._item_or_tuple(maybe_all_histories_ul),
            alignment_history_wl = self._item_or_tuple(maybe_all_histories_wl))

        if self._output_attention:
            return attention, next_state
        else:
            return tf.concat([cell_output, self._emotion_vector], 1), next_state
