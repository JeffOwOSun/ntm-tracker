from ntm_cell import NTMCell
import tensorflow as tf

class NTMTracker(object):
    def __init__(self, sequence_length=20, batch_size=32,
            initializer=tf.random_uniform_initializer(-0.1,0.1),
            **kwargs):
        """
        ntm tracker core.
        uses NTMCell to form the pipeline
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.cell = NTMCell(**kwargs)
        self.initializer = initializer

    def __call__(self, inputs, target, scope=None):
        """
        Args:
            inputs: [batch, sequence_len, num_features*num_channels]
            target: [batch, num_features]
        """
        self.outputs = []
        self.output_logits = []
        self.states = []
        self.debugs = []
        state = self.cell.zero_state(self.batch_size)
        self.states.append(state)
        indicator = target
        with tf.variable_scope(scope or 'ntm-tracker', initializer=self.initializer):
            for idx in xrange(self.sequence_length):
                if idx > 0:
                    tf.get_variable_scope().reuse_variables()
                    indicator = tf.zeros(shape=target.get_shape(),
                            name="dummy-target")

                ntm_output, ntm_output_logit, state, debug = self.cell(
                        inputs[:,idx,:], indicator, state)

                self.states.append(state)
                self.debugs.append(debug)
                self.outputs.append(ntm_output)
                self.output_logits.append(ntm_output_logit)
        return tf.stack(self.outputs, axis=1, name="outputs"),\
                tf.stack(self.output_logits, axis=1, name="output_logits"),\
                self.states, self.debugs




