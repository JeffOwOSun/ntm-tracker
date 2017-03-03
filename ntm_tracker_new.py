from ntm_cell import NTMCell
import tensorflow as tf
from utility import stack_into_tensor, unstack_into_tensorarray
class LoopNTMTracker(object):
    def __init__(self, sequence_length, output_dim,
            initializer=tf.random_uniform_initializer(-.1,.1), **kwargs):
        self.cell = NTMCell(output_dim, **kwargs)
        self.initializer = initializer
        self.sequence_length = sequence_length
        #self.sequence_length = tf.placeholder(tf.int32, name="sequence_length")


    def __call__(self, inputs, state=None, scope=None):
        with tf.variable_scope(scope or 'ntm-tracker', initializer=self.initializer):
            state = state or self.cell.zero_state(inputs.get_shape().as_list()[0],
                    self.initializer)
            inputs = unstack_into_tensorarray(inputs, 1)

            outputs = tf.TensorArray(tf.float32, self.sequence_length)
            output_logits = tf.TensorArray(tf.float32, self.sequence_length)
            time = tf.constant(0, dtype=tf.int32)
            M = state['M']
            w = state['w']
            read = state['read']
            controller_state = state['controller_state']

            final_result = tf.while_loop(
                    cond = lambda time, *_: time < self.sequence_length,
                    body = self._loop_body,
                    loop_vars = (time, outputs, output_logits, inputs, M, w,
                        read, controller_state),
                    parallel_iterations = 32,
                    swap_memory = True)

            return (stack_into_tensor(final_result[1], 1, name="outputs"),
                    stack_into_tensor(final_result[2], 1, name="output_logits"))

    def _loop_body(self, time, outputs, output_logits, inputs, M, w,
                        read, controller_state):
        step_input = inputs.read(time)
        ntm_output, ntm_output_logit, _, _, M, w, read, controller_state = self.cell(
                step_input, None, M_prev=M, w_prev=w, read_prev=read,
                controller_state=controller_state)
        outputs = outputs.write(time, ntm_output)
        output_logits = output_logits.write(time, ntm_output_logit)

        return (time+1, outputs, output_logits, inputs, M, w, read,
                controller_state)

class PlainNTMTracker(object):
    """
    This tracker is a general abstract version, where no assumption on the
    nature of input is made
    """
    def __init__(self, model_length, output_dim,
            initializer=tf.random_uniform_initializer(-0.1,0.1),
            **kwargs):
        """
        ntm tracker core.
        uses NTMCell to form the pipeline
        """
        self.model_length = model_length
        self.cell = NTMCell(output_dim, **kwargs)
        self.initializer = initializer

    def __call__(self, inputs, state=None, scope=None):
        with tf.variable_scope(scope or 'ntm-tracker', initializer=self.initializer):
            state = state or self.cell.zero_state(inputs.get_shape().as_list()[0],
                    self.initializer)
            self.outputs = []
            self.output_logits = []
            self.states = []
            self.debugs = []
            self.states.append(state)

            for idx in xrange(self.model_length):
                if idx > 0:
                    tf.get_variable_scope().reuse_variables()

                ntm_output, ntm_output_logit, state, debug = self.cell(
                        inputs[:,idx,:], state)
                self.states.append(state)
                self.debugs.append(debug)
                self.outputs.append(ntm_output)
                self.output_logits.append(ntm_output_logit)

        return tf.stack(self.outputs, axis=1, name="outputs"),\
                tf.stack(self.output_logits, axis=1, name="output_logits"),\
                self.states, self.debugs

class NTMTracker(object):
    def __init__(self, sequence_length, batch_size, output_dim,
            initializer=tf.random_uniform_initializer(-0.1,0.1),
            two_step=False,
            **kwargs):
        """
        ntm tracker core.
        uses NTMCell to form the pipeline
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.cell = NTMCell(output_dim, **kwargs)
        self.initializer = initializer
        self.two_step=two_step

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
        with tf.variable_scope(scope or 'ntm-tracker', initializer=self.initializer):
            zero_switch=tf.fill([inputs.get_shape().as_list()[0],1], 0.0,
                    name="zero_switch")
            one_switch=tf.fill([inputs.get_shape().as_list()[0],1], 1.0,
                    name="one_switch")
            dummy_input=tf.zeros_like(inputs[:,0,:], dtype=tf.float32, name="dummy_input")
            dummy_target = tf.zeros_like(target,
                    dtype=tf.float32, name="dummy-target")
            indicator=target
            for idx in xrange(self.sequence_length):
                if idx > 0:
                    tf.get_variable_scope().reuse_variables()
                    indicator = dummy_target

                if self.two_step:
                    if idx == 0:
                        #[batch_size, 1]
                        ntm_output, ntm_output_logit, state, debug = self.cell(
                                tf.concat_v2([zero_switch, inputs[:,idx,:],
                                    target], 1),
                                state)
                        self.states.append(state)
                        self.debugs.append(debug)
                        self.outputs.append(ntm_output)
                        self.output_logits.append(ntm_output_logit)
                    else:
                        """
                        1. step 1, present the input
                        """
                        ntm_output, ntm_output_logit, state, debug = self.cell(
                                tf.concat_v2([zero_switch, inputs[:,idx,:],
                                    dummy_target], 1),
                                state)
                        self.states.append(state)
                        self.debugs.append(debug)
                        self.outputs.append(ntm_output)
                        self.output_logits.append(ntm_output_logit)
                        """
                        1. step 2, ask for output
                        """
                        ntm_output, ntm_output_logit, state, debug = self.cell(
                                tf.concat_v2([one_switch, dummy_input,
                                    dummy_target], 1),
                                state)
                        self.states.append(state)
                        self.debugs.append(debug)
                        self.outputs.append(ntm_output)
                        self.output_logits.append(ntm_output_logit)
                else:
                    ntm_output, ntm_output_logit, state, debug = self.cell(
                            tf.concat_v2([inputs[:,idx,:], indicator], 1), state)

                    self.states.append(state)
                    self.debugs.append(debug)
                    self.outputs.append(ntm_output)
                    self.output_logits.append(ntm_output_logit)
        # in two step formation, the total length of the stacked output should
        # be 2*seq_length - 1
        return tf.stack(self.outputs, axis=1, name="outputs"),\
                tf.stack(self.output_logits, axis=1, name="output_logits"),\
                self.states, self.debugs




