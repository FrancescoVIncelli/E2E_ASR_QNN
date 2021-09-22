# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras.layers import Layer, Dense, Input
from tensorflow.keras.optimizers import Adam
# from complexnn import QuaternionDense
from .dense import QuaternionDense

"""
class QuaternionLSTM (Layer):
    def __init__(self, 
                 feat_size,
                 hidden_size,
                 **kwargs):
        # if 'input_shape' not in kwargs and 'input_dim' in kwargs:
        #     kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(QuaternionLSTM, self).__init__(**kwargs)
        
        self.input_dim = feat_size,
        self.hidden_dim = hidden_size
        self.act = activations.get("tanh")
        self.act_gate=activations.get("sigmoid")
        
        #### Moved to build() function ####
        # +1 because feat_size = the number on the sequence, and the output one hot will also have
        # a blank dimension so FEAT_SIZE + 1 BLANK
        self.output_dim = feat_size # + 1
        
        
        # Gates initialization
        self.wfx = QuaternionDense(units=self.hidden_dim, input_dim=self.input_dim) # Forget
        self.ufh = QuaternionDense(units=self.hidden_dim, use_bias=False, input_dim=self.hidden_dim) # Forget
        
        self.wix = QuaternionDense(units=self.hidden_dim, input_dim=self.input_dim) # Input
        self.uih = QuaternionDense(units=self.hidden_dim, use_bias=False, input_dim=self.hidden_dim) # Input
        
        self.wox = QuaternionDense(units=self.hidden_dim, input_dim=self.input_dim) # Output
        self.uoh = QuaternionDense(units=self.hidden_dim, use_bias=False, input_dim=self.hidden_dim) # Output
        
        self.wcx = QuaternionDense(units=self.hidden_dim, input_dim=self.input_dim) # Cell
        self.uch = QuaternionDense(units=self.hidden_dim, use_bias=False, input_dim=self.hidden_dim) # Cell
        
        # Output layer initialization
        self.fco = Dense(units=self.output_dim, input_dim=self.hidden_dim)
        
        # Optimizer
        self.adam = Adam(learning_rate=0.005)
        
    
    
    # def build(self, input_shape):
    #     print("[****] imput_shape: ", input_shape)
    #     # feat_size = input_shape[-1]
    #     print("[****] feat_size: ", feat_size)
        
    #     # Gates initialization
    #     self.wfx = QuaternionDense(units=self.hidden_dim, input_dim=feat_size) # Forget
    #     self.ufh = QuaternionDense(units=self.hidden_dim, use_bias=False, input_dim=self.hidden_dim) # Forget
        
    #     self.wix = QuaternionDense(units=self.hidden_dim, input_dim=feat_size) # Input
    #     self.uih = QuaternionDense(units=self.hidden_dim, use_bias=False, input_dim=self.hidden_dim) # Input
        
    #     self.wox = QuaternionDense(units=self.hidden_dim, input_dim=feat_size) # Output
    #     self.uoh = QuaternionDense(units=self.hidden_dim, use_bias=False, input_dim=self.hidden_dim) # Output
        
    #     self.wcx = QuaternionDense(units=self.hidden_dim, input_dim=feat_size) # Cell
    #     self.uch = QuaternionDense(units=self.hidden_dim, use_bias=False, input_dim=self.hidden_dim) # Cell
        
    #     # Output layer initialization
    #     self.fco = Dense(units=self.output_dim, input_dim=self.hidden_dim)
        
    # Optimizer
    # self.adam = Adam(learning_rate=0.005)
        
    def call(self, inputs):
        h_init = tf.Variable(lambda: tf.zeros(shape=(inputs.shape[1],self.hidden_dim)))
        
        # Feed-forward affine transformation
        wfx_out=self.wfx(inputs)
        wix_out=self.wix(inputs)
        wox_out=self.wox(inputs)
        wcx_out=self.wcx(inputs)
        
        # Processing time steps
        out = []
        c = h_init
        h = h_init
        
        # for k in range(inputs.shape[0]):
        #     ft=self.act_gate(wfx_out[k] + self.ufh(h))
        #     it=self.act_gate(wix_out[k] + self.uih(h))
        #     ot=self.act_gate(wox_out[k] + self.uoh(h))
            
        #     at = wcx_out[k] + self.uch(h)       
        #     c  = it * self.act(at) + ft * c
        #     h  = ot * self.act(c)  
            
        #     output = self.fco(h)
        #     out.append(tf.expand_dims(output,axis=0))
        
        ft=self.act_gate(wfx_out + self.ufh(h))
        it=self.act_gate(wix_out + self.uih(h))
        ot=self.act_gate(wox_out + self.uoh(h))
        
        at = wcx_out + self.uch(h)       
        c  = it * self.act(at) + ft * c
        h  = ot * self.act(c)  
        
        output = self.fco(h)
        # out.append(tf.expand_dims(output,axis=0))
            
        return output # tf.concat(out, axis=0)
"""

class QuaternionLSTM (Layer):
    def __init__(self, 
                 feat_size,
                 hidden_size,
                 **kwargs):
        # if 'input_shape' not in kwargs and 'input_dim' in kwargs:
        #     kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(QuaternionLSTM, self).__init__(**kwargs)
        
        # self.input_dim = feat_size,
        # self.hidden_dim = hidden_size
        # self.output_dim = feat_size # + 1
        self.act = activations.get("tanh")
        self.act_gate=activations.get("sigmoid")
    
    """
    def build(self, input_shape):
        print("[****] imput_shape: ", input_shape)
        # feat_size = input_shape[-1]
        print("[****] feat_size: ", feat_size)
        
        # Gates initialization
        self.wfx = QuaternionDense(units=self.hidden_dim, input_dim=feat_size) # Forget
        self.ufh = QuaternionDense(units=self.hidden_dim, use_bias=False, input_dim=self.hidden_dim) # Forget
        
        self.wix = QuaternionDense(units=self.hidden_dim, input_dim=feat_size) # Input
        self.uih = QuaternionDense(units=self.hidden_dim, use_bias=False, input_dim=self.hidden_dim) # Input
        
        self.wox = QuaternionDense(units=self.hidden_dim, input_dim=feat_size) # Output
        self.uoh = QuaternionDense(units=self.hidden_dim, use_bias=False, input_dim=self.hidden_dim) # Output
        
        self.wcx = QuaternionDense(units=self.hidden_dim, input_dim=feat_size) # Cell
        self.uch = QuaternionDense(units=self.hidden_dim, use_bias=False, input_dim=self.hidden_dim) # Cell
        
        # Output layer initialization
        self.fco = Dense(units=self.output_dim, input_dim=self.hidden_dim)
        
        # Optimizer
        # self.adam = Adam(learning_rate=0.005)
    """
    
    def build(self, input_shape):
        self.input_dim = input_shape[0]
        self.hidden_dim = 256
        self.output_dim = 256
        
        print(self.input_dim)
        print(self.hidden_dim)
        
        #"""
        # Gates initialization
        self.wfx = QuaternionDense(units=self.hidden_dim, input_dim=self.input_dim) # Forget
        self.ufh = QuaternionDense(units=self.hidden_dim, use_bias=False, input_dim=self.hidden_dim) # Forget
        
        self.wix = QuaternionDense(units=self.hidden_dim, input_dim=self.input_dim) # Input
        self.uih = QuaternionDense(units=self.hidden_dim, use_bias=False, input_dim=self.hidden_dim) # Input
        
        self.wox = QuaternionDense(units=self.hidden_dim, input_dim=self.input_dim) # Output
        self.uoh = QuaternionDense(units=self.hidden_dim, use_bias=False, input_dim=self.hidden_dim) # Output
        
        self.wcx = QuaternionDense(units=self.hidden_dim, input_dim=self.input_dim) # Cell
        self.uch = QuaternionDense(units=self.hidden_dim, use_bias=False, input_dim=self.hidden_dim) # Cell
        
        # Output layer initialization
        self.fco = Dense(units=self.output_dim, input_dim=self.hidden_dim)
        
        # Optimizer
        self.adam = Adam(learning_rate=0.005)
        #"""
    def call(self, inputs):
        h_init = tf.Variable(lambda: tf.zeros(shape=(inputs.shape[1],self.hidden_dim)))
        
        # Feed-forward affine transformation
        wfx_out=self.wfx(inputs)
        wix_out=self.wix(inputs)
        wox_out=self.wox(inputs)
        wcx_out=self.wcx(inputs)
        
        # Processing time steps
        out = []
        c = h_init
        h = h_init
        
        ft=self.act_gate(wfx_out + self.ufh(h))
        it=self.act_gate(wix_out + self.uih(h))
        ot=self.act_gate(wox_out + self.uoh(h))
        
        at = wcx_out + self.uch(h)       
        c  = it * self.act(at) + ft * c
        h  = ot * self.act(c)  
        
        output = self.fco(h)
            
        return output



class QuaternionRNNCell(Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 init_criterion='he',
                 kernel_initializer='quaternion',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seed=None,
                 **kwargs):
        
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(QuaternionDense, self).__init__(**kwargs)
        self.units = units
        self.q_units = units // 4
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.init_criterion = init_criterion
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        
        super(QuaternionRNNCell, self).__init__(**kwargs)
        
    def build(self, input_shape):
        input_dim = input_shape[-1] // 4
        kernel_shape = (input_dim, self.units)
        init_shape = (input_dim, self.q_units)
        
        print(type(input_shape))
        self.kernel = self.add_weight(shape=(input_shape[0][-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True
        
        # ==== #
        self.kernel_init = qdense_init(init_shape, self.init_criterion)

        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer='uniform', # self.kernel_init,
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer='zeros',
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
        else:
            self.bias = None
        # ==== #
    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs[0], self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]   
    
# =================== #
# =================== #

"""
input_dim  = 199168
hidden_dim = 256
act = activations.get("tanh")
act_gate=activations.get("sigmoid")

output_dim = 256

# Gates initialization
wfx = QuaternionDense(units=hidden_dim, input_dim=input_dim) # Forget
ufh = QuaternionDense(units=hidden_dim, use_bias=False, input_dim=hidden_dim) # Forget

wix = QuaternionDense(units=hidden_dim, input_dim=input_dim) # Input
uih = QuaternionDense(units=hidden_dim, use_bias=False, input_dim=hidden_dim) # Input

wox = QuaternionDense(units=hidden_dim, input_dim=input_dim) # Output
uoh = QuaternionDense(units=hidden_dim, use_bias=False, input_dim=hidden_dim) # Output

wcx = QuaternionDense(units=hidden_dim, input_dim=input_dim) # Cell
uch = QuaternionDense(units=hidden_dim, use_bias=False, input_dim=hidden_dim) # Cell

# Output layer initialization
fco = Dense(units=output_dim, input_dim=hidden_dim)

h_init = tf.Variable(lambda: tf.zeros(shape=(lstm1.shape[1],hidden_dim)))
        
# Feed-forward affine transformation
wfx_out=wfx(inputs)
wix_out=wix(inputs)
wox_out=wox(inputs)
wcx_out=wcx(inputs)

# Processing time steps
out = []
c = h_init
h = h_init

ft=act_gate(wfx_out + ufh(h))
it=act_gate(wix_out + uih(h))
ot=act_gate(wox_out + uoh(h))

at = wcx_out + uch(h)       
c  = it * act(at) + ft * c
h  = ot * act(c)  

output = fco(h)
out.append(tf.expand_dims(output,axis=0))
"""