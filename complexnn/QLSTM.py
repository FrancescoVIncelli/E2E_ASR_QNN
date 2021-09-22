# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from complexnn import QuaternionDense

class QLSTM (Layer):
    def __init__(self, feat_size, hidden_size):
        super(QLSTM, self).__init__()
    
        self.input_dim = feat_size,
        self.hidden_dim = hidden_size
        self.act = activations.get("tanh")
        self.act_gate=activations.get("sigmoid")
        
        # +1 because feat_size = the number on the sequence, and the output one hot will also have
        # a blank dimension so FEAT_SIZE + 1 BLANK
        self.output_dim = feat_size + 1
        
        # List initialization
        I = Input(shape=(input_dim,))
        
        # Gates initialization
        self.wfx = QuaternionDense(units=hidden_dim, input_dim=self.input_dim) # Forget
        self.ufh = QuaternionDense(units=hidden_dim, use_bias=False, input_dim=self.hidden_dim) # Forget
        
        self.wix = QuaternionDense(units=self.hidden_dim, input_dim=self.input_dim)) # Input
        self.uih = QuaternionDense(units=self.hidden_dim, use_bias=False, input_dim=self.hidden_dim) # Input
        
        self.wox = QuaternionDense(units=self.hidden_dim, input_dim=self.input_dim)) # Output
        self.uoh = QuaternionDense(units=self.hidden_dim, use_bias=False, input_dim=self.hidden_dim) # Output
        
        self.wcx = QuaternionDense(units=self.hidden_dim, input_dim=self.input_dim)) # Cell
        self.uch = QuaternionDense(units=self.hidden_dim, use_bias=False, input_dim=self.hidden_dim) # Cell
        
        # Output layer initialization
        self.fco = Dense(units=output_dim, input_dim=self.hidden_dim)
        
        # Optimizer
        self.adam = Adam(learning_rate=0.005)
    
    def call(self, inputs):
        
        h_init = tf.Variable(initial_value=tf.zeros(shape=(inputs.shape[1],self.hidden_dim)), trainable=True)
        
        # Feed-forward affine transformation (done in parallel)
        wfx_out=self.wfx(inputs)
        wix_out=self.wix(inputs)
        wox_out=self.wox(inputs)
        wcx_out=self.wcx(inputs)
        
        # Processing time steps
        out = []
        c = h_init
        h = h_init
        
        for k in range(inputs.shape[0]):
            ft=self.act_gate(wfx_out[k] + self.ufh(h))
            it=self.act_gate(wix_out[k] + self.uih(h))
            ot=self.act_gate(wox_out[k] + self.uoh(h))
                  
            at = wcx_out[k] + self.uch(h)       
            c  = it * self.act(at) + ft * c
            h  = ot * self.act(c)  

            output = self.fco(h)
            out.append(tf.expand_dims(output,axis=0))
            
        return tf.concat(out, axis=0)
