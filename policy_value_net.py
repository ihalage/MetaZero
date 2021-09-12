# -*- coding: utf-8 -*-
"""
Implementation of AlphaZero style policy-value network for metasurface design

Author: Achintha Avin Ihalage
Date:   05/03/2020


State = Metasurface position
Actions = (N*N+1)   # each player can switch the element in any position (N*N);  + 1 for pass action

"""
from __future__ import print_function

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
from keras.utils import np_utils

import numpy as np
import pickle

gpu_fraction = 0.46
sessconfig = tf.ConfigProto()
sessconfig.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
session = tf.Session(config=sessconfig)
tf.keras.backend.set_session(session)

class PolicyValueNet():
    """policy-value network """
    def __init__(self, metasurface_width, metasurface_height, model_file=None):
        self.metasurface_width = metasurface_width
        self.metasurface_height = metasurface_height 
        self.l2_const = 1e-4  # coef of l2 penalty 
        self.create_policy_value_net()   
        self._loss_train_op()

        if model_file:
            net_params = pickle.load(open(model_file, 'rb'))
            self.model.set_weights(net_params)
        
    def create_policy_value_net(self):
        """create the policy value network """   
        in_state = Input((5, self.metasurface_width, self.metasurface_height))

        # conv layers
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding="same", data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(in_state)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(conv1)
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(conv2)
        # action policy layers
        policy_net = Conv2D(filters=4, kernel_size=(1, 1), data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(conv3)
        policy_net = Flatten()(policy_net)
        self.policy_net = Dense(self.metasurface_width*self.metasurface_height, activation="softmax", kernel_regularizer=l2(self.l2_const))(policy_net)
        # state value layers
        value_net = Conv2D(filters=4, kernel_size=(1, 1), data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(conv3)
        value_net = Flatten()(value_net)
        value_net = Dense(32, kernel_regularizer=l2(self.l2_const))(value_net)
        self.value_net = Dense(1, activation="linear", kernel_regularizer=l2(self.l2_const))(value_net) # was tanh

        self.model = Model(in_state, [self.policy_net, self.value_net])
        
        def policy_value(state_input):
            state_input_union = np.array(state_input)
            results = self.model.predict_on_batch(state_input_union)
            return results
        self.policy_value = policy_value
        
    def policy_value_fn(self, metasurface):
        """
        input: metasurface state
        output: a list of (action, probability) tuples for each action and the score of the metasurface state (probability of winning from current position)
        """
        legal_positions = metasurface.availables
        current_state = metasurface.get_current_state()
        act_probs, value = self.policy_value(current_state.reshape(-1, 5, self.metasurface_width, self.metasurface_height)) # adding another dimension to be fed to CNN, i.e. batch size=1
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])
        return act_probs, value[0][0]

    def _loss_train_op(self):
        """
        Three loss terms：
        loss = (z - v)^2 + pi^T * log(p) + c||theta||^2
        """

        # get the train op   
        opt = Adam()
        losses = ['categorical_crossentropy', 'mean_squared_error']
        self.model.compile(optimizer=opt, loss=losses)

        def self_entropy(probs):
            return -np.mean(np.sum(probs * np.log(probs + 1e-10), axis=1))

        def train_step(state_input, mcts_probs, winner, learning_rate):
            state_input_union = np.array(state_input)
            mcts_probs_union = np.array(mcts_probs)
            # print (mcts_probs_union.shape)
            winner_union = np.array(winner)
            loss = self.model.evaluate(state_input_union, [mcts_probs_union, winner_union], batch_size=len(state_input), verbose=0)
            action_probs, _ = self.model.predict_on_batch(state_input_union)
            entropy = self_entropy(action_probs)
            K.set_value(self.model.optimizer.lr, learning_rate)
            self.model.fit(state_input_union, [mcts_probs_union, winner_union], batch_size=len(state_input), verbose=0)
            return loss[0], entropy
        
        self.train_step = train_step

    def get_policy_param(self):
        net_params = self.model.get_weights()        
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()
        pickle.dump(net_params, open(model_file, 'wb'), protocol=2)
