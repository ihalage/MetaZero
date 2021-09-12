# -*- coding: utf-8 -*-
"""
Self play to minimize radar cross section of metasurface

"""

from __future__ import print_function
import pickle
from meta_game import Metasurface, Game
from mcts import MCTSPlayer as MCTS_Pure
from mcts_with_policy import MCTSPlayer
# from policy_value_net import PolicyValueNetNumpy
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
from policy_value_net import PolicyValueNet  # Keras
import os
import numpy as np
import copy
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, metasurface):
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = metasurface.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in metasurface.availables:
            print("invalid move")
            move = self.get_action(metasurface)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    # n = 5
    width, height = 6, 6
    model_file = 'best_policy.model'
    try:
        metasurface = Metasurface(width=width, height=height)
        game = Game(metasurface)

        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

        # best_policy = PolicyValueNet(width, height, model_file = model_file)
        # mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        # load the provided model (trained in Theano/Lasagne) into a MCTS player written in pure numpy
        try:
            policy_param = pickle.load(open(model_file, 'rb'))
        except:
            policy_param = pickle.load(open(model_file, 'rb'),
                                       encoding='bytes')  # To support python3
        best_policy = PolicyValueNet(width, height, model_file) # was policy_param here
        mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=40)  # set larger n_playout for better performance

        # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

        # human player, input your move in the format: 2,3
        human = Human()

        # set start_player=0 for human first
        # start player is selected randomly below
        game.start_play(mcts_player, mcts_player, mcts_player, start_player=np.random.choice(3), is_shown=1)

        # # for winners evaluation
        # winning_players = []
        # min_rcs = []
        # min_rcs_states = []
        # for i in range (100):
        #     winner = game.start_play(mcts_player, mcts_player, mcts_player, start_player=np.random.choice(3), is_shown=1)
        #     if winner ==1:
        #         min_rcs_val = np.min(metasurface.player1_rcs_array)
        #         min_rcs_state =  metasurface.player1_states[np.argmin(metasurface.player1_rcs_array)]
        #     if winner==2:
        #         min_rcs_val = np.min(metasurface.player2_rcs_array)
        #         min_rcs_state =  metasurface.player2_states[np.argmin(metasurface.player2_rcs_array)]
        #     else:
        #         min_rcs_val = np.min(metasurface.player3_rcs_array)
        #         min_rcs_state =  metasurface.player3_states[np.argmin(metasurface.player3_rcs_array)]
        #     winning_players.append(winner)
        #     min_rcs.append(min_rcs_val)
        #     min_rcs_states.append(min_rcs_state)

        # print (winning_players)
        # print (min_rcs)




    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
