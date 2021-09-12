# -*- coding: utf-8 -*-
"""
Author: Achintha Avin Ihalage
Date: 05/03/2020

############################################################### Game Definition #########################################################################

    * Reduce the RCS of an NxN metasurface using '00', '01', '10', '11' coding elements corresponding to 0, pi/4, pi/2, pi phase responses
    * Metasurface is initialized with all '0's
    * Each player can play any available position
    * Game terminates after the unit cell is filled
    * The player who has recorded the minimum RCS reduction after a move wins. (after each move, the current RCS will be lower or higher than the previous RCS.
                                                                            Thus, the reduction of RCS of each player at each move is calculated.
                                                                            The player that records minimum RCS reduction value wins.)

##########################################################################################################################################################
"""

from __future__ import print_function
import numpy as np
import pandas as pd     # for pretty printing purposes
from rcs_calc import RCS
import copy

class Metasurface(object):
    """set the metasurface for the metasurface game"""

    def __init__(self, **kwargs):

        self.width = int(kwargs.get('width', 6))
        self.height = int(kwargs.get('height', 6))
        self.total_moves = int(kwargs.get('total_moves', 36))

        # initialize RCS class for RCS calculations
        self.rcs = RCS(self.width, self.height)
        # metasurface states stored as a dict,
        # key: move as location on the metasurface,
        # value: player as pieces type
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.move_arr = []  # to record the moves and number of moves
        # need how many pieces in a row to win
        # self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2, 3]  # player1, player2, player3

        self.player1_rcs_red_array = [] # a list to record the cumulative rcs reduction of player 1
        self.player2_rcs_red_array = [] # a list to record the cumulative rcs reduction of player 2
        self.player3_rcs_red_array = [] # a list to record the cumulative rcs reduction of player 3

        ##### following is used to find the winner based on the rcorded minimum RCS
        self.player1_rcs_array = [] # a list to record RCSs to find minimum at the end of the game
        self.player1_states = []    # keep track of the state after player 1 played

        self.player2_rcs_array = [] # a list of player2's RCSs, to find the minimum at the end
        self.player2_states = []    # keep track of the state after player 2 played

        self.player3_rcs_array = [] # a list to record RCSs to find minimum at the end of the game
        self.player3_states = []    # keep track of the state after player 3 played
        ##############################################################################

        self.current_state = np.zeros((5, self.width, self.height)) # in 4 player case, the current state shape would be (6,width, height)

        self.prev_rcs_val = self.rcs.get_RCS_numpy(self.current_state[3])    # the starting value of RCS is when all '0's in the metasurface   # keep track of previous RCS value
        self.current_rcs_val = 0    # keep track of current RCS value # this will be updated once a move is played


    def init_metasurface(self, start_player=0):
        # if self.width < self.n_in_row or self.height < self.n_in_row:
        #     raise Exception('metasurface width and height can not be '
        #                     'less than {}'.format(self.n_in_row))
        self.current_state = np.zeros((5, self.width, self.height))
        self.current_metasurface = [['0' for i in range(self.width)] for j in range(self.height)]
        self.current_player = self.players[start_player]  # start player
        self.prev_rcs_val = self.rcs.get_RCS_numpy(self.current_state[3])    # the starting value of RCS is when all '0's in the metasurface   # keep track of previous RCS value
        self.current_rcs_val = 0    # keep track of current RCS value
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.move_arr = []  # to record the moves and number of moves
        self.last_move = -1
        
        self.player1_rcs_red_array = [] # a list to record the cumulative rcs reduction of player 1
        self.player2_rcs_red_array = [] # a list to record the cumulative rcs reduction of player 2
        self.player3_rcs_red_array = [] # a list to record the cumulative rcs reduction of player 3

        ##### following is used to find the winner based on the recorded minimum RCS
        self.player1_rcs_array = [] # a list to record RCSs to find minimum at the end of the game
        self.player1_states = []    # keep track of the state after player 1 played

        self.player2_rcs_array = [] # a list of player2 s RCSs, to find the minimum at the end
        self.player2_states = []    # keep track of the state after player 2 played

        self.player3_rcs_array = [] # a list to record RCSs to find minimum at the end of the game
        self.player3_states = []    # keep track of the state after player 3 played
        ##############################################################################

    def move_to_location(self, move):
        """
        3*3 metasurface move locations are as follows:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def get_current_state(self):
        """return the metasurface state from the perspective of the current player.
        state shape: 4*width*height
        """

        return_current_state = self.current_state[:, ::-1, :]   # try to understand, make it consistent with the opposite player?

        return return_current_state

        # square_state = np.zeros((4, self.width, self.height))
        # if self.states:
        #     moves, players = np.array(list(zip(*self.states.items())))
        #     move_curr = moves[players == self.current_player]
        #     move_oppo = moves[players != self.current_player]
        #     square_state[0][move_curr // self.width,
        #                     move_curr % self.height] = 1.0
        #     square_state[1][move_oppo // self.width,
        #                     move_oppo % self.height] = 1.0
        #     # indicate the last move location
        #     square_state[2][self.last_move // self.width,
        #                     self.last_move % self.height] = 1.0
        # if len(self.states) % 2 == 0:
        #     square_state[3][:, :] = 1.0  # indicate the colour to play
        # return square_state[:, ::-1, :]

    def do_move(self, move):

        self.current_state[(self.current_player - 1)][move // self.width,
                                                        move % self.height] = self.current_player

        # this is the current visible metasurface state
        self.current_state[3] = (self.current_state[0] + self.current_state[1] + self.current_state[2]) # get the current metasurface position

        # switch the element of the played location in the current metasurface
        # self.current_metasurface[int(move // self.width)][int(move % self.height)] = ('0' if self.current_metasurface[int(move // self.width)][int(move % self.height)]=='𝞹' else '𝞹')

        self.move_arr.append(move)
        self.availables.remove(move)
        # self.states[move] = self.current_player
        # self.availables.remove(move)
        # self.prev_rcs_val = np.random.choice([0.0032, 0.0043, 0.00541, 0.00114])   # purge
        ##### calculate RCS and record the reduction in current player's RCS reduction list ########
        self.current_rcs_val = self.rcs.get_RCS_numpy(self.current_state[3])    # calculate RCS val using numpy  np.random.choice([0.0032, 0.00135, 0.00342, 0.00134])#
        # self.current_rcs_val = self.rcs.get_RCS_matlab(self.current_state[2])   # calculate RCS val using matlab
        self.current_rcs_red = (self.prev_rcs_val - self.current_rcs_val)   # find the rcs reduction differnece
        

        # self.current_rcs_red=0.0001
        # get a copy of the metasurface because it changes in-place
        metasurface_cpy=copy.deepcopy(self.current_metasurface)
        if self.current_player == self.players[0]:
            self.current_metasurface[int(move // self.width)][int(move % self.height)] = '𝞹/4'
            metasurface_cpy[int(move // self.width)][int(move % self.height)] = '𝞹/4'
            self.player1_rcs_array.append(self.current_rcs_val)
            self.player1_states.append(metasurface_cpy)
            self.player1_rcs_red_array.append(self.current_rcs_red)
            self.current_state[4][:, :] = 1.0   # indicate the colour to play
            self.current_player = self.players[1]   # switch the player

        elif self.current_player== self.players[1]:
            self.current_metasurface[int(move // self.width)][int(move % self.height)] = '𝞹/2'
            metasurface_cpy[int(move // self.width)][int(move % self.height)] = '𝞹/2'
            self.player2_rcs_array.append(self.current_rcs_val)
            self.player2_states.append(metasurface_cpy)
            self.player2_rcs_red_array.append(self.current_rcs_red)
            self.current_state[4][:, :] = 2.0   # indicate the colour to play
            self.current_player = self.players[2]   # switch the player

        else:
            self.current_metasurface[int(move // self.width)][int(move % self.height)] = '𝞹'
            metasurface_cpy[int(move // self.width)][int(move % self.height)] = '𝞹'
            self.player3_rcs_array.append(self.current_rcs_val)
            # print (self.player1_rcs_array)
            # print (self.current_state[3])
            # print ('Break.....')
            self.player3_states.append(metasurface_cpy)
            # print(len(self.player3_states))
            # print(self.player3_states)
            self.player3_rcs_red_array.append(self.current_rcs_red)
            self.current_state[4][:, :] = 3.0   # indicate the colour to play
            self.current_player = self.players[0]   # switch the player

        self.prev_rcs_val = self.current_rcs_val    # update previous RCS value
        # self.current_player = (
        #     self.players[0] if self.current_player == self.players[1]
        #     else self.players[1]
        # )
        
        # self.last_move = move

        # # indicate the last move location
        # self.current_state[2][self.last_move // self.width,
        #                     self.last_move % self.height] = 1.0

        # self.current_state = self.current_state[:, ::-1, :]     # try to understand, make it consistent with the opposite player

    def has_a_winner(self):

        # still need to consider the terminating condition where both players pass
        # print (len(self.move_arr))
        # if len(self.move_arr) == self.total_moves:
        # print ('has a winner availables: ', len(self.availables))
        if len(self.availables) == 0:
        #     player1_cum_rcs_red = np.sum(self.player1_rcs_red_array)
        #     player2_cum_rcs_red = np.sum(self.player2_rcs_red_array)

        #     # if player1_cum_rcs_red > player2_cum_rcs_red:
        #     #     winner = self.players[0]

        #     # elif player1_cum_rcs_red == player2_cum_rcs_red:
        #     #     print ('Match Drawn')
        #     #     winner = -1 #self.players[1]    # purge. just for testing

        #     # else:
        #     #     winner = self.players[1]

            player1_min_rcs = np.min(self.player1_rcs_array)
            player2_min_rcs = np.min(self.player2_rcs_array)
            player3_min_rcs = np.min(self.player3_rcs_array)

            if player1_min_rcs < player2_min_rcs and player1_min_rcs < player3_min_rcs:
                winner = self.players[0]
            elif player2_min_rcs < player1_min_rcs and player2_min_rcs < player3_min_rcs:
                winner= self.players[1]
            elif player3_min_rcs < player1_min_rcs and player3_min_rcs < player2_min_rcs:
                winner = self.players[2]
            else:
                winner = -1


            return True, winner

        # A different terminating condition
        # if (self.current_rcs_val < 0.0028) and (len(self.move_arr)>2):
        #     winner = self.current_player
        #     return True, winner
        # else:
        #     return False, -1

        return False, -1


        # width = self.width
        # height = self.height
        # states = self.states
        # n = self.n_in_row

        # moved = list(set(range(width * height)) - set(self.availables))
        # if len(moved) < self.n_in_row *2-1:
        #     return False, -1

        # for m in moved:
        #     h = m // width
        #     w = m % width
        #     player = states[m]

        #     if (w in range(width - n + 1) and
        #             len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
        #         return True, player

        #     if (h in range(height - n + 1) and
        #             len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
        #         return True, player

        #     if (w in range(width - n + 1) and h in range(height - n + 1) and
        #             len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
        #         return True, player

        #     if (w in range(n - 1, width) and h in range(height - n + 1) and
        #             len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
        #         return True, player

        # return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        # elif not len(self.availables):
        #     return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """game server"""

    def __init__(self, metasurface, **kwargs):
        self.metasurface = metasurface

    def graphic(self, metasurface, player1, player2, player3):
        """Draw the metasurface and show game info"""
        width = metasurface.width
        height = metasurface.height

        # metasurface = [['0' for i in range(width)] for j in range(height)]
        print (pd.DataFrame(metasurface.current_metasurface).iloc[::-1])
        print ()


        # print("Player", player1, "with X".rjust(3))
        # print("Player", player2, "with O".rjust(3))
        # print()
        # for x in range(width):
        #     print("{0:8}".format(x), end='')
        # print('\r\n')
        # for i in range(height - 1, -1, -1):
        #     print("{0:4d}".format(i), end='')
        #     for j in range(width):
        #         loc = i * width + j
        #         p = metasurface.states.get(loc, -1)
        #         if p == player1:
        #             print('X'.center(8), end='')
        #         elif p == player2:
        #             print('O'.center(8), end='')
        #         else:
        #             print('_'.center(8), end='')
        #     print('\r\n\r\n')

    def start_play(self, player1, player2, player3, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1, 2):
            raise Exception('start_player should be either 0 (player1 first) '
                            ' 1 (player2 first) or 2 (player3 first)')
        self.metasurface.init_metasurface(start_player)
        p1, p2, p3 = self.metasurface.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        player3.set_player_ind(p3)
        players = {p1: player1, p2: player2, p3: player3}
        if is_shown:
            self.graphic(self.metasurface, player1.player, player2.player, player3.player)
        while True:
            current_player = self.metasurface.get_current_player()
            if current_player==1:
                player_element = '(𝞹/4)'
            elif current_player==2:
                player_element = '(𝞹/2)'
            else:
                player_element='(𝞹)'
            print('\n\n')
            print ('Awaiting Move from Player', current_player, player_element, '.....')
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.metasurface)
            self.metasurface.do_move(move)
            if is_shown:
                self.graphic(self.metasurface, player1.player, player2.player, player3.player)
            end, winner = self.metasurface.game_end()
            print ('Current RCS: ',self.metasurface.current_rcs_val)
            if end:
                if is_shown:
                    if winner != -1:
                        # if winner != 1:
                        #     print ('You Lost!!!')
                        # else:
                        #     print ('You Won!!!')
                        print ('Game end. Winner is Player', winner)
                        # print("Game end. Winner is", players[winner])   # this doesn't work if the same player is used more than once, everytime the winner would be player 3 (due to set_player_ind)
                        print ('Player 1 (𝞹/4) RCS minimum: ', np.min(self.metasurface.player1_rcs_array))
                        print ('Player 2 (𝞹/2) RCS minimum:  ', np.min(self.metasurface.player2_rcs_array))
                        print ('Player 3 (𝞹) RCS minimum:  ', np.min(self.metasurface.player3_rcs_array))
                        print("Best RCS state")
                        if winner==1:
                            print (pd.DataFrame(self.metasurface.player1_states[np.argmin(self.metasurface.player1_rcs_array)]).iloc[::-1])
                            # print (np.argmin(self.metasurface.player1_rcs_array))
                        elif winner==2:
                            print (pd.DataFrame(self.metasurface.player2_states[np.argmin(self.metasurface.player2_rcs_array)]).iloc[::-1])
                            # print (np.argmin(self.metasurface.player2_rcs_array))
                        else:
                            print (pd.DataFrame(self.metasurface.player3_states[np.argmin(self.metasurface.player3_rcs_array)]).iloc[::-1])
                            # print (len(self.metasurface.player3_states))
                            # print (self.metasurface.player3_states)
                            # print (np.argmin(self.metasurface.player3_rcs_array))
                        # print ('Player 1 cumulative RCS reduction: ', np.sum(self.metasurface.player1_rcs_red_array))
                        # print ('Player 2 cumulative RCS reduction: ', np.sum(self.metasurface.player2_rcs_red_array))
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """

        self.metasurface.init_metasurface()
        p1, p2, p3 = self.metasurface.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.metasurface,
                                                 temp=temp,
                                                 return_prob=1)
            # print (move,move_probs)
            # store the data
            states.append(self.metasurface.get_current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.metasurface.current_player)
            # perform a move
            self.metasurface.do_move(move)
            if is_shown:
                self.graphic(self.metasurface, p1, p2, p3)
            end, winner = self.metasurface.game_end()
            # print (len(self.metasurface.availables))
            # print (end, winner)
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)


# metasurface = Metasurface(width=8,height=8,total_moves=100)
# game = Game(metasurface)
# metasurface.init_metasurface()
# metasurface.do_move(10)
# metasurface.do_move(34)
# metasurface.do_move(52)
# # metasurface.do_move(52)
# print (metasurface.current_rcs_val)
# print (metasurface.prev_rcs_val)
# print (metasurface.current_rcs_red)
# game.graphic(metasurface,1,2)
# rcs_numpy = rcs.get_RCS_numpy(state)

# print (cur_st)
# print (rcs_numpy)