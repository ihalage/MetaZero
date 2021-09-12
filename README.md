# MetaZero
MetaZero - A reinforcement learning program to optimise metasurface radar cross section (RCS) by self-playing. Uses Monte-Carlo tree search with rollout to find strong moves. Inspired by AlphaZero.

## Game definition 
    * Minimise the radar cross section (RCS) of an NxN metasurface using L number of coding unit cell elements
    
    * Four coding elements ('00', '01', '10', '11') corresponding to 0, pi/4, pi/2, pi phase responses are used here
    
    * Metasurface is initialized with all '0's
    
    * Each player can play any available position
    
    * A human can play the optimisation game too. By default, the RL agent is set to self-play mode
    
    * Game terminates after all unit cells are filled
    
    * The player who has recorded the minimum cumulative RCS reduction after a move wins
      (after each move, the current RCS will be lower or higher than the previous RCS.
      Thus, the reduction of RCS of each player at each move is calculated.
      The player that has recorded the minimum cumulative RCS reduction at the end wins.)

## Dependencies

* python-2.7
* tensorflow
* keras
* numpy
* pandas
## 

To run a self-play cycle with the trained model (N=6), run;

    python auto_play.py

You will observe that the RCS is decreasing in each move.

    Awaiting Move from Player 1 (𝞹/4) .....
         0  1    2  3    4    5
    5    0  0    0  0  𝞹/4    0
    4  𝞹/2  0    𝞹  0    0  𝞹/2
    3    𝞹  0    0  0    0    0
    2    0  0  𝞹/4  0    0    0
    1    0  0    0  0    0    0
    0    0  0    0  0    0    0

    Current RCS:  0.00598904616544579

    Awaiting Move from Player 2 (𝞹/2) .....
         0  1    2  3    4    5
    5    0  0    0  0  𝞹/4    0
    4  𝞹/2  0    𝞹  0  𝞹/2  𝞹/2
    3    𝞹  0    0  0    0    0
    2    0  0  𝞹/4  0    0    0
    1    0  0    0  0    0    0
    0    0  0    0  0    0    0

    Current RCS:  0.005868425074941015

    Awaiting Move from Player 3 (𝞹) .....
         0  1    2  3    4    5
    5    0  0    0  0  𝞹/4    0
    4  𝞹/2  0    𝞹  0  𝞹/2  𝞹/2
    3    𝞹  0    0  0    0    0
    2    0  0  𝞹/4  0    0    0
    1    0  0    0  0    0    0
    0    𝞹  0    0  0    0    0

    Current RCS:  0.005255953496455249

    Awaiting Move from Player 1 (𝞹/4) .....
         0    1    2  3    4    5
    5    0  𝞹/4    0  0  𝞹/4    0
    4  𝞹/2    0    𝞹  0  𝞹/2  𝞹/2
    3    𝞹    0    0  0    0    0
    2    0    0  𝞹/4  0    0    0
    1    0    0    0  0    0    0
    0    𝞹    0    0  0    0    0

    Current RCS:  0.005245027383191566
    .
    .
    .
    
MetaZero can be trained from scratch as follows. This will overwrite the existing model.

    python train.py
  
## Reference

Code implementation is inspired by [AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku).
  
