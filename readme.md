### One Sided Incomplete Information Game

This repo simulates a simple one-sided incomplete information game where the player 1 has private information about its
type. Player 2, on the other hand, has no private information. Nature picks the type for player 1 from some distribution
at the beginning of the game, and communicates this information to both players. As the game progresses, the initial
distribution is updated. 

### Code Structure
- `game_settings.py` defines the game -- state space, payoffs and so on. 
- `solver.py` solves an optimization problem to compute the strategy
- `value_functions.py` contains value functions for complete and incomplete information game
- `simulate_game.py` simulates one game
- `utils.py` contains utility functions

