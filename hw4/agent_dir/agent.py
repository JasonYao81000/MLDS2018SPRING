"""

### NOTICE ###
You DO NOT need to upload this file

"""
from environment import Environment

class Agent(object):
    def __init__(self, env):
        self.env = env


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        This function must exist in agent

        Input:
            When running dqn:
                observation: np.array
                    stack 4 last preprocessed frames, shape: (84, 84, 4)

            When running pg:
                observation: np.array
                    current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        raise NotImplementedError("Subclasses should implement this!")


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        raise NotImplementedError("Subclasses should implement this!")
