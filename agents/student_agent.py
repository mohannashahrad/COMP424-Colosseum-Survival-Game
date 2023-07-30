# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
from agents.MCT_search import *
import sys
import time


@register_agent("student_agent")
class StudentAgent(Agent):
    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.tree = MCT()
        self.autoplay = True

    def step(self, chess_board, my_pos, adv_pos, max_step):
        board_size = len(chess_board)
        next_move = self.tree.run_tree({"board": chess_board, "my_position": my_pos, "adv_position": adv_pos}, board_size, max_step)
        return next_move[:2], next_move[2]


