import sys
import time 
import random
import numpy as np
from copy import deepcopy, copy
from agents import student_agent
from ui import UIEngine
import click

class Node: 
    def __init__(self, state, parent, move, turn):
        self.state = state
        self.parent = parent
        self.wins = 0
        self.number_of_visits = 0
        self.children = []
        self.move = move
        self.toward_adv_moves = -1
        self.neutral_moves = -1
        self.away_adv_moves = -1
        self.turn = turn
        self.heuristic = 0

class MCT: 
    def __init__(self):
        self.board_size = 0
        self.max_step = 0
        self.first = True
        self.duration = 29.8  # For the first round we have 30 seconds to do the pre-processing

    def run_tree(self, state, board_size, max_step):
        start_time = time.time()
        flag= False
        if not self.first:
            self.duration = 1.9
            self.root = Node(state, None, None, True) 
        else :
            self.first = not self.first
            self.root = Node(state, None, None, True)    
        self.curr = self.root
        self.board_size = board_size
        self.max_step = max_step
        node = self.curr

        # Setting the heuristic of the root before starting the process
        walls = 0
        pos_x, pos_y = node.state['my_position']
        for i in range(4) :
            if node.state['board'][pos_x,pos_x,i]:
                walls += 1
        node.heuristic = -1*walls

        # This is the computational power limit
        while time.time() - start_time < self.duration:
            selected_node = self.selection(node)
            terminal_leaf = self.expansion(selected_node)
            score = self.rollout(terminal_leaf)
            if terminal_leaf.turn:
                self.backpropagate(terminal_leaf, score*-1)
            else:
                self.backpropagate(terminal_leaf, score)
            
        result = self.get_highest_visited_child(node)
        self.root = result
        return result.move

    def calculateUCB(self, node): 
        return node.wins/node.number_of_visits + 2*np.sqrt(np.log(node.parent.number_of_visits)/node.number_of_visits)    

    def selection(self, node):
        if len(node.children) == 0:     # If node is a terminal
            return node

        while True: 
            bestucb = 0
            bestNode = node

            for c in node.children:         # Find the child with max ucb
                if c.number_of_visits != 0:
                    # ucb = self.calculateUCB(c) 
                    ucb = self.calculateUCB(c) + ((0.05*c.heuristic)/c.number_of_visits)
                else:
                    ucb = 1
                if ucb > bestucb:
                    bestucb = ucb
                    bestNode = c
            if bestNode == node:
                break 
            node = bestNode
            if not node.children: 
                break 
        return node      

    def expansion(self, leaf):
        if leaf.number_of_visits != 0:
            if self.check_endgame(self.board_size, leaf.state['board'], leaf.state['my_position'],leaf.state['adv_position']) == -1:
                if not leaf.children:
                    self.createChildren(leaf)
                if leaf.turn and leaf.toward_adv_moves != -1:
                    try:
                        leaf = random.choices(population=leaf.children, 
                        weights=[35] * leaf.toward_adv_moves + [45] * leaf.neutral_moves + [20] * leaf.away_adv_moves,
                        k=1)[0]
                    except:
                        leaf = np.random.choice(leaf.children)
                else:    
                    leaf = np.random.choice(leaf.children)
        return leaf

    def rollout(self, leaf):
        tmp = Node(leaf.state, None, None, leaf.turn)

        # doing a sample playout from leaf
        while self.check_endgame(self.board_size, tmp.state['board'],
        tmp.state['my_position'], tmp.state['adv_position']) == -1: 
            if not tmp.children:
                self.createChildren(tmp) 
            if tmp.turn and tmp.toward_adv_moves != -1:
                try:
                    tmp = random.choices(population=tmp.children, 
                    weights=[35] * tmp.toward_adv_moves + [45] * tmp.neutral_moves + [20] * tmp.away_adv_moves,
                    k=1)[0]
                except: 
                    tmp = np.random.choice(tmp.children)
            else:    
                tmp = np.random.choice(tmp.children)  

        result = self.check_endgame(self.board_size,tmp.state['board'], tmp.state['my_position'], tmp.state['adv_position'])

        if result == 0:         # Tie
            score = 0
        elif result == 1:       # I won 
            score = 1
        else:
            score = -1          # adv won
        return score

    def backpropagate(self, node, score):
        node.wins += score 
        node.number_of_visits += 1
        if node.parent:
            self.backpropagate(node.parent, score*-1) # Not sure about updating sore 
        else:
            return  

    def get_highest_visited_child(self, node):
        max_visits = 0 
        result = None
        if len(node.children) == 0:
            print("Seems like a problem, no children here")
        for c in node.children:
            if c.number_of_visits > max_visits:
                max_visits = c.number_of_visits
                result = c
        return result    

    def createChildren(self, node):
        # If it is our turn in node
        if node.turn:
            adv_position = node.state['adv_position']
            possible_actions, toward, neutral, away = self.get_possible_actions(node.state['board'], node.state['my_position'], node.state['adv_position'], self.max_step)
        # If it is adv turn in node
        else:
            adv_position = node.state['my_position']
            possible_actions, toward, neutral, away = self.get_possible_actions(node.state['board'], node.state['adv_position'], node.state['my_position'], self.max_step)

        # Updating the nodes next children
        node.toward_adv_moves = toward
        node.neutral_moves = neutral
        node.away_adv_moves = away


        if node.turn:
            for move in possible_actions:
                board = copy(node.state['board'])
                self.set_barrier(move[0],move[1],move[2],board)
                new_child = Node({"board": board, "my_position": move[:2], "adv_position": adv_position}, node, move, False)
                walls = 0
                pos_x, pos_y = move[:2]
                for i in range(4) :
                    if board[pos_x,pos_x,i]:
                        walls += 1
                new_child.heuristic = -1*walls
                node.children.append(new_child)
        else: 
            for move in possible_actions:
                board = copy(node.state['board'])
                self.set_barrier(move[0],move[1],move[2],board)
                new_child = Node({"board": board, "my_position": adv_position, "adv_position": move[:2]}, node, move, True)
                walls = 0
                pos_x, pos_y = move[:2]
                for i in range(4) :
                    if board[pos_x,pos_x,i]:
                        walls += 1
                new_child.heuristic = -1*walls
                node.children.append(new_child)  
        return

    def get_possible_actions (self, chess_board, my_pos, adv_pos, max_step):
        # actions that take us towards the adversary will be at the start of the list
        towards_adv = []
        neutral = []
        away_adv = []
        
        towards_walls = []
        neutral_walls = []
        away_walls = []
        
        
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        explore_states = [(my_pos, max_step)]
        my_x, my_y = my_pos
        adv_x, adv_y = adv_pos
        
        # check for a win possibility (adversary enclosed by three walls)
        adv_walls = 0
        adv_enclosed = []
        for i in range(4):
            if chess_board[adv_x, adv_y, i]:
                adv_walls += 1
            else:
                adv_enclosed = i
        if adv_walls == 3:
            # must look for a way to reach the position that would enclose the adversary
            x, y = moves[adv_enclosed]
            adv_enclosed = (adv_x + x, adv_y + y, opposites[adv_enclosed])
            
        # get information about the quadrant of the adversary based on our position
        x_diff = adv_x - my_x
        y_diff = adv_y - my_y
        
        while explore_states:
            pos, step = explore_states.pop(0)
            if pos == adv_pos or pos in towards_adv or pos in away_adv or pos in neutral:
                continue
            
            if (adv_walls == 3 and pos == (adv_enclosed[0], adv_enclosed[1])):
                # we have found the move that would enclose the adversary in a 1x1 square
                return [adv_enclosed], -1, -1, -1
            
            x,y = pos
            # positions that take us to the same quadrant as the adversary
            if (self.is_right_direction(my_pos, pos, x_diff, y_diff)):
                towards_adv.append(pos)
                wall_count = 0
                positions = []
                
                for i in range(4):
                    # find out how many walls are present in this position
                    if chess_board[x,y,i]:
                        wall_count += 1
                    else:
                        positions.append((x,y,i)) 
                        
                if(wall_count == 2):
                    # We would be adding 3rd wall at this position, promote this position less 
                    neutral_walls.extend(positions)
                elif(wall_count < 2):
                    towards_walls.extend(positions)
                else:
                    #bad idea to reject walls outright, so put them in lowest probability category
                    away_walls.extend(positions)
                
            else:
                # positions that take us to the opposite quadrant
                if (self.is_wrong_direction(my_pos, pos, x_diff, y_diff)):
                    away_adv.append(pos)
                    wall_count = 0
                    positions = []
                    
                    for i in range(4):
                        if chess_board[x,y,i]:
                            wall_count += 1
                        else:
                            away_walls.append((x,y,i)) 
                
                # positions that take us to adjacent quadrants
                else:
                    neutral.append(pos)
                    wall_count = 0
                    positions = []
                    
                    for i in range(4):
                        if chess_board[x,y,i]:
                            wall_count += 1
                        else:
                            positions.append((x,y,i))
                    if(wall_count < 2):
                        neutral_walls.extend(positions)
                    else:
                        away_walls.extend(positions)
            
            if step != 0:
                for i in range(4):
                    x,y = pos
                    step_x, step_y = moves[i]
                    # check for wall in chosen direction
                    if not chess_board[x,y,i]: 
                        new_pos = (x + step_x, y + step_y)
                        explore_states.append((new_pos, step-1))   
        
        towards_len = len(towards_walls)
        neutral_len = len(neutral_walls)
        away_len = len(away_walls)
        
        towards_walls.extend(neutral_walls)
        towards_walls.extend(away_walls)
        
        return towards_walls, towards_len, neutral_len, away_len
    
    def is_right_direction(self, my_pos, pos, x_diff, y_diff):
        my_x, my_y = my_pos
        x, y = pos
        
        if x_diff == 0:
            return (y - my_y >= 0) == (y_diff >= 0)
        elif y_diff == 0:
            return (x - my_x >= 0) == (x_diff >= 0)
        else :
            return ((y - my_y >= 0) == (y_diff >= 0)) and ((x - my_x >= 0) == (x_diff >= 0))

    def is_wrong_direction(self, my_pos, pos, x_diff, y_diff):
        my_x, my_y = my_pos
        x, y = pos
        
        if x_diff == 0:
            return (y - my_y >= 0) != (y_diff >= 0)
        elif y_diff == 0:
            return (x - my_x >= 0) != (x_diff >= 0)
        else :
            return ((y - my_y >= 0) != (y_diff >= 0)) and ((x - my_x >= 0) != (x_diff >= 0))

    # Retun values 
    # If not finished yet: -1
    # If first player wins: 1
    # If second player wins: 2
    # If tie happens: 0
    def check_endgame(self, board_size, chess_board, p0_pos, p1_pos, test=False):
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        # Union-Find
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(moves[1:3]):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    if test:
                        print(f"Father of pos_a {father[pos_a]}")
                    pos_b = find((r + move[0], c + move[1]))
                    if test:
                        print(f"Father of pos_b {father[pos_b]}")
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(p0_pos))
        p1_r = find(tuple(p1_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if test:
            print(p0_pos)
            print(p1_pos)
            print(chess_board)
            print(p0_r)
            print(p1_r)
        if p0_r == p1_r:
            return -1
        if p0_score > p1_score:
            return 1
        elif p0_score < p1_score:
            return 2
        else:
            return 0

    def set_barrier(self, r, c, dir, chess_board):
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        # Set the barrier to True
        chess_board[r, c, dir] = True
        # Set the opposite barrier to True
        move = moves[dir]
        chess_board[r + move[0], c + move[1], opposites[dir]] = True      
