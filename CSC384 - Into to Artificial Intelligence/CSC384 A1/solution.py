#   Look for #IMPLEMENT tags in this file. These tags indicate what has
#   to be implemented to complete the warehouse domain.

#   You may add only standard python imports
#   You may not remove any imports.
#   You may not import or otherwise source any of your own files

import os  # for time functions
import math  # for infinity

from search import *  # for search engines
from sokoban import sokoban_goal_state, SokobanState, Direction, PROBLEMS  # for Sokoban specific classes and problems

previous_box_distance = None
# SOKOBAN HEURISTICS
def heur_alternate(state):
    # IMPLEMENT
    '''a better heuristic'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''
    # heur_manhattan_distance has flaws.
    # Write a heuristic function that improves upon heur_manhattan_distance to estimate distance between the current state and the goal.
    # Your function should return a numeric value for the estimate of the distance to the goal.
    # EXPLAIN YOUR HEURISTIC IN THE COMMENTS. Please leave this function (and your explanation) at the top of your solution file, to facilitate marking.

    # This is the distance of the previous states boxes to their closest storage
    global previous_box_distance
    dist = 0
    # These are all the boxes that are not at a storage location
    boxes = [box for box in state.boxes if box not in state.storage]

    # We first check is all the boxes from the parent state are the same as this state
    # If they are and the previous box distance is infinite we return infinite to reduce computation time
    # If not we find the distance from the robots to the box and add that to the previous boxes distance
    if (state.parent is not None) and (state.boxes == state.parent.boxes):
        if previous_box_distance == float('inf'):
            return float('inf')
        else:
            for box in boxes:
                dist = previous_box_distance + min([abs(box[0]-robot[0])+abs(box[1]-robot[1]) for robot in state.robots])
            return dist

    # These are the storage locations on the y and x axis
    storage_1 = [storage[1] for storage in state.storage]
    storage_0 = [storage[0] for storage in state.storage]

    # We iterate through all the boxes not in a storage location
    for box in boxes:
        # We can first check if there are any boxes that cannot reach a storage location
        # We do this by first checking if any box has its top or bottom and one of its side next to a wall, or obstacle
        # If it is we return infinite as it can never reach the goal state
        top, bottom, left, right = (box[0], box[1] + 1), (box[0], box[1] - 1), (box[0] - 1, box[1]), (box[0] + 1, box[1])
        top_move = top in state.obstacles or box[1] == (state.height - 1)
        bottom_move = bottom in state.obstacles or box[1] == 0
        left_move = left in state.obstacles or box[0] == 0
        right_move = right in state.obstacles or box[0] == (state.width - 1)
        if (top_move or bottom_move) and (left_move or right_move):
            previous_box_distance = float('inf')
            return float('inf')
        # Next we check if the box is cornered by another box and the edges
        # If so we return infinite as the goal state can never be reached
        top_box = top in state.boxes
        bottom_box = bottom in state.boxes
        left_box = left in state.boxes
        right_box = right in state.boxes
        if((box[1] == (state.height - 1) or box[1] == 0) and (left_box or right_box)) or ((top_box or bottom_box) and (box[0] == 0 or box[0] == (state.width - 1))):
            previous_box_distance = float('inf')
            return float('inf')
        # We can now look at if there is a box on the wall and no storage on that wall
        # If so we return infinite as the puzzle cannot be solved
        if (box[1] == (state.height - 1) or box[1] == 0) and (box[1] not in storage_1):
            previous_box_distance = float('inf')
            return float('inf')
        if (box[0] == (state.width - 1) or box[0] == 0) and (box[0] not in storage_0):
            previous_box_distance = float('inf')
            return float('inf')

        # We have now seen edge cases where the puzzle cannot be solved

        # We can now take the Manhattan Distance from the box to its nearest storage
        dist += min([abs(box[0]-storage[0])+abs(box[1]-storage[1]) for storage in state.storage])
        previous_box_distance = dist
        # We can also take the Manhattan Distance from the box to its nearest robot
        dist += min([abs(box[0]-robot[0])+abs(box[1]-robot[1]) for robot in state.robots])
    return dist  # CHANGE THIS


def heur_zero(state):
    '''Zero Heuristic can be used to make A* search perform uniform cost search'''
    return 0

def heur_manhattan_distance(state):
    # IMPLEMENT
    '''admissible sokoban puzzle heuristic: manhattan distance'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''
    # We want an admissible heuristic, which is an optimistic heuristic.
    # It must never overestimate the cost to get from the current state to the goal.
    # The sum of the Manhattan distances between each box that has yet to be stored and the storage point nearest to it is such a heuristic.
    # When calculating distances, assume there are no obstacles on the grid.
    # You should implement this heuristic function exactly, even if it is tempting to improve it.
    # Your function should return a numeric value; this is the estimate of the distance to the goal.
    dist = 0
    for box in state.boxes:
        minimum_dist = float('inf')
        for storage in state.storage:
            temp = (abs(box[0]-storage[0])+abs(box[1]-storage[1]))
            if temp < minimum_dist:
                minimum_dist = temp
        dist += minimum_dist
    return dist

def fval_function(sN, weight):
    # IMPLEMENT
    """
    Provide a custom formula for f-value computation for Anytime Weighted A star.
    Returns the fval of the state contained in the sNode.

    @param sNode sN: A search node (containing a SokobanState)
    @param float weight: Weight given by Anytime Weighted A star
    @rtype: float
    """
    return sN.gval + (weight * sN.hval)

# SEARCH ALGORITHMS
def weighted_astar(initial_state, heur_fn, weight, timebound):
    # IMPLEMENT
    '''Provides an implementation of weighted a-star, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False as well as a SearchStats object'''
    '''implementation of weighted astar algorithm'''
    wrapped_fval_function = (lambda sN: fval_function(sN, weight))
    search_engine = SearchEngine('custom', 'default')
    search_engine.init_search(initial_state, sokoban_goal_state, heur_fn, wrapped_fval_function)
    final, stats = search_engine.search(timebound)
    return final, stats  # CHANGE THIS

def iterative_astar(initial_state, heur_fn, weight=1., timebound=5):  # uses f(n), see how autograder initializes a search line 88
    # IMPLEMENT
    '''Provides an implementation of realtime a-star, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False as well as a SearchStats object'''
    '''implementation of iterative astar algorithm'''
    wrapped_fval_function = (lambda sN: fval_function(sN, weight))
    search_engine = SearchEngine('custom', 'default')
    search_engine.init_search(initial_state, sokoban_goal_state, heur_fn, wrapped_fval_function)
    final = False
    cost = (float('inf'), float('inf'), float('inf'))
    time = os.times()[0]
    end = time + timebound
    result, stats = search_engine.search(end - time)
    while time < end:
        wrapped_fval_function = (lambda sN: fval_function(sN, weight))
        search_engine.init_search(initial_state, sokoban_goal_state, heur_fn, wrapped_fval_function)
        time = os.times()[0]
        if result is not False and result.gval <= cost[2] :
            final = result
            cost = (float('inf'), float('inf'), final.gval)
        if weight > 1:
            weight -= 1
        result, stats = search_engine.search(end-time, costbound=cost)

    return final, stats  # CHANGE THIS

def iterative_gbfs(initial_state, heur_fn, timebound=5):  # only use h(n)
    # IMPLEMENT
    '''Provides an implementation of anytime greedy best-first search, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False'''
    '''implementation of iterative gbfs algorithm'''
    time = os.times()[0]
    end = time + timebound
    search_engine = SearchEngine('best_first', 'default')
    search_engine.init_search(initial_state, sokoban_goal_state, heur_fn)
    final = False
    cost = (float('inf'), float('inf'), float('inf'))
    result, stats = search_engine.search(end - time)
    while time < end:
        if result is not False and result.gval <= cost[0]:
            final = result
            cost = (final.gval, float('inf'), float('inf'))
        time = os.times()[0]
        result, stats = search_engine.search(end-os.times()[0], costbound=cost)

    return final, stats  # CHANGE THIS




