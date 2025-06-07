# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import Stack
from util import Queue
from util import PriorityQueue

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getInitialState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getNextStates(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (nextState,
        action, stepCost), where 'nextState' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getInitialState())
    print("Is the start a goal?", problem.isGoalState(problem.getInitialState()))
    print("Start's nextStates:", problem.getNextStates(problem.getInitialState()))
    """

    # Create a stack for DFS
    stack = Stack()

    # Set to keep track of visited states
    visited = set()

    # Start from the initial state with an empty path
    start = problem.getInitialState()
    stack.push((start, []))  # (current state, path taken)

    # Loop while there are states to explore
    while not stack.isEmpty():
        # Take the last state added (LIFO)
        state, path = stack.pop()

        # Check if we reached the goal
        if problem.isGoalState(state):
            return path  # Return the sequence of actions to the goal

        # If this state has not been visited before
        if state not in visited:
            visited.add(state)

            # Expand all next possible states
            for next_state, action, _ in problem.getNextStates(state):
                # Add the new state and the updated path to the stack
                new_path = path + [action]
                stack.push((next_state, new_path))

    # If no path found, return empty list
    return []


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    # Create a queue for BFS
    queue = Queue()

    # Set to keep track of visited states
    visited = set()

    # Start from the initial state with an empty path
    start = problem.getInitialState()
    queue.push((start, []))  # (current state, path taken)

    # Loop while there are states to explore
    while not queue.isEmpty():
        # Take the oldest state added (FIFO)
        state, path = queue.pop()

        # Check if we reached the goal
        if problem.isGoalState(state):
            return path  # Return the sequence of actions to the goal

        # If this state has not been visited before
        if state not in visited:
            visited.add(state)

            # Expand all next possible states
            for nextState, action, _ in problem.getNextStates(state):
                if nextState not in visited:
                    # Add the new state and the updated path to the queue
                    new_path = path + [action]
                    queue.push((nextState, new_path))

    # If no path found, return empty list
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    # Create a priority queue for UCS, which orders nodes by their total cost
    frontier = PriorityQueue()

    # Get the initial state of the problem and push it onto the frontier with a cost of 0
    start_state = problem.getInitialState()
    frontier.push((start_state, []), 0)  # The tuple contains the state and the current path, with cost 0

    # A dictionary to keep track of the cost to reach each state
    cost_so_far = {start_state: 0}

    # A set to keep track of visited states, to avoid revisiting the same state
    visited = set()

    # While there are states to explore in the frontier
    while not frontier.isEmpty():
        # Pop the state with the least cost (the one with the smallest accumulated cost)
        state, path = frontier.pop()

        # If we have reached the goal state, return the sequence of actions (path)
        if problem.isGoalState(state):
            return path

        # If this state has not been visited before, process it
        if state not in visited:
            visited.add(state)

            # Expand the current state to all possible next states
            for nextState, action, stepCost in problem.getNextStates(state):
                # Calculate the new cost to reach this next state
                new_cost = cost_so_far[state] + stepCost

                # Only consider this path if it's the first time we've reached this state
                # or if we've found a cheaper way to get to this state
                if nextState not in cost_so_far or new_cost < cost_so_far[nextState]:
                    # Update the cost to reach this state
                    cost_so_far[nextState] = new_cost
                    # Push the next state into the frontier with the updated path and cost
                    frontier.push((nextState, path + [action]), new_cost)

    # If no solution is found, return an empty list (the problem should be solvable)
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    # Start state
    start_state = problem.getInitialState()

    # Priority queue: f(n) = g(n) + h(n)
    frontier = PriorityQueue()
    frontier.push((start_state, [], 0), heuristic(start_state, problem))

    # Track visited states
    visited = set()
    cost_so_far = {start_state: 0}

    while not frontier.isEmpty():
        state, path, g = frontier.pop()

        # Goal check
        if problem.isGoalState(state):
            return path

        if state not in visited:
            visited.add(state)

            for nextState, action, stepCost in problem.getNextStates(state):
                new_cost = g + stepCost

                # If new or better path found
                if nextState not in cost_so_far or new_cost < cost_so_far[nextState]:
                    cost_so_far[nextState] = new_cost
                    priority = new_cost + heuristic(nextState, problem)
                    frontier.push((nextState, path + [action], new_cost), priority)

    # No solution found
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
