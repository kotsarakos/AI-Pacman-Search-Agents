# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getAvailableActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateNextState(agentIndex, action):
        Returns the nextState game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        def minimax(state, depth, agentIndex):
            # Terminal condition: win, lose, or max depth reached
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()

            # Pacman's turn (maximize)
            if agentIndex == 0:
                bestValue = float('-inf')
                for action in state.getAvailableActions(agentIndex):
                    successor = state.generateNextState(agentIndex, action)
                    value = minimax(successor, depth, 1)
                    bestValue = max(bestValue, value)
                return bestValue

            # Ghosts' turn (minimize)
            else:
                nextAgent = agentIndex + 1
                nextDepth = depth

                # If last agent, go back to Pacman and increase depth
                if nextAgent == numAgents:
                    nextAgent = 0
                    nextDepth += 1

                bestValue = float('inf')
                for action in state.getAvailableActions(agentIndex):
                    successor = state.generateNextState(agentIndex, action)
                    value = minimax(successor, nextDepth, nextAgent)
                    bestValue = min(bestValue, value)
                return bestValue

        # Initial call: Pacman chooses the best action
        bestScore = float('-inf')
        bestAction = None
        for action in gameState.getAvailableActions(0):
            successor = gameState.generateNextState(0, action)
            score = minimax(successor, 0, 1)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alphabeta(state, depth, agentIndex, alpha, beta):
            # Terminal condition
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()

            # Max node (Pacman)
            if agentIndex == 0:
                value = float('-inf')
                for action in state.getAvailableActions(agentIndex):
                    successor = state.generateNextState(agentIndex, action)
                    value = max(value, alphabeta(successor, depth, 1, alpha, beta))
                    if value > beta:  # prune
                        return value
                    alpha = max(alpha, value)
                return value

            # Min node (Ghosts)
            else:
                value = float('inf')
                nextAgent = agentIndex + 1
                nextDepth = depth

                # Loop back to Pacman and increase depth
                if nextAgent == numAgents:
                    nextAgent = 0
                    nextDepth += 1

                for action in state.getAvailableActions(agentIndex):
                    successor = state.generateNextState(agentIndex, action)
                    value = min(value, alphabeta(successor, nextDepth, nextAgent, alpha, beta))
                    if value < alpha:  # prune
                        return value
                    beta = min(beta, value)
                return value

        # Root call: Pacman's turn
        bestScore = float('-inf')
        bestAction = None
        alpha = float('-inf')
        beta = float('inf')

        for action in gameState.getAvailableActions(0):
            successor = gameState.generateNextState(0, action)
            score = alphabeta(successor, 0, 1, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)

        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        def expectimax(state, depth, agentIndex):
            # Terminal condition: win, lose, or max depth reached
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)  # Returns a number (float)

            numAgents = state.getNumAgents()

            # Pacman's turn (maximize)
            if agentIndex == 0:
                bestValue = float('-inf')
                for action in state.getAvailableActions(agentIndex):
                    successor = state.generateNextState(agentIndex, action)
                    value = expectimax(successor, depth, 1)
                    bestValue = max(bestValue, value)  # Update best value
                return bestValue  # Returns only the best score

            # Ghosts' turn (expectation - average value)
            else:
                nextAgent = agentIndex + 1
                nextDepth = depth

                # Loop back to Pacman and increase depth
                if nextAgent == numAgents:
                    nextAgent = 0
                    nextDepth += 1

                expectedValue = 0
                legalActions = state.getAvailableActions(agentIndex)
                prob = 1.0 / len(legalActions)  # Uniform distribution for ghosts

                for action in legalActions:
                    successor = state.generateNextState(agentIndex, action)
                    expectedValue += prob * expectimax(successor, nextDepth,
                                                       nextAgent)  # Make sure only float is returned
                return expectedValue  # Returns the expected value

        # In the main code block for the search
        bestScore = float('-inf')
        bestAction = None
        for action in gameState.getAvailableActions(0):
            successor = gameState.generateNextState(0, action)
            score = expectimax(successor, 0, 1)  # Takes only the score, not a tuple
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    from util import manhattanDistance

    # Get info from current game state about some positions
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    foodList = food.asList()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    walls = currentGameState.getWalls()
    score = currentGameState.getScore()

    # Start evaluation with current score
    evaluation = score

    # 1. Reward for being near scared ghosts
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()  # Ghost's position
        dist = manhattanDistance(pacmanPos, ghostPos)  # Distance to the ghost

        if ghost.scaredTimer > 0:  # Ghost is scared
            evaluation += max(10 - dist, 0)  # Reward for getting closer to scared ghost
        else:
            if dist < 3:  # Close to non-scared ghost
                # Check if there's a wall between Pacman and the ghost
                x, y = pacmanPos
                gx, gy = int(ghostPos[0]), int(ghostPos[1])
                wall_between = walls[min(x, gx)][min(y, gy)]
                if not wall_between:
                    evaluation -= (3 - dist) * 10  # Penalize for being too close

    # 2. Reward for fewer food left
    evaluation -= 5 * len(foodList)  # Penalize for remaining food

    # 3. Reward for being close to food
    if foodList:
        minFoodDist = min([manhattanDistance(pacmanPos, f) for f in foodList])  # Closest food
        evaluation += 5 / (minFoodDist + 1)  # Reward for being closer to food

    # 4. Reward for being close to capsules
    if capsules:
        minCapsuleDist = min([manhattanDistance(pacmanPos, c) for c in capsules])  # Closest capsule
        evaluation += 3 / (minCapsuleDist + 1)  # Reward for being closer to capsules

    return evaluation  # Return the final evaluation score

# Abbreviation
better = betterEvaluationFunction
