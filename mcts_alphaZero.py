
import numpy as np
import copy
from tqdm import tqdm
from tools import *
np.set_printoptions(suppress=True)

class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while 1:
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            if action in state.availables:
                state.do_move(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_probs, leaf_value = self._policy(state)
        # Check for end of game.
        end, winner = state.game_end()
        # p1, p2 = state.players
        # graphic(board=state, player1=p1, player2=p2)
        # print(leaf_value)
        if not end:
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=0.07, n_playout=-1, verbose=True):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        num_of_playout = n_playout if n_playout > 0 else self._n_playout
        if verbose:
             iterator = tqdm(range(num_of_playout))
        else:
             iterator = range(num_of_playout)

        for n in iterator:
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def update_policy_function(self, fun):
        self._policy = fun

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, is_selfplay=False, mcts=None, verbose=True):
        self.mcts = mcts
        self._is_selfplay = is_selfplay
        self.verbose = verbose
        self.set_player_name()

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def set_player_name(self, name="MCTS_AI"):
        self.name = name

    def update_mcts(self, new_mcts):
        self.mcts = new_mcts

    def get_action(self, board, *args, **kwargs):
        temp = kwargs.get("temp", 0.07)
        return_prob = kwargs.get("return_prob", False)
        is_first_step = kwargs.get("is_first_step", False)
        n_playout = kwargs.get("n_playout", -1)
        current_step = kwargs.get("current_step", -1)

        sensible_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width * board.height)
        move_probs_print = np.zeros(board.width * board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp, n_playout, verbose=self.verbose)
            move_probs[list(acts)] = probs
            move_probs_print = move_probs

            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for self-play training)
                if move_probs[np.argmax(move_probs)] >= 0.8:
                    if current_step >= 0:
                        if current_step < 5:
                            if move_probs[np.argmax(move_probs)] >= 0.9:
                                prob_temp = 0.9
                                probs = prob_temp * probs + (1 - prob_temp) * np.random.dirichlet(0.3 * np.ones(len(probs)))
                                move_probs_print[list(acts)] = probs
                            else:
                                prob_temp = 0.9
                                probs = prob_temp * probs + (1 - prob_temp) * np.random.dirichlet(0.3 * np.ones(len(probs)))
                                move_probs_print[list(acts)] = probs

                    move = np.random.choice(acts, p=probs)
                else:
                    prob_temp = 1
                    # 让棋局的前6步的随机性更高，可以探索更多不一样的棋局
                    if current_step >= 0:
                        if current_step < 5:
                            prob_temp = 1
                        else:
                            prob_temp = 1

                    # 在每一步的概率里面加上随机数
                    probs = prob_temp * probs + (1 - prob_temp) * np.random.dirichlet(0.3 * np.ones(len(probs)))
                    move_probs_print[list(acts)] = probs
                    move = np.random.choice(acts, p=probs)

                if is_first_step:
                    # move = int(((board.width * board.height) - 1) / 2)
                    pass

                # 打印出每个位置的概率
                if self.verbose:
                    tabulator_probs(move_probs_print, board, move, np.argmax(move_probs_print))
                    print("AI")
                    print(move, move_probs_print[move], np.argmax(move_probs_print), move_probs_print[np.argmax(move_probs_print)])
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # temp为1e-3时，几乎相当选择概率最高的移动

                if move_probs[np.argmax(move_probs)] >= 0.8:
                    move = np.argmax(move_probs)
                else:
                    move = np.random.choice(acts, p=probs)

                if is_first_step:
                    move = int(((board.width * board.height) - 1) / 2)

                # 打印出每个位置的概率
                if self.verbose:
                    tabulator_probs(move_probs, board, move, np.argmax(move_probs))
                # reset the root node
                self.mcts.update_with_move(-1)
                location = board.move_to_location(move)
                # print("AI move: %d,%d\n" % (location[0], location[1]))

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
