
import random
from Data.hparam import Data

import numpy as np
from tools import *


class Board(object):
    """游 戏 板"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 11))
        self.height = int(kwargs.get('height', 11))
        self.n_playout = kwargs.get('n_playout', -1)
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2
        self.board = np.zeros(self.width * self.height)
        # mcts的动态搜索次数
        self.dynamic_n_playout = int(self.width * self.height * 3.2)
        self.base_dynamic_n_playout = self.dynamic_n_playout

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        self.board = np.zeros(self.width * self.height)
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1
        self.dynamic_n_playout = int(self.width * self.height * 3.2)

    def update_n_playout(self, current_step, is_in_game=False):
        if is_in_game:
            if int(self.base_dynamic_n_playout * (1 + current_step / 30)) < Data.mcts.dynamic_n_playout_max:
                self.dynamic_n_playout = int(self.base_dynamic_n_playout * (1 + current_step / 30))
            else:
                self.dynamic_n_playout = Data.mcts.dynamic_n_playout_max
        else:
            if int(self.base_dynamic_n_playout * (1 + current_step / 30)) < Data.mcts.dynamic_n_playout_max_train:
                self.dynamic_n_playout = int(self.base_dynamic_n_playout * (1 + current_step / 30))
            else:
                self.dynamic_n_playout = Data.mcts.dynamic_n_playout_max_train

    def get_n_playout(self):
        if self.n_playout == -1:
            return self.dynamic_n_playout
        else:
            return self.n_playout

    def move_to_location(self, move):
        """
        3*3 board's moves like:
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

    def current_state(self):
        """从当前玩家的角度返回棋盘状态.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.board[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """检查游戏是否结束"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """游戏 server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def start_play(self, player1, player2, start_player=0, is_shown=1, is_in_train=False):
        """在两个玩家之间开始游戏"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        # 初始化棋盘
        self.board.init_board(start_player)
        # 获得到两个玩家
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        # 当前步数
        currentStep = 1
        is_first_step = True
        if is_shown:
            if is_in_train:
                print("正在检测模型性能...")
            graphic(self.board, player1.player, player2.player)
        while True:
            # 获得当前玩家
            current_player = self.board.get_current_player()
            # 拿到玩家的对象
            player_in_turn = players[current_player]
            # 获得位置
            move = player_in_turn.get_action(self.board, n_playout=self.board.get_n_playout(), is_first_step=is_first_step)
            if is_first_step:  is_first_step = False
            self.board.do_move(move)
            self.board.update_n_playout(currentStep)
            if is_shown:
                graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=1, temp=0.07, current_inning=-1):
        """ 使用MCTS玩家开始自对局游戏, 重用搜索树,
        并存储自对局数据: (state, mcts_probs, z) 用以训练
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        # 当前步数
        current_step = 1
        # 是否是第一步
        is_first_step = True
        while True:
            if current_step <= 6:
                current_temp = 1
            else:
                current_temp = temp
            move, move_probs = player.get_action(self.board, temp=current_temp, return_prob=True, is_first_step=is_first_step, n_playout=self.board.get_n_playout(), current_step=current_step)
            if is_first_step:  is_first_step = False
            # 存储数据
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # 执行移动
            self.board.do_move(move)
            # 对于当前步数更新搜索次数
            self.board.update_n_playout(current_step=current_step)
            current_step += 1
            if is_shown:
                print("当前是第：", current_inning, "局，第：", current_step, "步")
                graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # 从每个状态的当前玩家的角度来看赢家
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # 重置MCTS根节点
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)

    def start_train_with_person(self, human_player, ai_player, temp=1e-3, start_player=0, is_train_with_human_human=False, is_train_with_robot=False, current_inning=-1, random_choise_start=False):
        if random_choise_start:
            start_player = random.choice([0, 1])
        # 初始化棋盘
        self.board.init_board(0)
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        # 现在是否是第一步
        is_first_step = True
        ai_player.set_player_ind(p1)
        human_player.set_player_ind(p2)
        if start_player == 0:
            players = {p1: human_player, p2: ai_player}
        else:
            players = {p1: ai_player, p2: human_player}

        # 显示棋盘
        graphic(self.board, ai_player.player, human_player.player)
        current_step = 1  # 当前步数
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            print("当前玩家：", current_player)
            # 获得一个位置
            move, move_probs = player_in_turn.get_action(self.board, temp=temp, return_prob=True,
                                                         is_first_step=is_first_step,
                                                         n_playout=self.board.get_n_playout())
            if is_first_step:
                is_first_step = False
            # 存储数据
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # 执行移动
            self.board.do_move(move)
            # 对于当前步数更新搜索次数
            self.board.update_n_playout(current_step=current_step)
            # 显示移动
            graphic(self.board, ai_player.player, human_player.player)
            if current_inning >= 0 and is_train_with_robot:
                print("当前是训练的第:", current_inning, "局，第: ", current_step, "步")
            # 判断是否结束对局
            end, winner = self.board.game_end()
            current_step += 1
            if end:
                # 从每个状态的当前玩家的角度来看赢家
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                if not is_train_with_human_human:
                    # 重置MCTS根节点
                    ai_player.reset_player()
                if winner != -1:
                    print("Game end. Winner is player:", winner)
                else:
                    print("Game end. Tie")

                return winner, zip(states, mcts_probs, winners_z)
