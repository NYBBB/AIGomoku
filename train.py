
from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
from policy_value_net_tensorflow import PolicyValueNet  # Tensorflow
from mcts_alphaZero import MCTS
from tools import *
from robot import Robot


# from policy_value_net_keras import PolicyValueNet # Keras


class human_player_train(object):
    def __init__(self, policyValueNet, mcts):
        self.player = None
        self.temp = 1.0
        self.n_playout = 1000
        self.c_puct = 5
        self.mcts = mcts

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board,  *args, **kwargs):
        temp = kwargs.get("temp", 0.07)
        n_playout = kwargs.get("n_playout", -1)

        move_probs = np.zeros(board.width * board.height)

        while True:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]

            if location[0] > board.width:
                location = input("Your move: ")
                if isinstance(location, str):  # for python3
                    location = [int(n, 10) for n in location.split(",")]
            elif location[1] > board.height:
                location = input("Your move: ")
                if isinstance(location, str):  # for python3
                    location = [int(n, 10) for n in location.split(",")]

            move = board.location_to_move(location)
            if move in board.availables:
                break

        action, prob = self.mcts.get_move_probs(board, temp=temp, n_playout=n_playout)
        move_probs[list(action)] = prob
        tabulator_probs(move_probs, board, move, np.argmax(move_probs))
        print("human")
        print(move, move_probs[move], np.argmax(move_probs))

        self.mcts.update_with_move(move)

        return move, move_probs

    def __str__(self):
        return "Human {}".format(self.player)

class robot_in_train:
    def __init__(self, mcts=None):
        self.mcts = mcts
        self.robot = Robot(11, 11)
        self.set_player_name()

    def set_player_ind(self, p):
        self.player = p

    def set_player_name(self, name="Robot"):
        self.name = name

    def get_action(self, board,  *args, **kwargs):
        temp = kwargs.get("temp", 0.7)
        n_playout = kwargs.get("n_playout", -1)
        is_first_step = kwargs.get("is_first_step", False)
        is_return_prob = kwargs.get("is_return_prob", True)

        if not is_return_prob:
            move_x, move_y = self.robot.MaxValue_po(self.player, board.board, is_first_step)
            move = board.location_to_move((move_x, move_y))
            if is_first_step:
                move = 60
            return move

        move_probs = np.zeros(board.width * board.height)
        action, prob = self.mcts.get_move_probs(board, temp=temp, n_playout=n_playout)
        move_probs[list(action)] = prob

        move_x, move_y = self.robot.MaxValue_po(self.player, board.board, is_first_step)
        if move_x == -1 and move_y == -1:
            move = np.random.choice(action, p=prob)
        else:
            move = board.location_to_move((move_x, move_y))
        if is_first_step:
            move = 60

        tabulator_probs(move_probs, board, move, np.argmax(move_probs))
        print("robot")

        self.mcts.update_with_move(move)

        return move, move_probs


class TrainPipeline:
    def __init__(self, init_model=None):
        # params of the board and the game
        self.board_width = 11
        self.board_height = 11
        self.n_in_row = 5

        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # 基于KL自适应调整学习速率
        self.temp = 0.6  # 用于输出每个位置概率时保留多少参数，1为全部保留，小于0.01就只输出概率最大的内个位置
        self.n_playout = int(11 * 11 * 1.5)  # 每次移动mcts搜索多少次
        self.c_puct = 5
        self.play = 1  # 1为ai自我对战训练，2为ai于人对战训练
        self.buffer_size = 10000
        self.batch_size_mini = 521  # 训练集的最小值
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 4  # 每次更新的训练次数
        self.n_to_save = 1  # 多少局保存一次模型
        self.n_train_with_robot = -1  # 自我训练多少把后于机器人对决训练
        self.n_with_robot = 1  # 一次于机器人对决运行多少把
        self.kl_targ = 0.02
        self.check_freq = 500
        self.game_batch_num = 10000000
        self.best_win_ratio = 0.0
        # 用于纯MCT的模拟数，用作对手需要评估经过训练的策略
        self.pure_mcts_playout_num = 1763
        self.init_model = 'model/current_policy.model'
        self.model_path = self.init_model
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row,
                           n_playout=self.n_playout)
        self.game = Game(self.board)

        self.policy_value_net = PolicyValueNet(self.board_width,
                                               self.board_height,
                                               model_file=self.init_model)
        self.mcts = MCTS(self.policy_value_net.policy_value_fn, self.c_puct, self.n_playout)

        self.mcts_player = MCTSPlayer(mcts=self.mcts, is_selfplay=True)

        self.human_player = human_player_train(policyValueNet=self.policy_value_net.policy_value_fn, mcts=self.mcts)
        self.human_player2 = human_player_train(policyValueNet=self.policy_value_net.policy_value_fn, mcts=self.mcts)
        self.robot_player = robot_in_train(mcts=self.mcts)

        # 如果是在AI于人类的学习模式中，重写参数
        if self.play == 2:
            self.batch_size_mini = -1
            self.n_to_save = 1

    def get_equi_data(self, play_data):
        """
        通过旋转和翻转来扩充数据集
        输出数据：[（棋盘状态、mcts_prob、获胜者），…，…]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1, current_inning=-1):
        """为训练收集自我游戏数据"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp,
                                                          current_inning=current_inning)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def collect_train_with_person_data(self):

        winner, play_data = self.game.start_train_with_person(self.human_player, self.mcts_player, temp=self.temp,
                                                              is_train_with_human_human=False, start_player=1, random_choise_start=False)

        play_data = list(play_data)[:]
        self.episode_len = len(play_data)
        play_data = self.get_equi_data(play_data)
        self.data_buffer.extend(play_data)

    def collect_train_with_robot_data(self, current_inning):

        winner, play_data = self.game.start_train_with_person(self.robot_player, self.mcts_player, temp=self.temp,
                                                              is_train_with_human_human=False, is_train_with_robot=True,
                                                              start_player=0, current_inning=current_inning, random_choise_start=True)
        play_data = list(play_data)[:]
        self.episode_len = len(play_data)
        play_data = self.get_equi_data(play_data)
        self.data_buffer.extend(play_data)

    def policy_update(self):

        if self.batch_size_mini == -1:
            batch_size = len(self.data_buffer) - 1
            if batch_size >= 50:
                batch_size = 50
        else:
            batch_size = self.batch_size_mini

        winner_batch = []
        old_v = []
        new_v = []

        for j in range(4):
            # 从存储的数据里面抽取一部分的数据
            mini_batch = random.sample(self.data_buffer, batch_size)
            state_batch = [data[0] for data in mini_batch]
            mcts_probs_batch = [data[1] for data in mini_batch]
            winner_batch = [data[2] for data in mini_batch]
            old_probs, old_v = self.policy_value_net.policy_value(state_batch)
            # 对这部分的数据进行训练
            for i in range(self.epochs):
                loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, self.learn_rate * self.lr_multiplier)

                new_probs, new_v = self.policy_value_net.policy_value(state_batch)
                kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
                if kl > self.kl_targ * 4:  # 如果D_KL严重偏离，则提前停车
                    break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 - np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 - np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=7):
        """
        通过与纯MCTS玩家比赛来评估经过培训的策略
        注：这仅用于监控培训进度
        """
        current_mcts_player = MCTSPlayer(mcts=self.mcts)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=1,
                                          is_in_train=True)
            win_cnt[winner] += 1
            print("n_inning:{}, win: {}".format(i, winner))
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num,
            win_cnt[1], win_cnt[2], win_cnt[-1]))

        return win_ratio

    def run(self):
        """run the training pipeline"""
        # 于机器人训练进行了几局
        current_inning_for_robot = self.n_with_robot + 1
        current_inning = 0
        for i in range(self.game_batch_num):

            if current_inning_for_robot >= self.n_with_robot:
                if self.play == 1:
                    self.collect_selfplay_data(self.play_batch_size, current_inning=current_inning)
                else:
                    self.collect_train_with_person_data()
                current_inning += 1

            # 自我对局后于机器人对局
            if self.n_train_with_robot > 0:
                if current_inning % self.n_train_with_robot == 0:
                    current_inning_for_robot = 0
            if current_inning_for_robot < self.n_with_robot:
                self.collect_train_with_robot_data(current_inning)
                current_inning_for_robot += 1
                current_inning += 1

            print("batch i:{}, episode_len:{}".format(current_inning, self.episode_len))

            # 训练神经网络
            if current_inning % 10 == 0 and len(self.data_buffer) > self.batch_size_mini:
                self.policy_update()
                self.policy_value_net.save_model(self.model_path)
            # # 保存模型
            # if current_inning % self.n_to_save == 0 and len(self.data_buffer) > self.batch_size_mini:
            #     self.policy_value_net.save_model(self.model_path)


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
