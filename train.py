
from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet  # PyTorch implementation
from mcts_alphaZero import MCTS
from tools import *
from robot import Robot
import multiprocessing
import os

# Worker function for multiprocessing
def run_self_play(model_file, temp, n_playout, game_params):
    """
    Worker function to run a single game of self-play.
    This runs in a separate process.
    """
    try:
        # Re-initialize the environment for this process
        width, height, n_in_row = game_params
        board = Board(width=width, height=height, n_in_row=n_in_row)
        game = Game(board)
        
        # Load the policy network
        # Note: We create a new instance for each process to ensure thread safety
        # and to load the latest weights from disk.
        policy_value_net = PolicyValueNet(width, height, model_file=model_file)
        
        mcts = MCTS(policy_value_net.policy_value_fn, c_puct=5, n_playout=n_playout)
        player = MCTSPlayer(mcts=mcts, is_selfplay=True, verbose=False)
        
        # Run the game
        winner, play_data = game.start_self_play(player, temp=temp, is_shown=False)
        return list(play_data)
    except Exception as e:
        print(f"Error in worker process: {e}")
        return []

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
            if isinstance(location, str):
                try:
                    location = [int(n, 10) for n in location.split(",")]
                except:
                    continue

            if location[0] > board.width or location[1] > board.height:
                 continue

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
        self.game_params = (self.board_width, self.board_height, self.n_in_row)

        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # 基于KL自适应调整学习速率
        self.temp = 1.0  # 温度参数
        self.n_playout = 400  # MCTS模拟次数
        self.c_puct = 5
        self.play = 1  # 1为ai自我对战训练
        self.buffer_size = 10000
        self.batch_size_mini = 512
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # 每次更新的训练次数
        self.kl_targ = 0.02
        self.check_freq = 500
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 1000
        
        self.model_dir = 'model'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        self.init_model = os.path.join(self.model_dir, 'current_policy.model')
        self.model_path = self.init_model
        
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row,
                           n_playout=self.n_playout)
        self.game = Game(self.board)

        # Initialize the policy network
        self.policy_value_net = PolicyValueNet(self.board_width,
                                               self.board_height,
                                               model_file=self.init_model if os.path.exists(self.init_model) else None)
        
        self.mcts = MCTS(self.policy_value_net.policy_value_fn, self.c_puct, self.n_playout)
        self.mcts_player = MCTSPlayer(mcts=self.mcts, is_selfplay=True)

        # Save initial model to ensure workers can load it
        self.policy_value_net.save_model(self.model_path)
        
        # Parallel Processing Config
        self.num_workers = max(1, multiprocessing.cpu_count() - 1) # Leave one core for the trainer
        print(f"Parallel training enabled: Using {self.num_workers} worker processes.")

    def get_equi_data(self, play_data):
        """
        通过旋转和翻转来扩充数据集
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

    def collect_selfplay_data(self, n_games=1):
        """
        Collect self-play data using multiprocessing
        """
        # Prepare arguments for workers
        # Each worker needs: model_path, temp, n_playout, game_params
        worker_args = [(self.model_path, self.temp, self.n_playout, self.game_params) for _ in range(n_games)]
        
        # Run workers in parallel
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            results = pool.starmap(run_self_play, worker_args)
            
        # Process results
        for play_data in results:
            if play_data:
                self.episode_len = len(play_data)
                # Augment the data
                play_data = self.get_equi_data(play_data)
                self.data_buffer.extend(play_data)

    def policy_update(self):
        """Update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size_mini)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch, 
                    mcts_probs_batch, 
                    winner_batch, 
                    self.learn_rate * self.lr_multiplier)
            
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
                
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy))
        return loss, entropy

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                # Collect self-play data in parallel
                self.collect_selfplay_data(self.num_workers) # Run as many games as workers per batch
                print("batch i:{}, episode_len:{}".format(i + 1, self.episode_len))

                if len(self.data_buffer) > self.batch_size_mini:
                    loss, entropy = self.policy_update()
                    # Save the model so workers can pick up the new weights
                    self.policy_value_net.save_model(self.model_path)
                    
                if (i + 1) % 50 == 0:
                    print("Check point saved.")
                    
        except KeyboardInterrupt:
            print('\n\rquit')

if __name__ == '__main__':
    # On Windows, multiprocessing requires this protection
    multiprocessing.freeze_support()
    training_pipeline = TrainPipeline()
    training_pipeline.run()
