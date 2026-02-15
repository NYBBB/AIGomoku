
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
from torch.utils.tensorboard import SummaryWriter

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
        policy_value_net = PolicyValueNet(width, height, model_file=model_file, use_gpu=False)
        
        mcts = MCTS(policy_value_net.policy_value_fn, c_puct=5, n_playout=n_playout)
        player = MCTSPlayer(mcts=mcts, is_selfplay=True, verbose=False)
        
        # Run the game
        winner, play_data = game.start_self_play(player, temp=temp, is_shown=False)
        return list(play_data)
    except Exception as e:
        print(f"Error in worker process: {e}")
        return []

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
        self.buffer_size = 10000
        self.batch_size_mini = 512
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # 每次更新的训练次数
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 3000
        self.best_win_ratio = 0.0
        
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
        self.num_workers = max(1, multiprocessing.cpu_count() - 5)  # Leave one core for the trainer
        print(f"Parallel training enabled: Using {self.num_workers} worker processes.")

        # TensorBoard Writer
        self.writer = SummaryWriter('runs/gomoku_experiment')

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
        worker_args = [(self.model_path, self.temp, self.n_playout, self.game_params) for _ in range(n_games)]
        
        # Run workers in parallel
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            results = pool.starmap(run_self_play, worker_args)
            
        # Process results
        total_steps = 0
        for play_data in results:
            if play_data:
                self.episode_len = len(play_data)
                total_steps += self.episode_len
                # Augment the data
                play_data = self.get_equi_data(play_data)
                self.data_buffer.extend(play_data)
        
        avg_steps = total_steps / n_games if n_games > 0 else 0
        return avg_steps

    def policy_update(self, step_idx):
        """Update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size_mini)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        
        loss = 0
        entropy = 0
        kl = 0

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
        
        # Log to TensorBoard
        self.writer.add_scalar('Training/Loss', loss, step_idx)
        self.writer.add_scalar('Training/Entropy', entropy, step_idx)
        self.writer.add_scalar('Training/KL_Divergence', kl, step_idx)
        self.writer.add_scalar('Training/Learning_Rate_Multiplier', self.lr_multiplier, step_idx)
        
        return loss, entropy

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                # Collect self-play data in parallel
                avg_steps = self.collect_selfplay_data(self.num_workers) 
                print("batch i:{}, episode_len:{:.2f}".format(i + 1, avg_steps))
                
                # Log game length
                self.writer.add_scalar('Game/Average_Length', avg_steps, i + 1)
                self.writer.add_scalar('Game/Buffer_Size', len(self.data_buffer), i + 1)

                if len(self.data_buffer) > self.batch_size_mini:
                    loss, entropy = self.policy_update(i + 1)
                    # Save the model so workers can pick up the new weights
                    self.policy_value_net.save_model(self.model_path)
                    
                if (i + 1) % self.check_freq == 0:
                    print(f"Checkpoint saved at iteration {i+1}")
                    # Save a backup occasionally
                    self.policy_value_net.save_model(os.path.join(self.model_dir, f'policy_{i+1}.model'))
                    
        except KeyboardInterrupt:
            print('\n\rquit')
        finally:
            self.writer.close()

if __name__ == '__main__':
    # On Windows, multiprocessing requires this protection
    multiprocessing.freeze_support()
    training_pipeline = TrainPipeline()
    training_pipeline.run()
