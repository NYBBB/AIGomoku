import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Net(nn.Module):
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        
        # 共同层：卷积层
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 策略头 (Action Head)
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height, 
                                 board_width * board_height)
        
        # 价值头 (Value Head)
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # 共同层
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # 策略头
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)
        
        # 价值头
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        
        return x_act, x_val

class PolicyValueNet():
    """
    基于PyTorch的策略价值网络实现
    """
    def __init__(self, board_width, board_height, model_file=None, use_gpu=True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        self.device = torch.device("cuda" if self.use_gpu else "cpu")

        self.policy_value_net = Net(board_width, board_height).to(self.device)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            print(f"Loading model from {model_file}...")
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """
        输入: 一批状态
        输出: 一批动作概率和状态价值
        """
        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        self.policy_value_net.eval()
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.cpu().numpy())
            return act_probs, value.cpu().numpy()

    def policy_value_fn(self, board):
        """
        输入: 棋盘
        输出: (action, probability) 列表，以及棋盘状态的分数
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        
        state_batch = torch.FloatTensor(current_state).to(self.device)
        self.policy_value_net.eval()
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.cpu().numpy().flatten())
            
        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value.cpu().numpy()[0][0]

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """执行一步训练"""
        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        mcts_probs = torch.FloatTensor(np.array(mcts_probs)).to(self.device)
        winner_batch = torch.FloatTensor(np.array(winner_batch)).to(self.device)

        self.policy_value_net.train()
        self.optimizer.zero_grad()
        
        # 设置学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播
        log_act_probs, value = self.policy_value_net(state_batch)
        
        # 定义损失函数
        # Value Loss: MSE
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        # Policy Loss: Cross Entropy
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        
        loss = value_loss + policy_loss
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        # 计算熵 (用于监控)
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        
        return loss.item(), entropy.item()

    def save_model(self, model_file):
        """保存模型参数"""
        torch.save(self.policy_value_net.state_dict(), model_file)
