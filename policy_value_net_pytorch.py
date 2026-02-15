import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ResidualBlock(nn.Module):
    """
    Residual Block as used in AlphaZero / ResNet
    Structure: Conv -> BN -> ReLU -> Conv -> BN -> (+Input) -> ReLU
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class Net(nn.Module):
    """
    Policy-Value Network based on ResNet
    """
    def __init__(self, board_width, board_height, num_res_blocks=3):
        super(Net, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        
        # Initial Convolution Block
        self.conv_input = nn.Conv2d(4, 64, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(64)
        
        # Residual Tower
        self.res_tower = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_res_blocks)]
        )
        
        # Policy Head (Action Head)
        self.policy_conv = nn.Conv2d(64, 4, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(4)
        self.policy_fc = nn.Linear(4 * board_width * board_height, 
                                 board_width * board_height)
        
        # Value Head
        self.value_conv = nn.Conv2d(64, 2, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(2)
        self.value_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # Input Block
        x = F.relu(self.bn_input(self.conv_input(state_input)))
        
        # Residual Tower
        x = self.res_tower(x)
        
        # Policy Head
        x_act = F.relu(self.policy_bn(self.policy_conv(x)))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = F.log_softmax(self.policy_fc(x_act), dim=1)
        
        # Value Head
        x_val = F.relu(self.value_bn(self.value_conv(x)))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.value_fc1(x_val))
        x_val = torch.tanh(self.value_fc2(x_val))
        
        return x_act, x_val

class PolicyValueNet():
    """
    Wrapper for the PyTorch Policy-Value Network
    """
    def __init__(self, board_width, board_height, model_file=None, use_gpu=True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # L2 regularization coefficient
        self.device = torch.device("cuda" if self.use_gpu else "cpu")

        # Initialize the network with ResNet architecture
        self.policy_value_net = Net(board_width, board_height).to(self.device)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            try:
                print(f"Loading model from {model_file}...")
                net_params = torch.load(model_file, map_location=self.device)
                self.policy_value_net.load_state_dict(net_params)
            except Exception as e:
                print(f"Failed to load model: {e}. Starting from scratch.")

    def policy_value(self, state_batch):
        """
        Input: a batch of states
        Output: a batch of action probabilities and state values
        """
        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        self.policy_value_net.eval()
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.cpu().numpy())
            return act_probs, value.cpu().numpy()

    def policy_value_fn(self, board):
        """
        Input: board
        Output: a list of (action, probability) tuples for each available
        action and the score of the board state
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
        """Perform a training step"""
        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        mcts_probs = torch.FloatTensor(np.array(mcts_probs)).to(self.device)
        winner_batch = torch.FloatTensor(np.array(winner_batch)).to(self.device)

        self.policy_value_net.train()
        self.optimizer.zero_grad()
        
        # Set learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # Forward pass
        log_act_probs, value = self.policy_value_net(state_batch)
        
        # Loss calculation
        # Value Loss: MSE
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        # Policy Loss: Cross Entropy
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        
        loss = value_loss + policy_loss
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Calculate entropy (for monitoring)
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        
        return loss.item(), entropy.item()

    def save_model(self, model_file):
        """Save model parameters"""
        torch.save(self.policy_value_net.state_dict(), model_file)

    def restore_model(self, model_file):
        """Load model parameters"""
        if model_file and os.path.exists(model_file):
            try:
                self.policy_value_net.load_state_dict(torch.load(model_file, map_location=self.device))
                print(f"Successfully restored model from {model_file}")
            except Exception as e:
                print(f"Failed to restore model: {e}")
