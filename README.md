# Gomoku Zero (AlphaZero Implementation)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Active_Refactoring-orange)
![Algorithm](https://img.shields.io/badge/Algorithm-MCTS_Rx_ResNet-green)

An advanced Gomoku AI engine implementing the AlphaZero methodology. This project explores the integration of Monte Carlo Tree Search (MCTS) with Deep Reinforcement Learning to achieve superhuman performance without human knowledge.

> **Note**: This repository is currently undergoing a major refactor to migrate the legacy backend from TensorFlow 1.x to PyTorch, incorporating ResNet architectures for improved policy-value estimation.

## 🚀 Key Features

*   **AlphaZero Architecture**: Implements the core self-play reinforcement learning loop.
*   **MCTS + Neural Network**: Combines tree search with a policy-value network to guide decision-making.
*   **Performance Optimization**:
    *   **Multi-process self-play**: Utilizes Python `multiprocessing` to run parallel self-play games, saturating CPU cores for faster data generation.
    *   **Data augmentation**: Rotates and flips board states to maximize training efficiency.
*   **Human-AI Interface**: Includes a GUI for testing model performance against human players.

## 🛠️ Tech Stack & Evolution

This project represents a long-term exploration of Reinforcement Learning:

*   **v1.0 (Legacy)**: Initial implementation using TensorFlow 1.x and basic CNNs.
*   **v2.0 (Current Focus)**:
    *   **PyTorch Backend**: Migrating core network definitions to PyTorch for better flexibility and modern tooling.
    *   **ResNet Architecture**: Upgrading from simple CNNs to **Residual Networks (ResNet)** to solve the vanishing gradient problem in deeper networks and improve feature extraction.
    *   **Parallel Training**: optimizing the self-play pipeline to run concurrently.

## 📂 Project Structure

*   `mcts_alphaZero.py`: The core MCTS algorithm implementation.
*   `policy_value_net_pytorch.py`: Neural network definition (ResNet backbone + Policy/Value heads).
*   `train.py`: The multi-process self-play training pipeline.
*   `human_play.py`: Interactive GUI for testing.

## 🚀 Quick Start

### 1. Prerequisites

Ensure you have Python 3.8+ installed.

```bash
pip install torch numpy tqdm
```

### 2. Play Against AI

Run the GUI to play against the pre-trained model:

```bash
python human_play.py
```

### 3. Train from Scratch

To start the self-play training loop (warning: computationally intensive):

```bash
python train.py
```

## 📈 Performance

*   Achieved >80% win rate against amateur human players after 2000+ self-play iterations.
*   Demonstrates distinct strategic behaviors (e.g., forming "double threes") purely learned from rules.

## 📄 License

MIT License
