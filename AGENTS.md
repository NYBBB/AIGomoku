# AGENTS.md

This file provides context and guidelines for AI agents working on the AIGomoku codebase.

## 1. Build, Run, and Test Commands

This project is a Python-based Gomoku AI using TensorFlow (v1 compatibility mode) and MCTS. There is no standard build system (like `setup.py` or `poetry`), and dependencies are managed manually (ensure `tensorflow`, `numpy`, `PyYAML`, `tqdm`, `tkinter` are installed).

### Running the Game (GUI)
To play against the AI, run the `human_play.py` script. This launches a Tkinter-based GUI.
```bash
python human_play.py
```

### Training the Model
To train the AI model, use `train.py`. This script handles self-play data collection, model updates, and evaluation.
```bash
python train.py
```
*Note: `train.py` includes a `policy_evaluate` method that serves as a performance check.*

### Testing
There are **no explicit unit tests** (e.g., `pytest` or `unittest` files) in the repository.
*   **Verification:** When making changes, verify functionality by running `train.py` for a few epochs or playing a game using `human_play.py`.
*   **New Tests:** If you add new logic, creating a standalone test script (e.g., `tests/test_my_feature.py`) is highly recommended using `unittest` or `pytest`.

### Building the Executable
The project uses PyInstaller. To build a standalone executable:
```bash
pyinstaller human_play.spec
```

## 2. Code Style & Conventions

### Formatting & Structure
*   **Indentation:** Use **4 spaces**.
*   **Line Length:** generally standard (~80-100 chars), but not strictly enforced.
*   **Imports:** Group imports: Standard library -> Third-party (numpy, tensorflow) -> Local modules.
*   **Classes:** Use `CamelCase` (e.g., `PolicyValueNet`, `MCTSPlayer`).
*   **Functions/Variables:** Use `snake_case` (e.g., `get_action`, `train_step`).
*   **File Naming:** Use `snake_case` (e.g., `mcts_pure.py`).

### Type Safety & Comments
*   **Type Hints:** Type hints are **not** currently used. Python 3 style is preferred if adding them.
*   **Comments:** The codebase contains a mix of **English** and **Chinese** comments.
    *   When adding new comments, prefer **English** for broad compatibility, but respect existing Chinese comments if modifying them slightly.
    *   Docstrings are used for classes and complex methods.

### Libraries & Frameworks
*   **TensorFlow:** Uses `tensorflow.compat.v1` (`tf.disable_eager_execution()`). Do **not** upgrade to TF2 native style without a full refactor.
*   **NumPy:** Used extensively for board state manipulation and math.
*   **Tkinter:** Used for the GUI.
*   **Configuration:** Parameters are loaded from `Data/Data.yaml` via `Data/hparam.py`.

### Error Handling
*   Basic `try-except` blocks are used where necessary (e.g., network calls, file I/O).
*   Custom exceptions are rarely used; standard exceptions are preferred.

## 3. Project Structure
*   `game.py`: Core game logic (`Board`, `Game` classes).
*   `mcts_alphaZero.py` / `mcts_pure.py`: MCTS implementations.
*   `policy_value_net_tensorflow.py`: Neural network model (TF 1.x).
*   `train.py`: Training pipeline.
*   `human_play.py`: Main entry point for GUI game.
*   `Data/`: Contains configuration (`Data.yaml`, `hparam.py`) and model data.
*   `model/`: Directory for saving trained models.

## 4. Algorithm & Implementation Details

### MCTS (Monte Carlo Tree Search)
The project implements two versions of MCTS:
1.  **Pure MCTS (`mcts_pure.py`):** Uses random rollouts for evaluation. This serves as a baseline for performance comparison.
2.  **AlphaZero MCTS (`mcts_alphaZero.py`):** Uses the Neural Network (`PolicyValueNet`) to guide the search (evaluating leaf nodes) instead of random rollouts.
    *   **Selection:** Uses PUCT (Predictor + Upper Confidence Bound applied to Trees) algorithm.
    *   **Expansion:** Expands leaf nodes using the policy network's probabilities.
    *   **Evaluation:** Uses the value network to estimate the win probability.
    *   **Backup:** Updates Q-values and visit counts up the tree.

### Neural Network Architecture (`policy_value_net_tensorflow.py`)
The network is a Convolutional Neural Network (CNN) with a dual-head output:
*   **Input:** A 4-channel board state (Current player, Opponent, Last Move, Color to Play).
*   **Common Layers:** 3 Convolutional layers with ReLU activation.
*   **Policy Head:** Convolution -> Fully Connected -> Softmax (outputs move probabilities).
*   **Value Head:** Convolution -> Fully Connected -> Tanh (outputs win/loss estimation between -1 and 1).
*   **Loss Function:** Combined loss = Value Loss (MSE) + Policy Loss (Cross Entropy) + L2 Regularization.

## 5. Troubleshooting & Common Issues

*   **TensorFlow Compatibility:**
    *   **Issue:** `AttributeError: module 'tensorflow' has no attribute 'placeholder'`
    *   **Fix:** Ensure `import tensorflow.compat.v1 as tf` and `tf.disable_eager_execution()` are present. Do not use TF2 native ops without migrating the session management.
*   **Tkinter Freezes:**
    *   **Issue:** The GUI becomes unresponsive during AI thinking time.
    *   **Cause:** MCTS search runs on the main UI thread.
    *   **Workaround:** Be patient; for a robust fix, move the AI search to a separate thread (note: this requires careful state management with Tkinter).
*   **Model Loading Errors:**
    *   **Issue:** Checksum mismatch or shape mismatch.
    *   **Fix:** Ensure the board size (width/height) in `Data/Data.yaml` matches the model being loaded. The default is often 11x11 or 15x15.

## 6. Special Instructions for Agents
*   **Environment:** Assume a Windows environment based on file paths in `.spec` files (e.g., `D:\...`), but code should remain cross-platform compatible where possible.
*   **Model Loading:** When modifying model loading/saving, ensure compatibility with the `saver.restore` mechanism in `policy_value_net_tensorflow.py`.
*   **UI Changes:** If modifying `human_play.py`, be careful with Tkinter main loop and threading if adding long-running tasks.
