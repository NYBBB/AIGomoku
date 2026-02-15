
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet
from mcts_alphaZero import MCTS
from tools import *
from Data.hparam import Data
from train import robot_in_train
import tkinter as tk
from tkinter import *
from tkinter.ttk import *
import math



class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None
        self.set_player_name()

    def set_player_ind(self, p):
        self.player = p

    def set_player_name(self, name="Human"):
        self.name = name

    def get_action(self, board, *args, **kwargs):
        locationForHumanClicked = kwargs.get("locationForHumanClicked")
        move = board.location_to_move(locationForHumanClicked)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


width = Data.width
height = Data.height
temp = Data.temp

board = Board(width=width, height=height, n_in_row=5, n_playout=Data.n_playout)
game = Game(board)
# ############### human VS AI ###################
# 将训练好的policy_value_net加载进 Theano/Lasagne,
model_file = Data.SavePath
net_policy = PolicyValueNet(width, height, model_file=model_file)
mcts = MCTS(net_policy.policy_value_fn, 5, 1486)
mcts_player = MCTSPlayer(mcts=mcts)  # set larger n_playout for better performance
human_player = Human()
robot_player = robot_in_train()

def update_net():
    net_policy.restore_model(model_file)
    mcts.update_policy_function(net_policy.policy_value_fn)
    mcts_player.update_mcts(mcts)

who_start = 0  # 0为玩家先走，1为AI先走

PIECE_SIZE = 15

click_x = 0
click_y = 0
piece_color = "black"

pieces_x = [i for i in range(32, 32 + board.width * 35, 35)]
pieces_y = [i for i in range(38, 38 + board.height * 35, 35)]


class person_fight:
    def __init__(self):
        self.root = tk.Tk()

        self.root.title("Gobang")
        self.root.geometry("760x560")

        """参数"""
        self.click_y = 0
        self.click_x = 0
        self.current_step = 0
        self.current_player = None
        self.player_in_turn = None
        self.is_first_step = True

        self.player1 = human_player
        self.player2 = mcts_player

        self.net_board = self.get_net_board()  # 获得到棋盘中每个点的位置
        self.Is_start = False  # 是否开始游戏了
        self.game_mode = IntVar(value=0, name="game_mode")
        self.game_mode.set(0)  # 游戏模式：0为人类对战AI， 1为人类对战robot， 2为人类对战人类
        self.who_start = IntVar(value=0, name="who_start")  # 0为玩家先走，1为AI先走


        """棋子提示"""
        self.side_canvas = tk.Canvas(self.root, width=220, height=50)
        self.side_canvas.grid(row=0, column=1)
        self.side_canvas.create_oval(110 - PIECE_SIZE, 25 - PIECE_SIZE, 110 + PIECE_SIZE, 25 + PIECE_SIZE, fill=piece_color, tags="show_piece")

        """棋子提示标签"""
        self.var = tk.StringVar()
        self.var.set("执黑棋")
        self.person_label = tk.Label(self.root, textvariable=self.var, width=12, anchor=tk.CENTER, font=("Arial", 20))
        self.person_label.grid(row=1, column=1)

        """输赢提示标签"""
        self.var1 = tk.StringVar()
        self.var1.set("")
        self.result_label = tk.Label(self.root, textvariable=self.var1, width=12, height=4, anchor=tk.CENTER, fg="red", font=("Arial", 25))
        self.result_label.grid(row=2, column=1, rowspan=2)

        """游戏结束提示标签"""
        self.var2 = tk.StringVar()
        self.var2.set("")
        self.game_label = tk.Label(self.root, textvariable=self.var2, width=12, height=4, anchor=tk.CENTER, font=("Arial", 18))
        self.game_label.grid(row=4, column=1)

        """游戏模式选择"""
        self.game_mode_button = Radiobutton(self.root, text="Battle with AI", variable=self.game_mode, value=0, command=self.setMode)
        self.game_mode_button.place(relx=0, rely=0, x=580, y=370)

        self.game_mode_button2 = Radiobutton(self.root, text="Battle with algorithm", variable=self.game_mode, value=1, command=self.setMode)
        self.game_mode_button2.place(relx=0, rely=0, x=580, y=400)

        self.game_mode_button3 = Radiobutton(self.root, text="Battle with human", variable=self.game_mode, value=2, command=self.setMode)
        self.game_mode_button3.place(relx=0, rely=0, x=580, y=430)

        """游戏开始玩家选择"""
        self.game_start_button1 = Radiobutton(self.root, text="human start", variable=self.who_start, value=0, command=self.set_start_player)
        self.game_start_button1.place(relx=0, rely=0, x=580, y=280)

        self.game_start_button2 = Radiobutton(self.root, text="AI start", variable=self.who_start, value=1, command=self.set_start_player)
        self.game_start_button2.place(relx=0, rely=0, x=580, y=310)

        """重置按钮"""
        self.reset_button = tk.Button(self.root, text="重置并开始", font=25, width=15, command=self.gameReset)
        self.reset_button.grid(row=5, column=1)

        """棋盘绘制"""
        # 背景
        self.canvas = tk.Canvas(self.root, bg="saddlebrown", width=540, height=540)
        self.canvas.bind("<Button-1>", self.coorBack)  # 鼠标单击事件绑定
        self.canvas.grid(row=0, column=0, rowspan=6)
        # 线条
        for i in range(height):
            self.canvas.create_line(32, (int(35 * 15 // height) * i + 38), 35 * (height - 4) * 2 + height + 2, (int(35 * 15 // height) * i + 38))
            self.canvas.create_line((int(35 * 15 // height) * i + 32), 38, (int(35 * 15 // height) * i + 32), 35 * (height - 4) * 2 + width + 7)
        # 点
        point_x = [2, 5, 2, 8, 8]
        point_y = [2, 5, 8, 8, 2]
        for i in range(5):
            self.canvas.create_oval(int(35 * 15 // height) * point_x[i] + 28, int(35 * 15 // height) * point_y[i] + 33,
                                    int(35 * 15 // height) * point_x[i] + 37, int(35 * 15 // height) * point_y[i] + 42, fill="black")

        # 透明棋子（设置透明棋子，方便后面落子的坐标定位到正确的位置）
        for i in pieces_x:
            for j in pieces_y:
                self.canvas.create_oval(i - PIECE_SIZE, j - PIECE_SIZE, i + PIECE_SIZE, j + PIECE_SIZE, width=0, tags=(str(i), str(j)))

        # 数字坐标
        for i in range(height):
            self.label = tk.Label(self.canvas, text=str(i), fg="black", bg="saddlebrown", width=2, anchor=tk.E)
            self.label.place(x=2, y=int(35 * 15 // height) * i + 28)
        # 字母坐标
        count = 0
        for i in range(65, 65 + height):
            self.label = tk.Label(self.canvas, text=chr(i), fg="black", bg="saddlebrown")
            self.label.place(x=int(35 * 15 // height) * count + 25, y=2)
            count += 1

        """窗口循环"""
        self.root.mainloop()

    def gameReset(self):
        update_net()
        board.init_board()
        self.canvas.delete("piece")
        self.Is_start = True
        self.is_first_step = True
        self.current_step = 0
        self.update_var()

        p1, p2 = board.players
        self.player1.set_player_ind(p1)
        self.player2.set_player_ind(p2)
        self.players = {p1: self.player1, p2: self.player2}

        print("\n已重置")
        print("Player 1 is : ", self.player1.name)
        print("Player 2 is : ", self.player2.name)
        print("search accuracy parameter : ", temp)
        print("")

        if self.who_start.get() == 1 and self.game_mode.get() != 2:
            self.ai_go_one_step()

    def setMode(self):
        if self.who_start.get() == 0:
            if self.game_mode.get() == 0:
                print("Battle with AI\n")
                self.player1 = human_player
                self.player2 = mcts_player
            elif self.game_mode.get() == 1:
                print("Battle with robot\n")
                self.player1 = human_player
                self.player2 = robot_player
            else:
                print("Battle with human\n")
                self.player1 = human_player
                self.player2 = human_player
        elif self.who_start.get() == 1:
            if self.game_mode.get() == 0:
                print("Battle with AI\n")
                self.player1 = mcts_player
                self.player2 = human_player
            elif self.game_mode.get() == 1:
                print("Battle with robot\n")
                self.player1 = robot_player
                self.player2 = human_player
            else:
                print("Battle with human\n")
                self.player1 = human_player
                self.player2 = human_player
        pass

    def set_start_player(self):
        if self.who_start.get() == 0:
            print("start player: 0\n")
            if self.game_mode.get() == 0:
                self.player1 = human_player
                self.player2 = mcts_player
            elif self.game_mode.get() == 1:
                self.player1 = human_player
                self.player2 = robot_player
            else:
                self.player1 = human_player
                self.player2 = human_player
        elif self.who_start.get() == 1:
            print("start player: 1\n")
            if self.game_mode.get() == 0:
                self.player1 = mcts_player
                self.player2 = human_player
            elif self.game_mode.get() == 1:
                self.player1 = robot_player
                self.player2 = human_player
            else:
                self.player1 = human_player
                self.player2 = human_player
        pass

    def update_var(self, current_player=2, winner=-1):
        if current_player == 2:
            piece_color = "black"
            self.side_canvas.delete("show_piece")
            self.side_canvas.create_oval(110 - PIECE_SIZE, 25 - PIECE_SIZE,
                                    110 + PIECE_SIZE, 25 + PIECE_SIZE,
                                    fill=piece_color, tags="show_piece")
            self.var.set("执黑棋")

        elif current_player == 1:
            piece_color = "white"
            self.side_canvas.delete("show_piece")
            self.side_canvas.create_oval(110 - PIECE_SIZE, 25 - PIECE_SIZE,
                                         110 + PIECE_SIZE, 25 + PIECE_SIZE,
                                         fill=piece_color, tags="show_piece")
            self.var.set("执白棋")
        elif current_player == 3:
            if winner == 1:
                piece_color = "black"
                self.side_canvas.delete("show_piece")
                self.side_canvas.create_oval(110 - PIECE_SIZE, 25 - PIECE_SIZE,
                                             110 + PIECE_SIZE, 25 + PIECE_SIZE,
                                             fill=piece_color, tags="show_piece")
                self.var.set("胜者是: 黑棋")
            else:
                piece_color = "white"
                self.side_canvas.delete("show_piece")
                self.side_canvas.create_oval(110 - PIECE_SIZE, 25 - PIECE_SIZE,
                                             110 + PIECE_SIZE, 25 + PIECE_SIZE,
                                             fill=piece_color, tags="show_piece")
                self.var.set("胜者是: 白棋")

    def gameEnd(self, winner):
        print("胜者是：", winner)
        self.Is_start = False

    def ai_go_one_step(self):

        self.current_player = board.get_current_player()
        self.player_in_turn = self.players[self.current_player]
        current_player_name = self.player_in_turn.name
        human_move = None
        if self.current_step <= 4:
            current_temp = 1
        else:
            current_temp = temp
        move = self.player_in_turn.get_action(board, n_playout=board.get_n_playout(),
                                              locationForHumanClicked=human_move, temp=current_temp,
                                              is_first_step=self.is_first_step, is_return_prob=False)
        if self.is_first_step:  self.is_first_step = False
        board.do_move(move)
        board.update_n_playout(current_step=self.current_step)

        self.current_step += 1
        print(current_player_name, ": ", move)
        self.showChess(self.current_player, move)
        self.update_var(self.current_player)

    def coorBack(self, event):
        """点击棋盘后的回调函数"""

        if not self.Is_start:
            return

        self.click_y = event.y
        self.click_x = event.x
        item = self.get_nearest_po(self.click_x, self.click_y)  # 获得到离鼠标点击位置最近的点位//（263,267）
        human_move = self.board_to_step_location(item[0], item[1])  # 获得到点击的点对应的位置//（5,3）

        for i in range(2):
            self.current_player = board.get_current_player()
            self.player_in_turn = self.players[self.current_player]
            current_player_name = self.player_in_turn.name
            if self.current_step <= 4:
                current_temp = 1
            else:
                current_temp = temp
            move = self.player_in_turn.get_action(board, n_playout=board.get_n_playout(),
                                                  locationForHumanClicked=human_move, temp=current_temp,
                                                  is_first_step=self.is_first_step, is_return_prob=False)
            if self.is_first_step:  self.is_first_step = False
            if move not in board.availables:
                print("此处不能下")
                break

            board.do_move(move)
            board.update_n_playout(current_step=self.current_step, is_in_game=True)

            self.current_step += 1

            print(current_player_name, ": ", move)
            self.showChess(self.current_player, move)
            self.update_var(self.current_player)

            end, winner = board.game_end()
            if end:
                self.update_var(3, winner)
                self.gameEnd(winner)
                break

            if self.game_mode.get() == 2:
                break

        pass

    def showChess(self, current_player, move):
        # 显示棋子
        if current_player == 1:
            piece_color_ = "black"
        else:
            piece_color_ = "white"

        move_x = 0
        move_y = 0
        for i in range(height):
            if (i + 1) * height > move:
                move_y = move - i * height
                move_x = i

                break
        # print("玩家", current_player, "移动的位置: ", move_x, move_y)
        move_x, move_y = self.pos_in_board(move_x, move_y)
        self.canvas.create_oval(move_x - PIECE_SIZE, move_y - PIECE_SIZE,
                                move_x + PIECE_SIZE, move_y + PIECE_SIZE,
                                fill=piece_color_, tags="piece")

    def get_net_board(self):
        """得到棋盘的每个点的位置"""
        net_list = []
        for row in range(width):
            for col in range(height):
                point = self.pos_in_board(row, col)
                net_list.append(point)
        return net_list

    def get_nearest_po(self, x, y):
        """得到坐标（x, y）在棋盘各点中最近的一个点"""
        flag = 600
        position = ()
        for point in self.net_board:
            distance = self.get_distance([x, y], point)
            if distance < flag:
                flag = distance
                position = point
        return position

    def board_to_step_location(self, x, y):
        """将ui上的位置坐标转换为棋盘的的位置"""
        return (y - 35) // int(35 * 15 // height), (x - 30) // int(35 * 15 // height)

    def pos_in_board(self, x, y):
        """棋局中的点计算在棋盘中的位置"""
        return int(35 * 15 // height) * y + 32, int(35 * 15 // height) * x + 37

    def get_distance(self, p0, p1):
        """计算两个点之间的距离"""
        return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)



person_fight = person_fight()
