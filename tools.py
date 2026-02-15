import numpy as np

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def tabulator_probs(move_probs, board, move, max_prob, length=7):
    column = board.width

    def aligns(string, index, length=7):
        difference = length - len(string)  # 计算限定长度为20时需要补齐多少个空格
        if difference == 0:  # 若差值为0则不需要补
            return string
        elif difference < 0:
            print('错误：限定的对齐长度小于字符串长度!')
            return None
        new_string = string
        space = ' '
        return_string = None
        if index == move:
            return_string = "\033[33m{}\033[0m".format(new_string) + space * difference  # 返回补齐空格后的字符串
        if index == max_prob:
            return_string = "\033[32m{}\033[0m".format(new_string) + space * difference
        if index == move and index == max_prob:
            return_string = "\033[31m{}\033[0m".format(new_string) + space * difference
        if board.states.get(index) == 1:
            return_string = "\033[34m{}\033[0m".format("X  ") + space * difference
        if board.states.get(index) == 2:
            return_string = "\033[35m{}\033[0m".format("O  ") + space * difference
        if not index == move and not index == max_prob and index in board.availables:
            return_string = new_string + space * difference
        return return_string
    # 将每个位置的概率显示出来
    move_probs_ = move_probs
    # 将里面的每一项都变成保留一定小数的然后转换成字符串
    for i in range(len(move_probs)):
        move_probs_[i] = round(move_probs[i], 4)  # 保留n位小数并转换成字符串
    move_str = []
    for i in range(len(move_probs_)):
        move_str.append(str(move_probs_[i]))

    print("\n")
    p = ''
    num = 0
    sum = len(move_str)
    index = 0
    for i in move_str:
        p = p + aligns(i, index, length)
        num = num + 1
        sum = sum - 1
        if num >= column:
            print(p)
            p = ''
            num = 0
        elif sum <= 0:
            print(p)
        index += 1


def graphic(board, player1, player2):
    """绘制棋盘并显示游戏信息"""
    width = board.width
    height = board.height

    print("Player", player1, "with X".rjust(3))
    print("Player", player2, "with O".rjust(3))
    print()
    for x in range(width):
        print("{0:8}".format(x), end='')
    print('\r\n')
    for i in range(height):
        print("{0:4d}".format(i), end='')
        for j in range(width):
            loc = i * width + j
            p = board.states.get(loc, -1)
            if p == player1:
                print('X'.center(8), end='')
            elif p == player2:
                print('O'.center(8), end='')
            else:
                print('_'.center(8), end='')
        print('\r\n\r\n')
