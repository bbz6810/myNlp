import numpy as np


class Solution:
    def exist(self, board, word):
        if not word or not board:
            return True
        m, n = len(board), len(board[0])

        def next(pos):
            res = []
            if pos[1] - 1 >= 0 and path[pos[0]][pos[1] - 1] == 0:
                res.append((pos[0], pos[1] - 1))
            if pos[0] - 1 >= 0 and path[pos[0] - 1][pos[1]] == 0:
                res.append((pos[0] - 1, pos[1]))
            if pos[1] + 1 < n and path[pos[0]][pos[1] + 1] == 0:
                res.append((pos[0], pos[1] + 1))
            if pos[0] + 1 < m and path[pos[0] + 1][pos[1]] == 0:
                res.append((pos[0] + 1, pos[1]))
            return res

        start_dot = [(x, y) for x in range(len(board)) for y in range(len(board[x])) if board[x][y] == word[0]]

        def begin(dot, char_index, dots, stack):
            """当前 dot 元素和charindex索引的字符一样则继续，否则return
                如果字符一样则递归进入
                否则返回上一层

            :param dot:
            :param char_index:
            :param dots:
            :return:
            """
            if board[dot[0]][dot[1]] == word[char_index]:
                path[dot[0]][dot[1]] = 1
                stack.append(dot)
                nexts = next(dot)
                path2[dot[0]][dot[1]] = len(nexts)
                # print(22222, dots, nexts)
                for d in nexts:
                    if char_index < len(word) - 1:
                        begin(d, char_index + 1, dots + [d], stack)
            else:
                print(111111, dots)
                end = stack[-1]
                path[end[0]][end[1]] = 0
                stack = stack[:-1]
                return

                # path[dot[0]][dot[1]] = 1
                # nexts = next(dot)
                # # print(nexts, dots)
                # path2[dot[0]][dot[1]] = len(nexts)
                # # print(path)
                # for d in nexts:
                #     if char_index < len(word) - 1:
                #         if board[d[0]][d[1]] == word[char_index + 1]:
                #             begin(d, char_index + 1, dots + [d])
                #         else:
                #             # print('回退', dots)
                #             print(222222, path)
                #             for _d in dots:
                #                 if path2[_d[0]][_d[1]] == 1:
                #                     path[_d[0]][_d[1]] = 0
                #             print(333333, path)
                #     else:
                #         return

        for dot in start_dot[1:]:
            path = np.zeros(shape=(m, n), dtype='int8')
            path2 = np.zeros(shape=(m, n), dtype='int8')
            stack = []
            begin(dot, 0, [dot], stack)
            res = np.sum(path)
            print(path)
            print(path2)
            if res == len(word):
                return True
        return False


if __name__ == '__main__':
    board = [["A", "B", "C", "E"],
             ["S", "F", "C", "S"],
             ["A", "D", "E", "E"]]
    word = "SEE"

    # board = [["a", "b", "c"],
    #          ["a", "e", "d"],
    #          ["a", "f", "g"]]
    # word = "abcdefg"

    # from leetcode.leetcode_79data import board, word

    a = Solution().exist(board, word)
    print(a)
