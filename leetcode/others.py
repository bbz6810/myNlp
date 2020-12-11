def a2n():
    """ a - 1
        b - 2
        z - 26

        input 226
        output 22 6, 2 26 两种

    """

    al = dict((chr(ord('A') + k), k + 1) for k in range(26))
    # print(al)
    s = '2222'
    d = [0] * len(s)
    print(d)
    d[0] = 1
    if int(s[0]) <= 2:
        d[1] = 2
    else:
        d[1] = 1
    for i in range(2, len(s)):
        if int(s[i - 1]) > 2:
            d[i] += d[i - 1] + 1
        else:
            d[i] += d[i - 1] + d[i - 2]
    print(d)


"""
剑指offer
"""


def erjinzhi(n):
    count = 0
    # while n != 0:
    #     # print(n, n - 1, n & (n - 1))
    #     n = n & (n - 1)
    #     count += 1
    # # print(count)

    count = 0
    print(bin(n))
    while n != 0:
        if (n & 1) != 0:
            count += 1
        n >>= 1
    print(count)


if __name__ == '__main__':
    # a2n()
    erjinzhi(100)
