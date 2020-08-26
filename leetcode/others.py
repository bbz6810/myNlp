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
        if int(s[i-1]) > 2:
            d[i] += d[i-1] + 1
        else:
            d[i] += d[i - 1] + d[i - 2]
    print(d)


if __name__ == '__main__':
    a2n()
