def bin_find_loop():
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # print(a)

    n = 3

    start = 0
    end = len(a)
    while start < end:
        mid = int((end + start) / 2)
        if a[mid] > n:
            end = mid - 1
        elif a[mid] < n:
            start = mid + 1
        else:
            # print('index1', mid)
            break
    target = 3

    """1、先判断中间字段是否是target
       2、判断左边的那一串是否是有序的
            判断target是否在左边，否则在右边
       3、判断右边的那一串是否是有序的
            判断target是否在右边，否则在左边
    """
    for i in range(10):
        b = a[i:] + a[:i]
        print(b)
        start = 0
        end = len(b)
        while start < end:
            mid = int((end + start) / 2)
            if b[mid] == target:
                print('index2', mid)
                break
            elif b[start] < b[mid]:
                if b[start] <= target <= b[mid]:
                    end = mid
                else:
                    start = mid
            else:
                if b[mid] <= target <= b[end - 1]:
                    start = mid
                else:
                    end = mid


def matix_99():
    import numpy as np
    data = np.zeros(shape=(9, 9), dtype='int32')
    print(data)
    for i in range(9):
        data[i][0] = 1
        data[0][i] = 1
    print(data)
    for i in range(1, 9):
        for j in range(1, 9):
            data[i][j] = data[i - 1][j] + data[i][j - 1]
    print(data)


def reverse_link():
    """p q   j是临时存储
         p q

    :return:
    """

    class Node:
        def __init__(self):
            self.data = None
            self.next = None

    def build_link(n, point):
        node = Node()
        node.data = n
        point.next = node
        return node

    root = Node()
    root.data = 0

    a = [1, 2, 3, 4, 5, 6, 7]
    node = root
    for i in a:
        node = build_link(i, node)

    p = root
    while p:
        print(p.data, p.next)
        p = p.next

    p = root
    q = root.next
    while q:
        if p == root:
            p.next = None
        j = q.next
        q.next = p

        p = q
        q = j

    while p:
        print(p.data, p.next)
        p = p.next


def yuan():
    """不同面值集合

    :return:
    """
    a = [1, 5, 10, 20, 50, 100]
    b = 20

    def loop(l, k):
        for i in a:
            if k + i < b:
                loop(l + [i], k + i)
            if k + i == b:
                print(l + [i])
                return

    loop([], 0)


if __name__ == '__main__':
    # bin_find_loop()
    # matix_99()
    # reverse_link()
    yuan()
