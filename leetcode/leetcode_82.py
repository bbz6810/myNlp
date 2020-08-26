# Definition for singly-linked list.
# 给定一个排序链表，删除所有含有重复数字的节点，只保留原始链表中 没有重复出现 的数字。
#
#  示例 1:
#
#  输入: 1->2->3->3->4->4->5
# 输出: 1->2->5
#
#
#  示例 2:
#
#  输入: 1->1->1->2->3
# 输出: 2->3
#  Related Topics 链表
# 一个指针指向当前节点，另一个指针从当前节点的下一个节点开始推进
# 如果相等则找到不等的那个节点停止，如果不等则把当前节点加入到新连表


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def deleteDuplicates(self, head):
        """依次循环，找该节点之后的所有重复节点

        :param head:
        :return:
        """
        p = head
        while p:
            q = p.next
            cur_num = p.val
            while q:
                if cur_num == q.val:
                    p.val, q.val = None, None
                q = q.next
            p = p.next

        p = head
        c = []
        while p:
            if p.val is not None:
                c.append(p.val)
            p = p.next

        if len(c) >= 1:
            _root = ListNode(c[0])
            _p = _root
            for i in c[1:]:
                t = ListNode(i)
                _p.next = t
                _p = _p.next
            return _root
        else:
            return None


if __name__ == '__main__':
    root = ListNode(0)
    p = root

    for i in [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]:
        t = ListNode(i)
        p.next = t
        p = p.next

    c = Solution().deleteDuplicates(root)
    while c:
        print(c.val)
        c = c.next
