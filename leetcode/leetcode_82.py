# Definition for singly-linked list.
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
        retv = None
        while p:
            if p.val is not None:
                t = ListNode(p.val)
                retv = t

        p = head
        while p:
            q = p
            while q:
                print(q.val, end=' ')
                q = q.next
            print()
            p = p.next


if __name__ == '__main__':
    root = ListNode(0)
    p = root
    for i in [1, 2, 3, 3, 4, 4, 5]:
        t = ListNode(i)
        p.next = t
        p = p.next

    Solution().deleteDuplicates(root)
