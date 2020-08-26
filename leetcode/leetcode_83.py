# 给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。
#
#  示例 1:
#
#  输入: 1->1->2
# 输出: 1->2
#
#
#  示例 2:
#
#  输入: 1->1->2->3->3
# 输出: 1->2->3
#  Related Topics 链表


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def deleteDuplicates(self, head):
        p = head
        while p:
            q = p.next

            while q and q.val == p.val:
                q = q.next
            if q is not None:
                p.next = q
            else:
                p.next = None
                break
            p = p.next
        return head


if __name__ == '__main__':
    root = ListNode(0)
    p = root

    for i in [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 8]:
        # for i in [1, 2]:
        t = ListNode(i)
        p.next = t
        p = p.next

    c = Solution().deleteDuplicates(root)

    while c:
        print(c.val)
        c = c.next
