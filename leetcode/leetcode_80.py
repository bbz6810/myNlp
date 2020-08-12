class Solution:
    def removeDuplicates(self, nums):
        p = 1
        while p < len(nums):
            q = p + 1
            while q < len(nums):
                if nums[p] == nums[p - 1]:
                    if nums[p] == nums[q]:
                        nums[q] = None
                q += 1
            p += 1

        none_pos = 0
        while none_pos < len(nums):
            if nums[none_pos] is not None:
                none_pos += 1
            else:
                break

        def find_none_pos(begin):
            for i in range(begin, len(nums)):
                if nums[i] is None:
                    return i

        def find_not_none_pos(begin):
            for i in range(begin, len(nums)):
                if nums[i] is not None:
                    return i

        q = p = none_pos
        while True:
            _q = find_not_none_pos(q)
            _p = find_none_pos(p)
            if _q is not None and _p is not None:
                # print(_q, _p)
                nums[_p] = nums[_q]
                nums[_q] = None
                q = _q
                p = _p
            else:
                break
        return len(nums) - nums.count(None)


if __name__ == '__main__':
    # num = [1,1,1,2,2,3]
    num = [0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    Solution().removeDuplicates(num)
    print(num)
