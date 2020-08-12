class Solution:
    def search(self, nums, target):
        """1、先判断中间字段是否是target
           2、判断左边的那一串是否是有序的
                判断target是否在左边，否则在右边
           3、判断右边的那一串是否是有序的
                判断target是否在右边，否则在左边
           4、否则的话，判断开始和中间是否相等
        """

        start = 0
        end = len(nums)
        while start < end:
            mid = int((end + start) / 2)
            if nums[mid] == target:
                return True
            elif nums[start] < nums[mid]:
                if nums[start] <= target <= nums[mid]:
                    end = mid
                else:
                    start = mid
            elif nums[mid] < nums[end - 1]:
                if nums[mid] <= target <= nums[end - 1]:
                    start = mid
                else:
                    end = mid
            else:
                if nums[start] == nums[mid]:
                    start += 1
                else:
                    end -= 1
        return False


if __name__ == '__main__':
    a = [2, 3, 4, 5, 6, 1, 2, 2, 2]
    a = [3,4,5,6,0,0,1,2,3]
    # a = [1, 1, 1, 3, 1]

    n = 3
    s = Solution().search(a, n)
    print(s)
