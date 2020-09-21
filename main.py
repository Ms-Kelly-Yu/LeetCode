from inspect import stack
import heapq


def caculate(x, operator, y):
    if operator == "+":
        return x + y
    elif operator == "-":
        return x - y


def store_as_stack(s):
    stack = []
    i = len(s) - 1
    while i > -1:
        stack.append(s[i])
        i -= 1
    return stack


def parentheses(s):
    stack = []
    for ch in s:
        if stack and is_sym(ch, stack[len(stack) - 1]):
            stack.pop()
        else:
            stack.append(ch)
    if stack:
        return False
    return True


def is_sym(c, s):
    if s == "(" and c == ")":
        return True
    if s == "[" and c == "]":
        return True
    if s == "{" and c == "}":
        return True
    return False


class Solution:
    # open closing parentheses, + - ' '
    def cal(self, stack):
        operators = []
        result = 0
        current_num = 0
        while stack:
            c = stack.pop()
            if c.isdigit():
                current_num = current_num * 10 + int(c)
            else:
                if current_num != 0 and not operators:
                    result = current_num
                    current_num = 0

                if c == "+" or c == "-":
                    if operators:
                        result = caculate(result, operators.pop(), current_num)
                        current_num = 0
                        operators.append(c)
                    else:
                        operators.append(c)
                elif c == "(":
                    if operators:
                        result = caculate(result, operators.pop(), self.cal(stack))
                    else:
                        result = self.cal(stack)
                elif c == ")":
                    break
        if operators:
            result = caculate(result, operators.pop(), current_num)
        elif not result:
            return current_num
        return result


class Solution(object):
    def divide(self, dividend, divisor):
        if divisor == 1:
            return dividend
        elif divisor == -1:
            if dividend == -pow(2, 31):
                return pow(2, 31) - 1
            return -dividend
        result = 0
        multiple = 1
        sign = -1
        if (divisor > 0 and dividend > 0) or (divisor < 0 and dividend < 0):
            sign = 1
        divisor = abs(divisor)
        multiple_divisor = divisor
        dividend = abs(dividend)
        while dividend > 0 and dividend >= multiple_divisor:
            result += multiple
            dividend = dividend - multiple_divisor
            multiple_divisor += multiple_divisor
            multiple += multiple
        if abs(dividend) >= abs(divisor):
            result += self.divide(dividend, divisor)
        if sign == -1:
            result = -result
        return result

    def divide2(self, dividend, divisor):
        if divisor == 1:
            return dividend
        elif divisor == -1:
            if dividend == -pow(2, 31):
                return pow(2, 31) - 1
            return -dividend
        if -abs(dividend) > -abs(divisor):
            return 0
        sign = -1
        if (divisor > 0 and dividend > 0) or (divisor < 0 and dividend < 0):
            sign = 1
        divisor = -abs(divisor)
        dividend = -abs(dividend)
        if sign == -1:
            return -self.divide_loop(dividend, divisor)
        return self.divide_loop(dividend, divisor)

    def divide_loop(self, dividend, divisor):
        result = 0
        multiple_divisor = divisor
        while 1:
            if result == 0:
                result = 1
            else:
                result += result
            if multiple_divisor + multiple_divisor < dividend:
                break
            multiple_divisor += multiple_divisor
        if dividend - multiple_divisor <= divisor:
            result += self.divide(dividend - multiple_divisor, divisor)
        return result

    def divide3(self, dividend, divisor):
        if divisor == 1:
            return dividend
        elif divisor == -1 and dividend == -2147483648:
            return 2147483647
        elif divisor == -1:
            return -dividend
        sign = 1
        if divisor ^ dividend < 0:
            sign = -1
        if sign == -1:
            return -(self.divide3_loop(-abs(dividend), -abs(divisor)))
        return self.divide3_loop(-abs(dividend), -abs(divisor))

    def divide3_loop(self, dividend, divisor):
        if divisor < dividend:
            return 0
        result = 0
        while 1:
            if result == 0 and divisor << 1 <= dividend:
                break
            elif result == 0:
                result = 1
            if (divisor << (result + result)) < dividend:
                while 1:
                    if (divisor << (result + 1)) < dividend:
                        break
                    result += 1
                break
            result += result
        if dividend - (divisor << result) < divisor:
            return (1 << result) + self.divide3_loop(dividend - (divisor << result), divisor)
        return 1 << result


# print(Solution().divide3(-27483648, 2))


class Solution:
    def findRepeatedDnaSequences(self, s: str):
        match = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        nums = [int(match.get(s[i])) for i in range(len(s))]
        scale, L, num = 4, 10, 0
        scaleL = pow(scale, L)
        seen, output = set(), set()
        for i in range(len(s) - L + 1):
            if i == 0:
                for n in range(L):
                    num = scale * num + nums[n]
            else:
                num = num * scale - nums[i - 1] * scaleL + nums[i + L - 1]
            if num in seen:
                output.add(s[i: i + L])
            seen.add(num)
        return list(output)


# Solution().findRepeatedDnaSequences("ACACACACACACAC")
# print(Solution().divide3(10, -3))
# print(Solution().divide3(-27483648, -2))
import heapq


class KthLargest1:

    def __init__(self, k, nums):
        """
        :type k: int
        :type nums: List[int]
        """
        self.nums = [_ for _ in range(nums)]
        self.k = k
        heapq.heapify(self.nums)
        while len(self.nums) > self.k:
            heapq.heappop(self.nums)

    def add(self, val):
        """
        :type val: int
        :rtype: int
        """
        if self.nums[0] < val:
            heapq.heappop(self.nums)
            heapq.heappush(self.nums, val)
        return self.nums[0]


def acc(nums):
    stones = [-i for i in nums]
    heapq.heapify(stones)
    while len(stones) > 1:
        y = heapq.heappop(stones)
        x = heapq.heappop(stones)
        if x > y:
            heapq.heappush(stones, y - x)
    if len(stones) == 1:
        return -stones[0]
    return 0


def smallestK(k, arr):
    if k == 0:
        return 0
    smallest = [-i for i in arr[:k]]
    heapq.heapify(smallest)
    for a in arr[k:]:
        if smallest[0] < -a:
            heapq.heapreplace(smallest, -a)
    return [-i for i in smallest]


def stock(prices):
    benefit = 0
    min = 0
    for i in range(len(prices) - 1):
        if prices[i + 1] > prices[i]:
            benefit += prices[i + 1] - prices[i]
        i += 1
    return benefit


def stock2(prices):
    benefit = 0
    min, max = prices[0], prices[0]
    for price in prices[1:]:
        if price >= max:
            max = price
        else:
            benefit += max - min
            min, max = price, price
    return benefit + max - min


def stock3(prices):
    dp = [[0, 0] for i in range(len(prices))]
    dp[0][0] = 0
    dp[0][1] = -prices[0]
    for i in range(1, len(prices)):
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
        dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
    return dp[-1][0]


def contain(sub, t):
    j = 0
    for s in t:
        if sub[j] == s:
            j += 1
        if j == len(sub):
            return True
    return False


def contain2(sub, t):
    state = [[-1] * len(t) for i in range(26)]
    l = len(t)
    for i in range(1, len(t) + 1):
        for j in range(26):
            state[j][-i] = state[j][-i + 1]
        state[ord(t[-i]) - 97][-i] = l - i
    for s in sub:
        index = 0
        res = True
        for i in range(len(s)):
            if (index == -1 and i != 0) or index == l - 1:
                res = False
                break
            index = state[ord(s[i]) - 97][index + 1]
        print(res)


def contain3(s, t):
    state = [[-1] * len(t) for i in range(26)]
    l = len(t)
    for i in range(1, len(t) + 1):
        for j in range(26):
            state[j][-i] = state[j][-i + 1]
        state[ord(t[-i]) - 97][-i] = l - i
    index = -1
    for i in range(len(s)):
        if (index == -1 and i != 0) or index == l - 1:
            return False
        index = state[ord(s[i]) - 97][index + 1]
    if index == -1:
        return False
    return True


def group(candidates, target):
    can = sorted(candidates)
    i = -1
    while can[-1] > target:
        can.remove(can[-1])
    res = [[]]
    if can[-1] == target:
        res.append(can[-1])
        can.remove(can[-1])
    cur = []
    for i in range(1, len(can) + 1):
        cur.clear()
        cur.append(can[-i])
        curn = can[-i]
        j = 0
        while j:
            if curn + can[j] > target:
                cur.pop()
            elif curn + can[j] < target:
                cur.append(can[j])
                curn += can[j]
                j += 1


def anagram(s, t):
    if len(s) != len(t):
        return False
    arr = [0] * 26
    for c in s:
        arr[ord(c) - 97] += 1
    for c in t:
        arr[ord(c) - 97] -= 1
        if arr[ord(c) - 97] < 0:
            return False
    for i in arr:
        if i != 0:
            return False
    return True


def sort(self, head):
    if not head or head.next:
        return head
    slow, fast = head, head.next
    while fast and fast.next:
        fast, slow = fast.next.next, slow.next
    mid, slow.next = slow.next, None
    left, right = self.sort(head), self.sort(mid)
    h = res = []
    while left and right:
        if left.val < right.val:
            h.nex, left = left, left.next
        else:
            h.next, right = right, right.next
        h = h.next
        h.next = left if left else right
        return res.next


# print(anagram('sfs','ssf'))
# print(contain3('acb', 'ahbgdc'))
# print(stock3([7, 1, 5, 3, 6, 4]))


# 归并-递归
def mergesort(arr):
    if len(arr) < 2:
        return arr
    import math
    middle = math.floor(len(arr) / 2)
    left, right = arr[: middle], arr[middle:]
    return merge(mergesort(left), mergesort(right))


def merge(left, right):
    result = []
    while left and right:
        if left[0] <= right[0]:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))
    while left:
        result.append(left.pop(0))
    while right:
        result.append(right.pop(0))
    return result


# 归并-迭代
def mergesortloop(arr):
    block = 2
    l = len(arr)
    import math
    while block < l * 2:
        result = []
        for j in range(0, math.ceil(l / block)):
            x, y = block * j, block * j + int(block / 2)
            while x < block * j + (block / 2) and y < block * (j + 1) and x < l and y < l:
                if arr[x] <= arr[y]:
                    result.append(arr[x])
                    x += 1
                else:
                    result.append(arr[y])
                    y += 1
            while x < block * j + (block / 2) and x < l:
                result.append(arr[x])
                x += 1
            while y < block * (j + 1) and y < l:
                result.append(arr[y])
                y += 1
        block *= 2
        arr = result
    return arr


# 计数排序
def countsorting(arr, maxvalue):
    store, result = [0] * (maxvalue + 1), []
    for n in arr:
        store[n] += 1
    for i in range(len(store)):
        while store[i] > 0:
            result.append(i)
            store[i] -= 1
    return result


# 计数排序 - minimum
def countsorting(arr, maxvalue, minvalue):
    store, result = [0] * (maxvalue - minvalue + 1), []
    for n in arr:
        store[n - minvalue] += 1
    for i in range(len(store)):
        while store[i] > 0:
            result.append(i + minvalue)
            store[i] -= 1
    return result


# 5 buckets
def bucketsorting(arr, maxvalue):
    import math
    bucketzone = math.ceil(maxvalue / 5)
    buckets = [[], [], [], [], []]
    result = []
    for n in arr:
        buckets[math.floor(n / bucketzone)].append(n)
    for i in range(len(buckets)):
        buckets[i] = countsorting(buckets[i], (i + 1) * bucketzone, i * bucketzone)
        result.extend(buckets[i])
    return result


def quicksort(arr, l, r):
    base = arr[l]
    i, j = l, r
    while i < j:
        while i < j:
            if arr[j] < base:
                arr[i] = arr[j]
                break
            j -= 1
        while i < j:
            if arr[i] > base:
                arr[j] = arr[i]
                break
            i += 1
    arr[i] = base
    if l < i:
        quicksort(arr, l, i)
    if i + 1 < r:
        quicksort(arr, i + 1, r)
    return arr


def greedalgorithm(g, s):
    g, s, res = sorted(g), sorted(s), 0
    for i in g:
        while s:
            if i <= s.pop(0):
                res += 1
                break
    return res


def canJump(nums):
    if not nums:
        return True
    target, mid, i = len(nums) - 1, 0, 0
    while i < len(nums):
        if nums[i] + i >= target:
            return True
        if nums[i] == 0:
            return False
        mid, index = 0, i
        for j in range(1, nums[i] + 1):
            if nums[i + j] + i + j >= target:
                return True
            if nums[i + j] + j > mid:
                mid = nums[i + j] + j
                index = i + j
        i = index


def canjumpgreed(nums):
    if not nums or len(nums) == 1:
        return True
    l, counting = len(nums), 0
    for i in range(l):
        if counting < i:
            break
        if counting < nums[i] + i:
            counting = nums[i] + i
        if nums[i] + i + 1 >= l:
            return True
    return False


'''
print(canjumpgreed([3, 0, 8, 2, 0, 0, 1]))
print(canjumpgreed([2, 0, 0]))
print(canjumpgreed([2, 3, 1, 1, 4]))
print(canjumpgreed([3, 2, 1, 0, 4]))
print(canjumpgreed([0]))'''


# print(quicksort([2, 4, 6, 8, 1, 0, 45, 56, 45, 23], 0, 9))
# print(bucketsorting([2, 4, 6, 8, 1, 0, 45, 56, 45, 23], 56))
# print(countsorting([23, 45, 56, 45, 26, 50], 56, 23))
# print(countsorting([2, 4, 6, 8, 1, 0, 45, 56, 45, 23], 56))
# print(mergesortloop([3, 1, 54, 7, 34, 89, 9, 0]))
# print(mergesort([3, 1, 54, 7, 34, 89, 9, 0]))

# ""1111111"
# 3" 1
def removeKdigits(num, k):
    if len(num) == 0 or len(num) <= k:
        return '0'
    if k == 0:
        return num
    for i in range(k):
        sec = 1
        while sec < len(num) and num[0] == num[sec]:
            sec = sec + 1
        sec = sec if sec < len(num) else sec - 1
        if len(num) > 2 and num[0] >= num[sec]:
            num = num[1:]
        elif len(num) > 2 and num[0] < num[sec]:
            num = num[:sec] + num[sec + 1:]
        else:
            num = num[1] if num[0] > num[1] else num[0]
        while len(num) > 0 and num[0] == '0':
            num = num[1:]
    return num if len(num) > 0 else "0"


def removeKdigits(num, k):
    if len(num) == 0 or len(num) <= k:
        return '0'
    if k == 0:
        return num
    res = [num[0]]
    for i in range(1, len(num)):
        if k > 0 and res[-1] > num[i]:
            k -= 1
            res.pop(-1)
        else:
            res.append(num[i])


print(removeKdigits("1234567890", 9))
