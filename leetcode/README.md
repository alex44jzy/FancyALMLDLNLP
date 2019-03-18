# Python 刷Leetcode包含思路题解，每日一题
## 所有答案均可以在Leetcode上直接运行。

## Leetcode-70 爬楼梯问题（DP）
> You are climbing a stair case. It takes n steps to reach to the top.
Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
Note: Given n will be a positive integer.
Input: 2
Output: 2
Explanation: There are two ways to climb to the top.

<!--more-->

一道经典的爬楼梯问题，直觉上第一个想到的就是采用递归，也就是要计算爬到第3层楼梯有几种方式，可以从第2层爬1级上来，也可以从第1层爬2级上来，所以爬到第3级有几种方式只需要将到第2层总共的种数，加上到第1层总共的种数就可以了。推广到一般，写出递推公式
$stairs(n) = stairs(n-1) + stairs(n-2) $，只需要初始化好退出递归的条件就算写完了。

方法1，直接采用递归。
```python
class Solution:
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 1:
            return 1
        if n == 2:
            return 2
        return self.climbStairs(n - 1) + self.climbStairs(n - 2)
```
然而，没有AC (ーー゛)，理由是超时。这就引出了在递归里经常会采用的备忘录法，因为这里面同一个n被重复计算了n次，因此一定程度上影响了性能，比如stairs(5) = stairs(4) + stairs(3), stairs(4) = stairs(3) + stairs(2)，stairs(3)就被计算了2次，因此借助一个字典存储计算过的值，就可以大大减少重复的计算了，就诞生了备忘录形式的递归。

方法2，备忘录递归
```python
class Solution:
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        refs = dict() # 建立字典类型备忘录
        def rec(n):
            # 初始条件写入备忘录
            if n <= 1:
                refs[n] = 1
                return 1
            if n == 2:
                refs[n] = 2
                return 2
            # 存在于字典的直接输出
            if n in refs:
                return refs[n]
            else:
                refs[n] = rec(n - 1) + rec(n - 2)
                return refs[n]
        return rec(n)
```
这次AC了，︿(￣︶￣)︿

既然都已经做到备忘录了，那其实和动态规划也就没有什么两样，递归采用自顶向下，动态规划采用自底向上，借助一个数组来加以实现，要计算n阶就往对应数组里插入到n阶。

方法3，动态规划
```python
class Solution(object):    
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        res = [1, 1, 2] #初始化
        if n >=3:
            for i in range(3, n + 1):
                res.append(res[i - 1] + res[i - 2])
        return res[n]
```
也是AC的，\（￣︶￣）/

不难发现，这个递推公式有点像Fibonacci数列，其实就是Fibonacci数列。。。因此也可以借助Fibonacci数列递推的思想直接就可以写出来了。

方法4，Fibonacci递推
```python
class Solution(object):    
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 1:
            return 1
        if n == 2:
            return 2
        if n >= 3:
            first, second = 1, 2
            for i in range(3, n + 1):
                third = first + second
                first = second
                second = third
            return second
```
比较简单和基础的一道题，以上


## leetcode-242 重排校验
> Given two strings s and t , write a function to determine if t is an anagram of s.
Example 1:
Input: s = "anagram", t = "nagaram"
Output: true

要满足重排，就一定要含有相同个数的字幕，那么就可以转化成 list of chars，看每一个sort过后的list是否相同就可以了。
写一个最简单的排序方法

```python
class Solution:
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        origin = sorted(list(s))
        current = sorted(list(t))
        return origin == current
```
另外就是可以尝试一下Counter这种计数器的方法。虽然这个方法挺不要脸的，利用Counter直接生成一个hashmap。

```python
from collections import Counter
class Solution:
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        return Counter(s) == Counter(t)
```

## leetcode-104 二叉树最大深度
> Given a binary tree, find its maximum depth.
The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
Note: A leaf is a node with no children.
Given binary tree [3,9,20,null,null,15,7],return its depth = 3.

直觉上，看例子以为是要用数组来做，但是应该是用树。用DFS的思想借助递归实现，（**重点**： 树的题一般都要用递归来解决），先递归左子树，层层深入下去，注意一点在python中，
```python
max(0, None) = 0 
```
也就意味着，对某个节点，如果左节点叶子存在，右节点叶子不存在为None的话，这时max()函数取值也为0。
整理得出如下：

```python
class Solution:
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root is None:
            return 0
        else:
            left_height = self.maxDepth(root.left)
            right_height = self.maxDepth(root.right)
            return max(left_height, right_height) + 1
```
<br>

## leetcode-160 链表交集
> Write a program to find the node at which the intersection of two singly linked lists begins.
For example, the following two linked lists:
```java
A:          a1 → a2
                   ↘
                     c1 → c2 → c3
                   ↗            
B:     b1 → b2 → b3
```
begin to intersect at node c1.

链表题用指针，或者双指针还是挺常用的方法。有看到说用字典来存储的，感觉虽然能解决问题，但是一定程度上破坏了链表的数据结构。分析题目，大概有那么2个思路。
1. 指针顺序遍历，解决一个问题就是2个链表长度不同，所以第一步要遍历得到2个链表的长度（这块可能空间复杂度开销比较大）。将长链表向前移动至剩余链表长度与短链表一致。
```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        if headA == None or headB == None:
            return None
        len_a, len_b = 0, 0
        # 赋值计算长度
        p, q = headA, headB
        while p:
            p = p.next
            len_a += 1
        while q:
            q = q.next
            len_b += 1
        
        # 赋值截断到相同长度
        p, q = headA, headB
        if len_a > len_b:
            for i in range(len_a - len_b):
                p = p.next
        else:
            for i in range(len_b - len_a):
                q = q.next
        
        while p != q:
            p = p.next
            q = q.next
        return p
```

2. 最大的障碍是2个链表长度不同，所以有一个巧妙的办法来补齐。就是当一个链表先行到达链表尾部时，将Next指针去指向另一个链表的头部，同理另一个链表也同样如此，这样就保证了在O（m+n）内一定能找到结果。举个例子：
ListNodeA = 0, 9, 1, 2, 4
LIstNodeB = 3, 2, 4
Path of A -> B = 0, 9, 1, 2, 4, 3, **2, 4**
Path of B -> A = 3, 2, 4, 0, 9, 1, **2, 4**
这样只需要遍历一次，如果遍历结束2个指针仍然不同都指向None，也就没有交集.
```python
class Solution(object):
        
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        p, q = headA, headB
        
        while p != q:
            p = headB if p is None else p.next
            q = headA if q is None else q.next
        return p
```

## Leetcode-204 质数个数
> Count the number of prime numbers less than a non-negative number, n.
Example:
Input: 10
Output: 4
Explanation: There are 4 prime numbers less than 10, they are 2, 3, 5, 7.

乍一看以为应该没什么的题目，结果写了一版，先是边界条件也写好，再就是TLE感觉过不去了。先来一个基本方法吧，毕竟我不可能在实战中短时间想到一个fancy的方法。

```python
import math
class Solution:
    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """
        import math
        count = 0

        # 一个判断是否为质数的方法
        def judge_prime(w):
            sqrt_w = int(math.sqrt(w))
            # 迭代相除直到sqrt(w)
            # 注意传入2时，sqrt(2) + 1 = 2， range(2, 2)不会执行，因此还是会返回1
            # 不然边界条件太乱了。
            for i in range(2, sqrt_w + 1):
                if x % i == 0:
                    return 0
                
            return 1

        for x in range(2, n):
            count = count + judge_prime(x)

        return count
```
但是这个方法没有在Leetcode上AC，主要还是太慢了。

看了网上的大神介绍了一个厄拉多塞筛法(Sieve of Eeatosthese)。先上代码，

```python
def countPrimes(self, n):
    if n < 3:
        return 0
    primes = [True] * n
    primes[0] = primes[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if primes[i]:
            primes[i * i: n: i] = [False] * len(primes[i * i: n: i])
    return sum(primes)
```

对着代码解释下大概的思想是：
1. 创建一个数组，长度为n个True，第0，1个位置设置为False，即0和1不是质数。
2. 从2开始（True，初始化），用2举例，以2*2作为起点开始迭代，迭代的终点是n，步长为2，将满足的都标记为False。
3. 同理3，4由于已经被2标记了跳过，5同理，6被标记，7同理，循环的终点就是$\sqrt{n}$。这里解释一下，因为一共有n个数，每次的起点是i\*i，这是因为避免了重复的计算，比如3是从3\*3开始的，因为3*2已经计算过了，$\sqrt{n} * \sqrt{n}$作为终点也是保证了不重复计算。总的遍历从2到$\sqrt{n}$，次数下降了指数级别。
4. 这里又借助了python list的可以设置变步长的性质 [i*i : n : i]，也不需要额外开辟更大的空间，可以说是时间复杂度和空间复杂度都做到了极致。
5. AC了，但是凭空我肯定是想不到的。

## Leetcode-17 Letter Combinations of a Phone Number 字母组合
> Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent.
A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

> Input: "23"
Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].

其中按键对应的字母映射要自己建一个dictionary，汗。。。如下：

(letters = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', 
                   '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'})


这个题乍一看，典型的递归，深度遍历的长相，于是乎，开始写了起来，毕竟是递归嘛，脑子大概想着要有一个重复子问题。也差不多想到了就是每一次迭代都要拿上一次的结果，拼接上新一次的字母，而且这个拼接要是一个组合，所以总体来说规模是要逐步扩大的。因此，既然要扩大，应该是在for循环里套一层递归的大概样子。写了一个大概的版本，尽然AC了。。。猝不及防
```python
class Solution:
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        letters = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', 
                   '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        
        def dfs(res, index):    
            new_lts = [i for i in letters[input[index]]]
            res = [i + j for i in res for j in new_lts] # 拼接生成最新的结果
            index += 1
            # 递归结束标志
            if index > len(input) - 1:
                return res
            else:
                return dfs(res, index) 每次传入当前结果，以及下一个新的数字键
                
        input = list(digits)
        如果输入为空，则返回空。
        if len(input) < 1:
            return []
        # 初始化从1开始
        initial = [i for i in letters[input[0]]]
        idx = 1
        if len(input) == 1:
            return initial
        
        return dfs(initial, idx)
```

但是这个递归是写的有点问题的，其实似乎就有点不像个递归了。原因在于因为递归是把复杂问题化小，递归到越来越简单的规模。而我这里只是借助了一个递归来传递我每一次新的值给下一个数字键，既然是传递，那其实可以直接写for循环来实现的，于是我改了改：

```python
class Solution:
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        letters = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', 
                   '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        
        input = list(digits)
        if len(input) < 1:
            return []
        
        comb = [i for i in letters[input[0]]]
        for num in input[1:]:
            comb = [i + j for i in comb for j in letters[num]]
        return comb
```
这样改写之后，就比我之前的第一种顺眼多了。这里的技巧就是循环重复更改结果，第一种里的res和第二种里的comb。
而真正使用递归来求解，我参照了一下大神的标准答案，很巧妙运用了化繁为简的策略，这里运用了一个小技巧就是使用了list的[:-1]来递归前一种情况。也就是说本来比如输入5个数字，那递归到第5次得到的结果就是前4次加上第5次的，那第4次就是前3次加第4次的，以此类推。递推公式为 
$$comb(n) = F(\ comb(n-1),\ curr(n)\ )$$
其中$F()$函数就是求组合数，这样想了一下就可以写了如下：
```python
class Solution:
    # @param {string} digits
    # @return {string[]}
    def letterCombinations(self, digits):
        mapping = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', 
                   '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        if len(digits) == 0:
            return []
        if len(digits) == 1:
            return list(mapping[digits[0]])
        prev = self.letterCombinations(digits[:-1])
        # additional就是current
        additional = mapping[digits[-1]]
        return [s + c for s in prev for c in additional] #生成新的组合
```
## Leetcode-416 Partition Equal Subset Sum 分割相等子集
> Given a non-empty array containing only positive integers, find if the array can be partitioned into two subsets such that the sum of elements in both subsets is equal.

> Note:
Each of the array element will not exceed 100.
The array size will not exceed 200.
Example 1:
Input: [1, 5, 11, 5]
Output: true
Explanation: The array can be partitioned as [1, 5, 5] and [11].

经典题，说实话写这个题花了不少时间，尤其是想清楚这个里外里的道理。这道题是一道典型的背包问题，但是在成为背包问题之前，要做一些必要的转换。
- 首先，题目要求2个子数组和相等，因此，只有大数组的和为偶数，才可能使得2个子数组和是相等，所以一旦和为奇数就可以立即排除输出`False`。
- 其次对于和为偶数的情况，可以将总和除2作为每个子数组的目标和，所以问题就转化为找出子数组和为$target = sum / 2$。
- DP问题就是要借助DP的思想，就要写一下状态转移方程，作为写代码的指导思想。动态规划都是借助一个2维或者1维的表格来建立整个过程，于是我们借一个例子来画一下表格：
```python
	0	1	2	3	4	5	6	8	9	10	11
0	1	0	0	0	0	0	0	0	0	0	0
1	1	1	0	0	0	0	0	0	0	0	0
5	1	1	0	0	0	1	1	0	0	0	0
5	1	1	0	0	0	1	1	0	0	0	1
```
行表示每一个数，列表示能够达到所有可能的和（这里可以记忆一下，DP问题是由繁化简，所以一定是有状态从最起始开始一直到我们的目标，在例子中起始为0，目标是11，中间所有可能的值也都要考虑，这点和背包问题是一样的）。由于我们现在的问题是数组和能否达到某个固定的数，因此可以写得递推公式为$dp[i][j] = dp[i-1][j] || dp[i-1][j-nums[i]]$。

稍微解释一下这个公式，第一项$dp[i-1][j]$是说，如果上一行（i-1）已经满足和为j了，那下一行也一定会满足和为j，所以i项不加也可以满足，状态照搬下来就可以，就好比01背包问题，不取当前项，最大价值仍然延续上一个状态一个意思。而$dp[i-1][j-nums[i]]$表示前i-1个数，在target - nums[i
]处已经满足，那加上nums[i]也同样会满足，显示了DP的递推性。这样主体就搭建好了。

在这里也要注意一下初始值的问题，j为0意味着和为0，也就是不取任何数，所以dp[:][j=0]要设置为True（一个都不取，什么都不用做就满足了，也是强）。于是就可以写下代码了：

```python
class Solution(object):
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if sum(nums) % 2 == 1:
            return False
        target = sum(nums) // 2
        
        # dp[i][j] = dp[i-1][j] || dp[i-1][j-nums[i]]
        dp = [[False for i in range(target + 1)] for j in range(len(nums) + 1)]
        dp[0][0] = True
        for i in range(1, len(nums) + 1):
            for j in range(1, target + 1):
                dp[i][j] = dp[i-1][j]
                if j >= nums[i-1] and dp[i-1][j-nums[i-1]]:
                    dp[i][j]  = True
        return dp[-1][-1]
```
这里有可以改进的地方，就是原始是采用了一个二维数组来存储数据的，其实可以优化为一维数组的，这一点和背包问题是一致的，需要注意的是内层的遍历要降序遍历，来防止对已经生成的数据做重复修改，不展开了，可以参考[背包问题整理（二维转一维数组)](https://blog.csdn.net/sunshine_lyn/article/details/79482477)。改写如下：
```python
class Solution(object):
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if sum(nums) % 2 == 1:
            return False
        target = sum(nums) // 2
        
        # dp[i][j] = dp[i-1][j] || dp[i-1][j-nums[i]]
        dp = [False for i in range(target + 1)]
        dp[0] = True
        for i in range(0, len(nums)):
            for j in range(target, nums[i] - 1, -1):
                dp[j] = dp[j] | dp[j - nums[i]]
        return dp[-1]
                
```
这题有用别的奇技淫巧来解的，我就不推荐了，主要还是可以借机会复习01背包动态规划的思想，以及转移状态的确定和思维的转换。其他大神的办法无非是在DFS上做文章，我是真的想不到，就老老实实的吧，毕竟慢学。


## Leetcode-215 第K个最大的数
> Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the kth distinct element.

> Example 1:
Input: [3,2,1,5,6,4] and k = 2
Output: 5

这道题聊解题没有任何意义，主要是周末想借这个机会复习一下所有的排序算法思想，先给个最简单的题解吧。
```python
class Solution:
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        nums.sort(reverse=True)
        return nums[k - 1]
```
这里顺便复习几个比较常用的排序算法：冒泡，选择，插入，归并，快排。
冒泡：
```python
class Solution:
    def findKthLargest(self, nums, k):
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if nums[j] > nums[i]:
                    nums[i], nums[j] = nums[j], nums[i]
        return nums[k-1]
```
选择：
```python
class Solution:
    def findKthLargest(self, nums, k):
        for index in range(len(nums)):
            tmp = index
            for j in range(index + 1, len(nums)):
                if nums[j] > nums[tmp]:
                    tmp = j
            nums[index], nums[tmp] = nums[tmp], nums[index]
        return nums[k - 1]
```
插入，这个还是想了一下的，主要是用while的话更加清楚一些。
```python
class Solution:
    def findKthLargest(self, nums, k):
        for i in range(1, len(nums)):
            key = nums[i]
            j = i - 1
            while j >= 0 and nums[j] < key:
                # 只要key比已排序好的数大，先对原数进行移位操作，腾出空间。
                nums[j + 1] = nums[j] #会有个重复
                j -= 1
            # while条件不满足时，这里nums[j + 1]其实是(j - 1) + 1 = j，将key置于上步腾出的那个位置
            nums[j + 1] = key
        return nums[k-1]
```
归并：
```python
class Solution:
    # 2. 再对left和right作归并排序
    def merge_sort(self, left, right):
        i, j = 0, 0 
        result = []
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result += left[i:]
        result += right[j:]
        return result
    # 1. 先拆分，运用递归
    def divide_merge(self, nums):
        if len(nums) <= 1:
            return nums
        num = len(nums) // 2
        left = self.divide_merge(nums[:num])
        right = self.divide_merge(nums[num:])
        return self.merge_sort(left, right)
    
    def findKthLargest(self, nums, k):
        res = self.divide_merge(nums)  
        return res[-k]
```
快排：
```python
class Solution:
def findKthLargest(self, nums, k):
    pivot = nums[0]
    left  = [l for l in nums if l < pivot]
    equal = [e for e in nums if e == pivot]
    right = [r for r in nums if r > pivot]

    if k <= len(right):
        return self.findKthLargest(right, k)
    elif (k - len(right)) <= len(equal):
        return equal[0]
    else:
        return self.findKthLargest(left, k - len(right) - len(equal))
```
这里面比较难想的是插入，归并和快排，有必要做一个专题来攻克一下。虽然以前也学过，而且说来说去也大概知道原理，但是真正自己实现的时候，还是没有那么快能解决。


## Leetcode-46 Permutations 排列
> Given a collection of distinct integers, return all possible permutations.
Example:
Input: [1,2,3]
Output:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]

一道经典的回溯题，看到这样的例子，第一反应就是用DFS搭配回溯，既然是深度优先遍历，一般会用一个visited来记录遍历到的位置，运用for加递归的方式进行遍历，设置退出的条件。回溯算法需要将visited的状态回退，这里运用了一个pop方法将栈中的元素末尾弹出后回溯到上一层状态。这里因为是for循环加上递归的结构，递归其实可以看成是一个一般的函数调用（有返回值），当次的for循环结束后，弹出末尾元素，下一次循环后再压入，以此往复。
```python
class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        def helper(nums, visited):
            if len(visited) == len(nums):
                res.append(visited[:])
            
            for item in nums:
                # 如果已经存在就不再记入
                if item in visited:
                    continue
                visited.append(item)
                helper(nums, visited)
                visited.pop()
        helper(nums, [])
        return res
```
AC了，看到另外的解法，避免了在循环中进行判断，而是在传入下层递归的时候，就只传入除去当次循环的那个元素后的剩余元素，要巧妙的多。
```python
class Solution(object):
    # DFS
    def permute(self, nums):
        res = []
        self.dfs(nums, [], res)
        return res

    def dfs(self, nums, path, res):
        if not nums:
            res.append(path)
            # return # backtracking
        for i in range(len(nums)):
            self.dfs(nums[:i]+nums[i+1:], path+[nums[i]], res)
```

## Leetcode-647 Palindromic Substrings 回文子串个数
> Given a string, your task is to count how many palindromic substrings in this string.
The substrings with different start indexes or end indexes are counted as different substrings even they consist of same characters.

> Example 1:
Input: "abc"
Output: 3
Explanation: Three palindromic strings: "a", "b", "c".
Example 2:
Input: "aaa"
Output: 6
Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".

经典题，求回文子串个数，应该只要循环加递归就可以搞定了，理一下思路：回文的特征就是以某一个中心位置为轴对称，但是这里有一点要注意就是奇数个子串有中心位置，但是偶数个数的子串中心位置应该是两个数。明确了这一点之后，循环到某一个位置时，设置两个指针i和j分别从中心位置++和--，直到任何一个指针到达边界小于0或超过数组长度截止，如果有满足 s[i] == s[j] 就继续迭代进行。思路不难，实现了一下：
```python
class Solution:
    def __init__(self):
        self.count = 0
        
    def countSubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        def move_check(input, i, j):
            if i < 0 or j >= len(input):
                return        
            
            if input[i] == input[j]:
                self.count += 1
            else: # 不满足及时跳出，有效减少无效递归。
                return
                
            i -= 1
            j += 1
            move_check(input, i, j)
            
        index = 0
        while index < len(list(s)):
            move_check(list(s), index, index) #奇数
            move_check(list(s), index - 1, index) #偶数
            index += 1
        return self.count
```
AC了，看了个大神的方法，用了for嵌套while的结构，看着还是挺清楚的，我承认我之前非要写个递归，其实就是想自己练练写递归，也没什么特别的道理哈哈。这里用了一个技巧来一步实现奇数偶数的判断，left = center / 2, right = center / 2 + center % 2. 实现如下：
```python
class Solution:
    def countSubstrings(self, S):
        N = len(S)
        ans = 0
        for center in range(2*N - 1):
            left = center // 2
            right = left + center % 2
            while left >= 0 and right < N and S[left] == S[right]:
                ans += 1
                left -= 1
                right += 1
        return ans
```
## Leetcode-230 Kth Smallest Element in a BST BST树中第k个最小值
> Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.
Note: 
You may assume k is always valid, 1 ≤ k ≤ BST's total elements.

> Example 1:
Input: root = [3,1,4,null,2], k = 1. Output: 1
   3
  / \
 1   4
  \
   2

最近翻看了很多树类的题目，凡是见到树的题目，脑子就要有一个递归的大体框架，如果递归比较弱的，比如我自己，就可以有针对性的练几个树的题目提高一下。
这一题，乍一看也挺简单的，直觉上利用BST树的基本性质，左小右大，于是写了一个遍历左子树的递归，但是报错，这才发现如果K值过大的话，还是需要右子树的，会觉得需要对K做分情况考虑了一下子好像觉得有点复杂了。而像这种第几个XXX的题目，排序似乎永远都是一个不错的方案，全局排序会导致较大的时间开销，设置条件提前停止是一个比较不错的优化方案。

由于每一棵树都是左>根>右，因此采用中序遍历（LDR）代码如下：
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        vals = [] # 初始化一个数组
        def ldr(node):
            if node is None:
                return 
            dfs(node.left)
            vals.append(node.val)
            dfs(node.right)
        ldr(root)
        return vals[k - 1]        
```

为了训练，也可以用非递归的形式来解，但是这边需要，我这里练习并记录一下
```python
class Solution:
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        stack = []  # stack以栈来放置node，到达放入，结束后弹出
        node = root # node用来表示当前的节点
        res = []    # 存放排序结果
        while node or stack:
            while node:
                stack.append(node) # 插入当前节点
                node = node.left
            node = stack.pop() # 弹出最底层的节点
            res.append(node.val)
            node = node.right
        return res[k - 1]
```

这里的两个方法，其实都只是直接考虑全局排序的，时间消耗是很大的，从AC后的耗时可以看出，要优化的话，就要从这个k入手，记录并精准返回第k个然后程序结束。写一下：
```python
class Solution:
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        self.k = k
        self.res = None
        self.helper(root)
        return self.res

    def helper(self, node):
        if not node:
            return
        self.helper(node.left)
        self.k -= 1 # 计数并精准返回
        if self.k == 0:
            self.res = node.val
            return
        self.helper(node.right)
```
## Leetcode-69 Sqrt(x) 开根号
> Implement int sqrt(int x).
Compute and return the square root of x, where x is guaranteed to be a non-negative integer.
Since the return type is an integer, the decimal digits are truncated and only the integer part of the result is returned.

> Example 1:
Input: 4
Output: 2

一道简单题，python可以导入math，或者用x**都可以得到结果，但是这不是目的。这是一题二分查找的题，考虑了一些边界条件和结束条件，就写了如下的一个方法，有点繁杂但是可以AC：
```python
class Solution:
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x == 0:
            return 0
        
        def calculate(i, j, x):
            if i * i == x:
                return i
            if j * j == x:
                return j
            if j - i <= 1:
                return i
            mid = (i + j) // 2
            if mid * mid > x:
                return calculate(i, mid, x)
            else:
                return calculate(mid, j, x)
        return calculate(1, x, x)
```
我是故意写递归的，目的是让自己习惯。下面照着大神的做法，理解了写了个别的方法
```python
class Solution:
    def mySqrt(self, x):
        l, r = 0, x
        while l <= r:
            mid = l + (r-l)//2
            if mid * mid <= x < (mid+1)*(mid+1):
                return mid
            elif x < mid * mid:
                r = mid
            else:
                l = mid + 1
```
这里巧妙的在于，初始是0也就不需要额外判断了，然后把我上面的三个条件并成了一个，如果x在mid\*mid和(mid+1) \*（mid+1）之间即输出。不过这个题，只要想到二分查找就至少答到题意了，有些优化可能第一时间想不到也没关系。就这样吧。

## Leetcode-226 Invert Binary Tree 翻转二叉树
> Invert a binary tree.
Example:
Input:
     4
   /   \
  2     7
 / \   / \
1   3 6   9

> Output:
     4
   /   \
  7     2
 / \   / \
9   6 3   1

这个题来源于一个小故事，Google: 90% of our engineers use the software you wrote (Homebrew), but you can’t invert a binary tree on a whiteboard so f*** off. 哈哈哈。这个题也是一个典型树类的递归应用，直接交换左右子树位置就好了。代码如下：
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return None
        # tmp = root.left
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
```
还可以进一步简化为这样子：
```python
class Solution:
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if root:
          invert = self.invertTree
          root.left, root.right = invert(root.right), invert(root.left)
          return root
```
反正也是一个意思吧。
总体来说，树类的问题套路一般就是直上直下的遍历，采用递归或者while的循环都可以，遍历的3种模式中，前序遍历，中序遍历可以相互转化，后序遍历和层次遍历（BFS）可以一起记忆。另一类就是DFS，需要维护和记录一个visited数组。
大概就是这样吧，学的比较慢，下周继续。
