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
其中按键对应的字母映射要自己建一个dictionary，汗。。。如下：
(letters = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', 
                   '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'})
Input: "23"
Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].

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

