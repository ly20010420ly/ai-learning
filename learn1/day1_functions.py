# day1_functions.py - 5个经典函数
import math


# 函数1：斐波那契数列
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    result = [0,1]
    for i in range(2,n):
        result.append(result[-1]+result[-2])
    return result




# 函数2：素数判断
def is_prime(num):
    if num < 2:
        return False
    for i in range(2,int(math.sqrt(num)+1),2):
        if num % i == 0:
            return False
    return True


# 函数3：回文字符串
def is_palindrome(s):
    s = s.replace(" ","").lower()
    return s == s[::-1]



# 函数4：字符频率统计
def char_frequency(s):
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch,0)+1
    return freq

# 函数5：二分查找
def binary_search(arr, target):
    left,right = 0,len(arr)-1
    while left<=right:
        mid = (left+right)//2
        if arr[mid] == target:
            return mid
        elif arr[mid] > target:
            mid -= 1
        else:
            mid += 1
    return -1


# ========== 测试代码（不要修改） ==========
if __name__ == "__main__":
    print("测试斐波那契数列:")
    print(f"fibonacci(5) = {fibonacci(5)}  # 期望: [0,1,1,2,3]")

    print("\n测试素数判断:")
    print(f"is_prime(7) = {is_prime(7)}    # 期望: True")
    print(f"is_prime(10) = {is_prime(10)}  # 期望: False")

    print("\n测试回文字符串:")
    print(f"is_palindrome('racecar') = {is_palindrome('racecar')}  # 期望: True")

    print("\n测试字符频率:")
    print(f"char_frequency('hello') = {char_frequency('hello')}  # 期望: {{'h':1, 'e':1, 'l':2, 'o':1}}")

    print("\n测试二分查找:")
    print(f"binary_search([1,3,5,7,9], 5) = {binary_search([1, 3, 5, 7, 9], 5)}  # 期望: 2")

