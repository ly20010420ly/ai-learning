print('=' * 50)
print("列表推导式")
print("=" *50)
#1-10的平方
squares = [x**2 for x in range(1,11)]
print(squares)
#过滤偶数
evens = [x for x in range(20) if x%2==0]
print(evens)
#循环联系
pairs = [(x,y) for x in range(2) for y in range(3)]
print(f'坐标对{pairs}')

print('=' * 50)
print("字典和集合推导式式")
print("=" *50)
#字典推导式
square_dict = {x:x**2 for x in range(5)}  #键和值
print(f'数字平方字典{square_dict}')
#集合推导式
unique_dict = {char for char in "hello world" if char != ' '}
print(f"唯一字符集合：{unique_dict}")

print('=' * 50)
print("lambda和map/filter")
print("=" *50)
#lambda
add = lambda x,y:x+y
print(f'lambda加法{add(3,5)}')
#map
number = [1,2,3,4,5]
double = list(map(lambda x:x*2,number))
print(f"map加倍{double}")
#filter
num = range(10)
even = list(filter(lambda x: x%2 == 0,num))
print(f'filter筛选偶数{even}')

print('=' * 50)
print("zip和enumerate")
print("=" *50)
#zip
names = ['A','B','C']
age = [10,20,30]
per = list(zip(names,age))
print(per)
#enumerate
for index,name in enumerate(names,start=0):
    print(f'索引对应的元素{(index,name)}')

print('=' * 50)
print("文件读写")
print("=" *50)
with open('test.txt', 'w', encoding='utf-8') as f:
    f.write("Hello Python\n")
    f.write("这是第二行内容")
with open('test.txt', 'r', encoding='utf-8') as f:
    content = f.read()
print(f"文件内容：{content}")

print('=' * 50)
print("异常处理")
print("=" *50)
try:
    result = 10 / 0
except ZeroDivisionError:
    print("捕获到除零错误")
finally:
    print("无论正常错误都会执行")

print('=' * 50)
print("类和对象")
print("=" *50)
class Student():
    def __init__(self,name,garde):
        self.name  = name
        self.garde = garde
    def interduce(self):
        return f"我是{self.name}，成绩为{self.garde}"
    def improve(self,points):
        self.garde += points
        return f"我是{self.name}，成绩提高了{points}，现在是{self.garde}"
stu1 = Student('小明',85)
print(stu1.interduce())
print(stu1.improve(5))

print('=' * 50)
print("测试结束")