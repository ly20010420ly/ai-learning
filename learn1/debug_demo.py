# debug_demo.py - 调试练习

def calculate_average(grades):
    """计算平均分，可能出错"""
    total = 0
    for grade in grades:
        total += grade
    return total / len(grades)  # 如果grades为空会出错

def process_students(students_data):
    """处理学生数据"""
    results = []
    for student in students_data:
        name = student['name']
        grades = student['grades']
        avg = calculate_average(grades)
        results.append(f"{name}: {avg}")
    return results

# 测试数据
students = [
    {'name': '张三', 'grades': [85, 90, 78]},
    {'name': '李四', 'grades': [92, 88, 84]},
    {'name': '王五', 'grades': []},  # 这个会出问题
]

# 运行
result = process_students(students)
print(result)