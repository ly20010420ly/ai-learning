import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

print('=' * 50)
print("环境检测报告")
print('=' * 50)

#numpy测试
arr = np.array([1,2,3])
arr1 = np.array([[1,2,3],[4,5,6]])
print(f"numpy版本:{np.__version__}")
print(f"Numpy数组;{arr}")
print(f"Numpy广播：{arr1+arr}")

#torch测试
x = torch.tensor([1.0,2.0,3.0])
print(f"\ntorch版本：{torch.__version__}")
print(f"torch张量;{x}")
print(f"GPU可用：{torch.cuda.is_available()}")

#pandas测试
df = pd.DataFrame({'A':[1,2,3],'B':[4,5,6]})
print(f"\nPandas版本：{pd.__version__}")
print(f"DataFrame:\n{df}")

print("\n环境配置成功")