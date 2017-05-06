import numpy as np

'''
np.select(condlist, choicelist, default=0): 三部曲，
    1. 依据condlist选取满足条件的元素
    2. 对于第一步中取出来的元素，依据choicelist做对应的输出变换
    3. 当所有条件都不满足时，返回default值
Parameters:
    1. condlist: bool ndarrays, 包含了一组bool类型的条件
    2. choicelist: 与condlist的元素一一对应，每个条件都有对应的choice，长度必须一致
    3. default: 与前两个参数不同，它是个标量，只有当所有条件都不满足的时候，返回default指定的值

np.extract(condition, array):
    提取array中满足condition的所有元素，压缩成一个一维数组返回
'''

x = np.arange(10)
condlist = [x<3, x>5]
choicelist = [x, x**2]
print('x: ', x)
print('condlist: ', condlist)
print('choicelist: ', choicelist)
# 小于3的元素按原值输出，大于5的元素取平方输出，其余的都用0来填充
print('select from x: ', np.select(condlist, choicelist, default=np.NaN))

# 对所有不满足任何条件的元素，都用default指定的值来填充，默认为0
condlist = [x>10]
print('select from x with no elements match condlist: ', np.select(condlist, [x], default=np.NaN))

arr = np.arange(12).reshape((3, 4))
condition = np.mod(arr, 3)==0
print('array is: \n', arr)
print('condition (np.mod(arr, 3) == 0): \n', condition)
print('extract from array with condition: ', np.extract(condition, arr))

# 该方法特别适合用于按照某种条件分割一维数组
arr = np.arange(13)
condition = [arr > 6]
print('array is: \n', arr)
print('condition (np.mod(arr, 3) == 0): \n', condition)
print('extract from array with condition: ', np.extract(condition, arr))
