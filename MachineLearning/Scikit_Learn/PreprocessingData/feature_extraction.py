import numpy as np
import sklearn.feature_extraction as sfe

'''
所谓Feature Extraction, 字面意思是提取特征。是将各种非数值形式的数据集，比如字典、文本、图像等，
转化成数值型的数据集，以供各种机器学习的工具能处理和分析它们。

DictVectorizer: 用于将标准的python dict转化为数值型矩阵，对于无序的分类型的key-pair，其处理方式与
    OneHotEncoder一样。也是采取形如s0, 0, 0, 0, 1这种形式来编码，每种分类只有一个值为1. 但两者又有
    不同之处，前者只对string类型的变量采取OneHotEncoding，而对于后者来说，对数值型分类变量也可以
'''

measurements = [{'city': 'Dubai', 'temperature': 33.},
                {'city': 'London', 'temperature': 12.},
                {'city': 'San Fransisco', 'temperature': 18.}]

vec = sfe.DictVectorizer()
print('measurements: \n', measurements)
print('measurements after feature_extraction: \n', vec.fit_transform(measurements))
print('all the feature names: ', vec.feature_names_) 
print('all the vocabulary: ', vec.vocabulary_)
