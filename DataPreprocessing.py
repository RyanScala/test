# -*- coding: utf-8 -*-
"""
@author: ryanfan
"""

import time
import warnings
import pandas as pd
import numpy as np
from numpy import nan as NA
from sklearn.preprocessing import LabelEncoder


"""
Ignore all warning info.
"""
def ignoreWarning():
	warnings.filterwarnings("ignore")


"""
1、判断特征的类型，连续型 or 离散型
"""
def featureType(dataFrame):
	# 存放 “连续型特征” 和 “离散型特征” 名称的数据盒子
	conFeatureList = []
	disFeatureList = []

	# 判断每个字段的数据类型
	allType = dataFrame.dtypes

	for index in xrange(len(allType)):
		if allType.values[index] in ['object','string_','unicode_']:
			disFeatureList.append(allType.index[index])
		else:
			conFeatureList.append(allType.index[index])

	# 返回的是 特征的名称。
	return conFeatureList, disFeatureList

"""
2、定义一个批量转为百分比的func
"""
def trim(x):
	return '%.4f%%' % (x * 100)

"""
3、求出TOP N类别的记录数 占 总记录数的比例，并画出图形。
a. 前TOP N的 class 包含的记录数 占 总记录数的比例；
b. 输出不大于TOP 10 的情况；
c. 画出完整的分布情况；

----------------------------------------------------------------------------------------------------------------
当TOP N 共包含的记录数 > Threshold(至少70%，最好90%以上)，则做以下转换：将 N 个类别重新归类为：A_1,...,A_N,其它;
为什么这么做：
1.一般无法确定是一定没作用，除非35个特征把数据集绝对均分。
2.有的时候有微小作用，把它做组合特征之后就有用了。
3.对于删特征，我理解是出于资源问题才会删，对模型本身训练没什么影响。
----------------------------------------------------------------------------------------------------------------
"""
def featureAnalyse(dataSet, disFeatureList):
	print '*' * 80
	for disFeatureName in disFeatureList:
		# 选定当前的feature
		disFeature = dataSet[disFeatureName]

		# Look at categories of all object variables
		rawData = disFeature.value_counts()  # The resulting object will be in descending order

		# 1、计算每一个category所占比例
		catProportion = rawData / float(disFeature.shape[0])

		# 转换为Series
		catProportion = pd.Series(catProportion, index = rawData.index)

		# 2、计算累加比例
		cumProportion = catProportion.values.cumsum()

		# 转换为Series
		cumProportion = pd.Series(cumProportion, index = rawData.index)

		# 3、计算累加比例的增幅
		ampProportion = []

		for i in xrange(len(cumProportion)):
			if i == 0:
				ampProportion.append('-')
			else:
				amplitude = (cumProportion[i] - cumProportion[i-1]) / float(cumProportion[i-1])
				ampProportion.append('%.2f%%' % (amplitude* 100))


		# 合并成一个DataFrame
		dataFrame = pd.DataFrame({'valuesCounts' :rawData,
								  'catProportion':catProportion.apply(trim), 
								  'cumProportion':cumProportion.apply(trim), 
								  'ampProportion':np.array(ampProportion)},
								  index = rawData.index
								)

		# 重新排序
		dataFrame = dataFrame[['valuesCounts','catProportion','cumProportion','ampProportion']]

		print '-' * 80
		print 'Top 10 rows of feature analyse result for variable [%s]'%disFeatureName
		print ' '
		print dataFrame.head(10)

	print '-' * 80

"""
4、判断缺失值情况
"""
def checkMissing(dataSet):
	# 统计缺失值的个数
	missingCnt = dataSet.apply(lambda x: sum(x.isnull()))

	# 统计缺失占比，值为小数
	missingPer = dataSet.apply(lambda x: sum(x.isnull()) / float(dataSet.shape[0]))

	# 合并成一个DataFrame
	dataFrame = pd.DataFrame({'missingCounts'    :missingCnt,
							  'missingPercentage':missingPer.apply(trim),
							  'featureType':dataSet.dtypes}, 							  
							   index = missingPer.index
							)

	# 重新排序
	dataFrame = dataFrame[['missingCounts','missingPercentage','featureType']]

	print '*' * 80
	print 'Check missing value for dataSet'
	print ' '
	print dataFrame
	print '-' * 80

	return dataFrame, missingPer

"""
5、转换特征：
a. DOB -> Age
b. Treat missing value appropriately：
"""
"""
b1) If missing proportion is high(>30%), we will create a new var to represent whether present or not. 0-> not missing，1->missing
"""
# 判断哪些feature需要做(b1)转换，默认阈值是0.3
def needStamp(missingPer, threshold):
	# 需要做 stampMissing 转换的 feature
	stampList = []
	valueList = []

	for index in xrange(len(missingPer.values)):
		if missingPer.values[index] >= threshold:
			stampList.append(missingPer.index[index])
			valueList.append(missingPer.values[index])
		else:
			pass

	# 构建一个dataFrame
	dataFrame = pd.DataFrame({'missingProportion':np.array(valueList)},
							  index = np.array(stampList)
							)


	print '*' * 80
	print 'If missing proportion is higher than %s,'%(str(threshold))
	print 'we will create a new feature to represent whether present or not.'
	print ' '
	print dataFrame
	print '-' * 80

	return stampList

# 根据needStamp func 的输出，做stampMissing的转换
def stampMissing(dataSet, featureList):
	print '*' * 80
	print 'Creat a new feature when missing proportion is high!~'
	print ' '

	for featureName in featureList:
		# 定义新的特征名称
		newFeature = featureName + '_Missing'

		# 给数据集添加新的特征
		dataSet[newFeature] = dataSet[featureName].apply(lambda x: 1 if pd.isnull(x) else 0)

		# 删除旧的特征
		dataSet.drop([featureName], axis = 1, inplace = True)
		print 'Have dropped old feature: *%s*' %(featureName)
		print 'Have added   new feature: [%s]' %(newFeature)
		print ' '

	print '-' * 80

	return dataSet

"""
b2) Impute with median or mean when proportion missing is low.
"""
# 使用中位数或者平均值来填充缺失值，默认是用中位数。（这部分仍需完善，毕竟缺失值的填充是一个很大的方向。）
def fillMissing(dataSet, featureNameList, fillType):
	# 默认是用 中位数 填充缺失值
	if fillType == 'median':
		# 用 中位数 填充缺失值
		dataSet[featureNameList] = dataSet[featureNameList].fillna(dataSet[featureNameList].median(), inplace = True)
	else:
		# 用 平均数 填充缺失值
		dataSet[featureNameList] = dataSet[featureNameList].fillna(dataSet[featureNameList].mean(), inplace = True)

	return dataSet

"""
c) 用关键的20%来重塑feature：
某一个离散型特征，含有很多的类别，但很多的记录都来自某几个类别，记为A_1,...A_N。
则，将该feature的类别数，由原来的M压缩为N+1，其中M远大于N+1
"""
def remoldFeature(dataSet, disFeatureNameList, majorClassList):
	# 这里必须用loop，是因为不同的feature有不同的majorClassList。
	for index in xrange(len(disFeatureNameList)):
		dataSet[disFeatureNameList[index]] = dataSet[disFeatureNameList[index]].apply(lambda x: 'others' if x not in majorClassList[index] else x)

	return dataSet

"""
d) 考虑到服务器扛不住，可以删除一些很有可能对label贡献不大的feature，it is always discrete and has too many unique values.
"""
def dropFeature(dataSet, delFeatureNameList):
	# 设置'inplace'为True，原DataFrame直接就被替换，即对应的内存值直接改变。
	dataSet.drop(delFeatureNameList, axis = 1, inplace = True)

	return dataSet


"""
7、One-Hot Coding using fit_transform() function and get_dummies() function.
"""
def one_hot(formerSet, dataSet, disFeatureList):
	# one-hot 之前的数据集维度
	formerShape = str(formerSet.shape)

	# convert labels into numbers, including Nan.
	for feature in disFeatureList:
		dataSet[feature] = LabelEncoder().fit_transform(dataSet[feature])

	# Use pd.get_dummies function to do One-Hot Coding.
	dataSet = pd.get_dummies(dataSet, columns = disFeatureList)

	print '*' * 80
	print 'Finally, we have got following features:'
	print ' '
	print dataSet.columns
	print ' '
	print "Shape of dataset changed from *{shape_1}* to *{shape_2}*".format(shape_1 = formerShape, shape_2 = str((dataSet.shape)))
	print '-' * 80

	return dataSet


# 主函数
def main():
	# 构建计时器
	start_time = time.time()

	# 不显示任何warning info
	ignoreWarning()

	"""1、数据集准备"""
	# Load data
	trainSet = pd.read_csv('C:/Users/Ryan Fan/Desktop/DataMining Learning/Scripts/Dataset/Train.csv', low_memory=False)
	testSet  = pd.read_csv('C:/Users/Ryan Fan/Desktop/DataMining Learning/Scripts/Dataset/Test.csv', low_memory=False)

	# Combine into data
	trainSet['sourceTag']= 'train'
	testSet['sourceTag'] = 'test'

	# 去掉因变量有缺失的记录
	trainSet = trainSet[trainSet['Disbursed'].notnull()]
	# testSet = testSet[testSet['Disbursed'].notnull()]
	
	# 因为这里的测试集中没有因变量，故不做合并。
	# dataSet = pd.concat([trainSet, testSet], ignore_index = True)
	dataSet = trainSet


	"""2、判断feature的类型：离散 or 连续"""
	conFeatureList, disFeatureList = featureType(dataSet)


	"""3、分析离散型变量：共有多少个类别，每个类别含有多少个记录。"""
	featureAnalyse(dataSet, disFeatureList)


	"""4、分析每个feature的缺失情况"""
	missingFrame, missingPer = checkMissing(dataSet)


	"""5.1 判断哪些feature需要做(b1)转换，默认阈值是0.3""" 
	threshold = 0.3
	stampList = needStamp(missingPer, threshold)


	"""5.2 做stampMissing的转换"""
	dataSet = stampMissing(dataSet, stampList)


	"""5.3 缺失值填充"""
	fillType = 'median'
	featureNameList = ['Existing_EMI', 'Loan_Amount_Applied', 'Loan_Tenure_Applied']
	dataSet = fillMissing(dataSet, featureNameList, fillType)


	"""5.4 做remoldFeature转换"""
	disFeatureNameList = ['Source'
						 ,'Var1'
						 ]

	majorClassList = [['S122','S133']
					 ,['HBXX','HBXC']
					 ]

	dataSet = remoldFeature(dataSet, disFeatureNameList, majorClassList)


	"""5.5 考虑服务器负荷，删除一些无用的feature"""
	# 设置要删特征列表
	delFeatureNameList = ['City', 'Employer_Name', 'Lead_Creation_Date', 'LoggedIn', 'Salary_Account']
	
	# Drop them
	dataSet = dropFeature(dataSet, delFeatureNameList)


	"""8、处理DOB特征"""
	# 新建一个 年龄 字段
	dataSet['Age'] = dataSet['DOB'].apply(lambda x: 2017 - int(x[:4]))  # 先观察DOB的格式，再选择做对应的运算规则

	# 删除旧的字段
	dataSet = dropFeature(dataSet, ['DOB'])


	"""10、进行One-Hot处理"""
	# 设置哪些feature需要做One-Hot处理
	disFeatureList = ['Device_Type','Filled_Form','Gender','Var1','Var2','Var5','Mobile_Verified','Source']
	dataSet = one_hot(trainSet, dataSet, disFeatureList)


	"""10、Separate train & test:"""
	train = dataSet.loc[dataSet['sourceTag'] == 'train']
	test  = dataSet.loc[dataSet['sourceTag'] == 'test']

	train.drop('sourceTag', axis = 1, inplace = True)
	test.drop(['sourceTag','Disbursed'], axis = 1, inplace = True)

	# 输出训练集和测试集
	train.to_csv('train_modified.csv', index = False)
	test.to_csv('test_modified.csv', index = False)


	# 输出运行时间
	timeUsed = (time.time() - start_time)

	if timeUsed > 60:
		print 'Total Time Used: {Time} min'.format(Time = str(timeUsed / 60.0)[0:4])
	else:
		print 'Total Time Used: {Time} s'.format(Time = str(timeUsed)[0:4])

if __name__ == '__main__':
	main()

