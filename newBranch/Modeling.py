# -*- coding: utf-8 -*-
"""
@author: ryanfan
"""
import sys
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from pyfiglet import Figlet
from matplotlib.pylab import rcParams
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn import cross_validation, metrics  
from sklearn.grid_search import GridSearchCV  
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


rcParams['figure.figsize'] = 12, 4

train = pd.read_csv('train_modified.csv')
target = 'Disbursed'
IDcol = 'ID'

# 使用传统的GBDT拟合dataSet
def GDBTfit(alg, dtrain, predictors, performCV = True, printFeatureImportance = True, cv_folds = 5):
	# 构建计时器
	start_time = time.time()

	# transform format of dataSet: change dataform into array.
	train_x = dtrain[predictors].values
	train_y = dtrain['Disbursed'].values

	# Fit the GBDT algorithm on the data
	fittedGBDT = alg.fit(train_x, train_y)

	# Predict training set by GBDT
	dtrain_predictions = alg.predict(train_x)  # return class
	dtrain_predprob = alg.predict_proba(train_x)[:,1] # 只返回正例的概率。默认是返回两列，第一列是负例的概率，第二列是正例的概率。

	#Perform cross-validation:
	if performCV:
	    cv_score = cross_validation.cross_val_score(alg, train_x, train_y, cv = cv_folds, scoring = 'roc_auc')
	    
	    """
	    cv : int, To determines the cross-validation splitting strategy. 
	    Possible inputs for cv are:
		- None, to use the default 3-fold cross-validation,
		- integer, to specify the number of folds.
		- An object to be used as a cross-validation generator.
		- An iterable yielding train/test splits.
		
		For integer/None inputs, if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold(分层) is used. 
		In all other cases, KFold is used.
	    """
	    """
	    scoring 的默认取值以及对应的函数名称
		- Classification 
		1) 'accuracy'：sklearn.metrics.accuracy_score
		2) 'average_precision'：sklearn.metrics.average_precision_score
		3) 'f1'：F-measure
		4) 'precision'：sklearn.metrics.precision_score
		5) 'recall'：sklearn.metrics.recall_score
		6) 'roc_auc'：sklearn.metrics.roc_auc_score
		
		- Clustering 
		1) 'adjusted_rand_score'：sklearn.metrics.adjusted_rand_score
		
		- Regression
		1) 'mean_squared_error'：sklearn.metrics.mean_squared_error
		2) 'r2'：sklearn.metrics.r2_score
	    """

	# Print model report:
	print "GBDT Model Report"
	print "Accuracy : %.4g" % metrics.accuracy_score(train_y, dtrain_predictions)
	print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)

	print '*' * 70

	if performCV:
		print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score)
																				,np.std(cv_score)
																				,np.min(cv_score)
																				,np.max(cv_score)
																				)
		
	# Print Feature Importance:
	if printFeatureImportance:
		feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
		feat_imp.plot(kind='bar', title='Feature Importances')
		plt.ylabel('Feature Importance Score')


	# 输出运行时间
	timeUsed = (time.time() - start_time)

	print '*' * 70

	if timeUsed > 60:
		print 'GBDT Time Used: {Time} min'.format(Time = str(timeUsed / 60.0)[0:4])
	else:
		print 'GBDT Time Used: {Time} s'.format(Time = str(timeUsed)[0:4])


	return fittedGBDT

# 使用GBDT+LR拟合dataSet
def GDBT_LR_fit(fittedGBDT, dtrain, predictors, topTrees = None,performCV = True, cv_folds = 5):
	# 构建计时器
	start_time = time.time()

	# transform format of dataSet: change dataform into array.
	train_x = dtrain[predictors].values
	train_y = dtrain['Disbursed'].values

	# One-hot
	grd_enc = OneHotEncoder()
	"""
	- 使用[:, 0:topTrees, 0] 控制只取前30棵树的特征组合
	- 使用[:, :, 0] 则取全部树产生的特征组合
	"""
	if topTrees is None:
		print '*' * 70
		print 'Use transformed features from GBDT of *all* trees.' 
	else:
		print '*' * 70
		print 'Use transformed features from GBDT of Top {topN} trees.'.format(topN = topTrees)


	grd_enc.fit(fittedGBDT.apply(train_x)[:, 0:topTrees, 0])

	# Fit the GBDT +LR algorithm on the data
	grd_lm = LogisticRegression()
	grd_lm.fit(grd_enc.transform(fittedGBDT.apply(train_x)[:, 0:topTrees, 0]), train_y)

	# Predict training set by GBDT+LR
	dtrain_predictions_LR = grd_lm.predict(grd_enc.transform(fittedGBDT.apply(train_x)[:, 0:topTrees, 0]))  # return class
	dtrain_predprob_LR = grd_lm.predict_proba(grd_enc.transform(fittedGBDT.apply(train_x)[:, 0:topTrees, 0]))[:, 1]

	#Perform cross-validation:
	if performCV:
		cv_score_lr = cross_validation.cross_val_score( grd_lm, grd_enc.transform(fittedGBDT.apply(train_x)[:, 0:topTrees, 0]), 
														train_y, cv = cv_folds, scoring = 'roc_auc')

	print '*' * 70

	print "GBDT+LR Model Report"
	print "Accuracy : %.4g" % metrics.accuracy_score(train_y, dtrain_predictions_LR)
	print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob_LR)

	if performCV:
		print "CV_LR Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % ( np.mean(cv_score_lr)
																					,np.std(cv_score_lr)
																					,np.min(cv_score_lr)
																					,np.max(cv_score_lr)
																					)

	# 输出运行时间
	timeUsed = (time.time() - start_time)

	print '*' * 70

	if timeUsed > 60:
		print 'GBDT Time Used: {Time} min'.format(Time = str(timeUsed / 60.0)[0:4])
	else:
		print 'GBDT Time Used: {Time} s'.format(Time = str(timeUsed)[0:4])

# 构建feature set
predictors = [x for x in train.columns if x not in [target, IDcol]]

# 设置GBDT参数
traditionalGBDT = GradientBoostingClassifier( 
									  loss = 'deviance'    # 使用'deviance'，等价于和LR一样的Loss Function
									, n_estimators = 100   
									, learning_rate = 0.1   # 默认0.1学习速率/步长0.0-1.0的超参数  每个树学习前一个树的残差的步长
									, max_depth = 3   	    # 默认值为3每个回归树的深度  控制树的大小 也可用叶节点的数量max leaf nodes控制
									, subsample = 1  		# 树生成时对样本采样 选择子样本<1.0导致方差的减少和偏差的增加
									, min_samples_split = 2 # 生成子节点所需的最小样本数 如果是浮点数代表是百分比
									, min_samples_leaf = 1  # 叶节点所需的最小样本数  如果是浮点数代表是百分比
									, max_features = None
									, max_leaf_nodes = None     # 叶节点的数量 None不限数量
									, min_impurity_split = 1e-7 # 停止分裂叶子节点的阈值
									, verbose = 0    		    # 打印输出 大于1打印每棵树的进度和性能
									, warm_start = False    	# True在前面基础上增量训练(重设参数减少训练次数) False默认擦除重新训练
									, random_state = 24 		# 随机种子，方便重现
									)

timeNow = datetime.datetime.now()
	
# 转换时间戳的格式
yearPart  = timeNow.strftime("%Y") 
monthPart = timeNow.strftime("%m")
dayPart   = timeNow.strftime("%d")
hourPart  = timeNow.strftime("%H")
minPart   = timeNow.strftime("%M")
secPart   = timeNow.strftime("%S")

timeUnit = yearPart + '-' + monthPart + '-'+ dayPart

"""begin to train GDBT"""
fittedGBDT = GDBTfit(traditionalGBDT, train, predictors)

# 保存已经训练的模型
modelPath = 'C:/Users/Ryan Fan/Desktop/DataMining Learning/Scripts/trainedModels/'
modelName = 'traditionalGBDT_{stampTime}.pkl'.format(stampTime = timeUnit)

joblib.dump(fittedGBDT, modelPath + modelName) 

"""begin to train GDBT+LR"""
# 读取已经训练好的模型
fittedGBDT = joblib.load(modelPath + modelName) 

# 设置一些参数
topTrees = 50

# begin to fit
GDBT_LR_fit(fittedGBDT, train, predictors,topTrees)

# Mission End
Ryan_Spec = Figlet()
print (Ryan_Spec.renderText('Mission Succeed'))






