# -*- encoding:utf-8 -*-

import os
import re
import json
import math
import fileinput
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from collections import Counter

from utils import *


def statistic_phenos_correlation(phenosfile='../data/multi_phenos.txt'):
	'''
		统计症状相关性
	'''
	from scipy.stats import pearsonr
	from sklearn.metrics import normalized_mutual_info_score
	phenos = np.array([[int(i) for i in line.split()] for line in open(phenosfile,'r').read().split(u'\n')[1:-1]])
	mutual_info = np.zeros((10,10)); pearsonr_sim = np.zeros((10,10))
	for i in range(10):
		for j in range(i,10):
			mutual_info[i,j] = mutual_info[j,i] = normalized_mutual_info_score(phenos[:,i],phenos[:,j])
			pearsonr_sim[i,j] = pearsonr_sim[j,i] = pearsonr(phenos[:,i],phenos[:,j])[0]
	print mutual_info; print pearsonr_sim


def multiphenos_learning(Ninstance=1000, Nsnp=9445, Nfeature=40, Nphenos=10, p_threshold=0.05):
	'''
		多症状学习
	'''
	import pandas as pd
	from scipy.stats import ranksums
	from sklearn.cross_validation import KFold
	from sklearn.metrics import accuracy_score
	from sklearn.preprocessing import OneHotEncoder
	from sklearn.ensemble import RandomForestClassifier

	genos = pd.read_table('../data/genotype_encode.dat', sep=' ')
	labels = pd.read_table('../data/multi_phenos.txt', sep=' ')

	for i in xrange(1, Nphenos+1): genos['L' + str(i)] = labels['L' + str(i)]
	x = genos; result, count = [], 0
	for i in xrange(1, Nphenos+1):
		result.append([])
		x0 = x[x['L' + str(i)]==0]
		x1 = x[x['L' + str(i)]==1]
		for j in xrange(0, Nsnp):
			pvalue = ranksums(x0.iloc[:,j], x1.iloc[:,j]).pvalue
			result[count].append(pvalue)
		count += 1
	result = np.array(result)

	p_select = [[] for i in xrange(result.shape[0])]
	for i in xrange(result.shape[0]):
		for j in xrange(result.shape[1]):
			if result[i,j] < p_threshold:
				p_select[i].append(j)
	p = set(p_select[0])
	for i in xrange(1, Nphenos):
		p = p.union(set(p_select[i]))
	p = sorted(list(p))
	
	value = [0 for i in xrange(len(p))]
	for i in xrange(len(p)):
		for j in xrange(len(p_select)):
			if p[i] in p_select[j]: value[i] += 1
	kv_pair = sorted([(value[i], p[i]) for i in xrange(len(p))], reverse=True)
	p = [kv_pair[i][0] for i in xrange(len(kv_pair))]
	
	plt.plot([i for i in xrange(len(p))], p)
	plt.xlabel('Genotype')
	plt.ylabel('Frequency')
	plt.savefig('../figures/multiphenos_genotype_select_frequency.png')

	# 选择位点
	p = [kv_pair[i][1] for i in xrange(200)]
	p_dict = {kv_pair[i][1]:kv_pair[i][0] for i in xrange(len(kv_pair))}
	g_features = genos.iloc[:,:Nsnp]; g_labels = genos.iloc[:,Nsnp:Nsnp+Nphenos]
	X = np.array([list(g_features.iloc[i,p]) for i in xrange(genos.shape[0])])
	X = OneHotEncoder().fit_transform(X).toarray()
	Y = np.array([list(g_labels.iloc[i,:]) for i in xrange(genos.shape[0])])
	
	RF = RandomForestClassifier(n_estimators=1600).fit(X,Y)
	feature_weight = RF.feature_importances_
	feature_kv = [(feature_weight[i], i) for i in xrange(len(feature_weight))]
	select_features = sorted(feature_kv, reverse=True)[:Nfeature*3]
	select_index = [select_features[i][1] for i in xrange(len(select_features))]
	select_genos = np.array(p)[[select_index[i]/3 for i in xrange(len(select_index))]]
	final_genos = []
	for i in xrange(len(select_genos)):
		if len(final_genos) == Nfeature: break
		if select_genos[i] not in final_genos:
			final_genos.append(select_genos[i])
	
	# 40个位点的名称
	print '位点名称：'
	print list(g_features.columns[final_genos])
	
	# 40个位点的Importance
	print '位点Importance：'
	print [p_dict[final_genos[i]] for i in xrange(len(final_genos))]

	# 融合模型准确率评估
	kf = KFold(Ninstance, n_folds=10)
	cv_result = []; X = X[:,select_index]
	for train_index, test_index in kf:
		RF = RandomForestClassifier(n_estimators=1600).fit(X[train_index],Y[train_index])
		y_pred = RF.predict(X[test_index]); y_true = Y[test_index]
		accu = np.mean([accuracy_score(y_true[:,i], y_pred[:,i]) for i in xrange(y_true.shape[1])])
		cv_result.append(accu)
	print '准确率统计：'
	print np.mean(cv_result)


def multiphenos_validation(Ninstance=1000, Nsnp=9445, Nfeature=40, Nphenos=10, p_threshold=0.05):
	'''
		多症状评估
	'''
	import pandas as pd
	from scipy.stats import ranksums
	from sklearn.cross_validation import KFold
	from sklearn.metrics import accuracy_score
	from sklearn.preprocessing import OneHotEncoder
	from sklearn.ensemble import RandomForestClassifier

	genos = pd.read_table('../data/genotype_encode.dat', sep=' ')
	labels = pd.read_table('../data/multi_phenos.txt', sep=' ')

	for i in xrange(1, Nphenos+1): genos['L' + str(i)] = labels['L' + str(i)]
	x = genos; result, count = [], 0
	for i in xrange(1, Nphenos+1):
		result.append([])
		x0 = x[x['L' + str(i)]==0]
		x1 = x[x['L' + str(i)]==1]
		for j in xrange(Nsnp):
			pvalue = ranksums(x0.iloc[:,j], x1.iloc[:,j]).pvalue
			result[count].append(pvalue)
		count += 1
	result = np.array(result)

	p_select = [[] for i in xrange(result.shape[0])]
	for i in xrange(result.shape[0]):
		for j in xrange(result.shape[1]):
			if result[i,j] < p_threshold:
				p_select[i].append(j)

	# 对比实验
	all_result = []
	for k in xrange(Nphenos):
		# 选择位点
		g_features = genos.iloc[:,p_select[k]]
		g_labels = genos.iloc[:,Nsnp+k]
		X = np.array([list(g_features.iloc[i,:]) for i in xrange(genos.shape[0])])
		X = OneHotEncoder().fit_transform(X).toarray()
		Y = np.array(g_labels)

		RF = RandomForestClassifier(n_estimators=1600).fit(X,Y)
		feature_weight = RF.feature_importances_
		feature_kv = [(feature_weight[i], i) for i in xrange(len(feature_weight))]
		select_features = sorted(feature_kv, reverse=True)[:Nfeature*3]
		select_index = [select_features[i][1] for i in xrange(len(select_features))]
		X = X[:,select_index]

		# 子模型准确率评估
		total_result = []
		for i in xrange(Nphenos):
			Y = np.array(genos.iloc[:,Nsnp+i])
			kf = KFold(Ninstance, n_folds=10)
			cv_result = []
			for train_index, test_index in kf:
				RF = RandomForestClassifier(n_estimators=1600).fit(X[train_index],Y[train_index])
				y_pred = RF.predict(X[test_index]); y_true = Y[test_index]
				accu = accuracy_score(y_true, y_pred)
				cv_result.append(accu)
			total_result.append(np.mean(cv_result))
		all_result.append(np.mean(total_result))
	print '准确率统计：'
	print all_result


if __name__ == '__main__':
	#### 统计症状相关性 ####
	# statistic_phenos_correlation()

	#### 􏱧􏱩􏱨􏰞􏱫􏱧􏱩􏱨􏰞􏱫􏱧􏱩􏱨􏰞􏱫多症状学习 ####
	# multiphenos_learning()

	#### 􏱧􏱩􏱨􏰞􏱫􏱧􏱩􏱨􏰞􏱫􏱧􏱩􏱨􏰞􏱫多症状评估 ####
	# multiphenos_validation()

	pass
