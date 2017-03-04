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
from correlation_gene import *


def display_person_distance(Nmin=100):
	'''
		检验个体独立
	'''
	x, y = load_x_y()
	pair_distance = [(i,j,((x[i]-x[j])**2).sum()**0.5) for i in xrange(len(x)-1) for j in xrange(i+1,len(x))]
	Nmin_distance = sorted(pair_distance,key=lambda x:x[2])[:Nmin]
	pair_distance = np.array([pair[2] for pair in pair_distance])
	print pair_distance.mean(), pair_distance.std(), pair_distance.max(), pair_distance.min()
	print Nmin_distance


def significance_test(p_threshold=0.05):
	'''
		位点显著检验
	'''
	from scipy.stats import chi2_contingency, fisher_exact
	x, y = load_x_y(); snp_dict = get_snp_dict()
	x0, x1 = x[:len(x)/2], x[len(x)/2:]

	def compute_OR(m):
		fes = 0
		for permutation in ((0,1),(0,2),(1,2)):
			fe = fisher_exact(m[:,permutation])[0]
			fes += fe if fe >= 1 else 1./fe
		return fes/3

	stats = []
	for i in xrange(len(x[0])):
		table = np.array([[Counter(xc[:,i]).get(0,0),Counter(xc[:,i]).get(1,0),Counter(xc[:,i]).get(2,0)] for xc in (x0,x1)])
		pvalue = chi2_contingency(table)[1]
		if pvalue < p_threshold:
			stats.append((i,pvalue,compute_OR(table)))
	print len(stats), [i for i,pv,OR in sorted(stats,key=lambda x:x[2])]
	with open('../data/important_snp_pvalue_OR.csv','w') as output:
		output.write('\n'.join(['{0},{1},{2}'.format(i,pv,OR) for i,pv,OR in stats]))


def feature_selection(p_threshold=0.05, Niteration=100, Nnoise=10, classifier='ExtraTrees', do_plot=False):
	'''
		加噪特征抽取
	'''
	from scipy.stats import chi2_contingency
	from sklearn.preprocessing import OneHotEncoder
	from sklearn.ensemble import ExtraTreesClassifier

	x, y = load_x_y(); snp_dict = get_snp_dict()
	n = np.random.permutation(len(x)); x, y = x[n], y[n]; x0, x1 = x[y==0], x[y==1]
	snp = [(i,chi2_contingency(np.array([[Counter(xc[:,i]).get(0,0),Counter(xc[:,i]).get(1,0),Counter(xc[:,i]).get(2,0)] \
				for xc in (x0,x1)]))[1]) for i in xrange(len(x[0]))]	
	snp_candidate, snp_noise = np.array([ind for ind,pv in snp if pv<p_threshold]), np.array([ind for ind,pv in snp if pv>=p_threshold])

	candidate_stat = {_snp:{'importance':0,'count':0} for _snp in snp_candidate}; candidate_importances = np.zeros(len(snp_candidate))
	for _iter in xrange(Niteration):
		print _iter
		snp_test = np.hstack([snp_candidate,snp_noise[np.random.permutation(len(snp_noise))][:Nnoise]])
		x_test = OneHotEncoder().fit_transform(x[:,snp_test]).toarray()

		if classifier == 'ExtraTrees':
			clf = ExtraTreesClassifier(n_estimators=100) # 可用其他分类器替换
		else:
			raise Exception('Classifer not supported!')

		clf = clf.fit(x_test, y)
		importances = np.reshape(clf.feature_importances_,(-1,3)).max(axis=1)
		selected = snp_test[np.where((importances > importances[-Nnoise:].max())[:len(snp_candidate)])]
		
		if do_plot:
			plt.figure(figsize=(6,5)); plt.ylim(0, len(importances)+20); plt.ylim(0, importances.max()*1.2)
			plt.plot(range(len(importances[:-Nnoise])), importances[:-Nnoise], 'kx')
			plt.plot(range(len(importances[:-Nnoise]),len(importances)), importances[-Nnoise:], 'bx')
			plt.plot([0,len(importances)+20], [importances[-Nnoise:].max()]*2, 'r--', alpha=0.5)
			plt.xlabel('SNP')
			plt.ylabel('Importance')
			plt.savefig('../figures/noiseimp.png')
			break

		candidate_importances += importances[:len(snp_candidate)]
		for snp_potential in selected: candidate_stat[snp_potential]['count'] += 1

	if not do_plot:
		for snp_potential, importance in zip(snp_candidate, candidate_importances):
			candidate_stat[snp_potential]['importance'] = importance
		with open('../data/important_snp_importance_noiseimp.csv','w') as outfile:
			outfile.write('\n'.join(['{0},{1:.10f},{2}'.format(snp, v['importance'], v['count']) for snp, v in sorted(candidate_stat.items(),key=lambda x:x[1]['importance'],reverse=True)]))


def snp_validation(Nfeature=40, classifier=''):
	'''
		位点相关评估
	'''
	from sklearn import svm
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.ensemble import AdaBoostClassifier
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.ensemble import GradientBoostingClassifier
	from sklearn.ensemble import ExtraTreesClassifier
	from sklearn.linear_model import Lasso
	from sklearn.linear_model import SGDClassifier
	from sklearn.linear_model import LogisticRegression
	from sklearn.naive_bayes import GaussianNB
	from sklearn.naive_bayes import BernoulliNB
	from sklearn.naive_bayes import MultinomialNB
	from sklearn.feature_selection import SelectKBest
	from sklearn.feature_selection import chi2, f_classif
	from sklearn.cross_validation import KFold
	from sklearn.preprocessing import OneHotEncoder

	x, y = load_x_y(); snp_dict = get_snp_dict()
	n = np.random.permutation(len(x)); x, y = x[n], y[n]
	snp_candidate = np.array([int(line.split(',')[0]) for line in open('../data/important_snp_pvalue_OR.csv','r').read().split('\n')])

	# forest = ExtraTreesClassifier(n_estimators=200, random_state=0)
	# forest.fit(x[:Ntrain], y[:Ntrain])
	# importances = forest.feature_importances_
	# indices = np.argsort(importances)[::-1][:10]
	# x = x[:,indices]

	# fs = SelectKBest(chi2, k=1000).fit(x, y)
	# x = fs.transform(x)

	if classifier == 'SVM_rbf':
		clf = svm.SVC(kernel='rbf')
	elif classifier == 'SVM_linear':
		clf = svm.SVC(kernel='linear')
	elif classifier == 'DecisionTree':
		clf = DecisionTreeClassifier()
	elif classifier == 'ExtraTrees':
		clf = ExtraTreesClassifier(n_estimators=2000, max_depth=10)
	elif classifier == 'RandomForest':
		clf = RandomForestClassifier(n_estimators=2000, max_depth=10)
	elif classifier == 'SGDClassifier_L1':
		clf = SGDClassifier(loss="hinge", penalty="l1")
	elif classifier == 'SGDClassifier_L2':
		clf = SGDClassifier(loss="hinge", penalty="l2")
	elif classifier == 'AdaBoost':
		clf = AdaBoostClassifier(n_estimators=2000)
	elif classifier == 'GradientBoosting':
		clf = GradientBoostingClassifier(n_estimators=2000, max_depth=10)
	elif classifier == 'BernoulliNB':
		clf = BernoulliNB()
	elif classifier == 'LogisticRegression_L1':
		clf = LogisticRegression(penalty="l1")
	elif classifier == 'LogisticRegression_L2':
		clf = LogisticRegression(penalty="l2")
	else:
		raise Exception('Classifer not supported!')

	x_train = x[:,snp_candidate[:Nfeature]]
	x_train = OneHotEncoder().fit_transform(x_train).toarray()

	TP = 0; FP = 0; FN = 0; correct = 0
	for train_index, test_index in KFold(len(x_train), n_folds=10):
		clf.fit(x_train[train_index],y[train_index])
		correct += (clf.predict(x_train[test_index])==y[test_index]).sum()
		TP += ((clf.predict(x_train[test_index])==1)*(y[test_index]==1)).sum()
		FP += ((clf.predict(x_train[test_index])!=1)*(y[test_index]==1)).sum()
		FN += ((clf.predict(x_train[test_index])==1)*(y[test_index]!=1)).sum()
		# print ((clf.predict(x_train[test_index]) == y[test_index]) * (y[test_index] == 0)).sum(), (y[test_index] == 0).sum()
		# print ((clf.predict(x_train[test_index]) == y[test_index]) * (y[test_index] == 1)).sum(), (y[test_index] == 1).sum()
	precision = 1.*TP/(TP+FP); recall = 1.*TP/(TP+FN); F = 2*precision*recall/(precision+recall)
	print precision, recall, F, 1.*correct/len(x_train)


def plot_snp_difference(ilist=[], Nsubplot=12):
	'''
		绘图位点分布
	'''
	mpl.rcParams['font.sans-serif'] = ['SimHei']
	mpl.rcParams['axes.unicode_minus'] = False
	x, y = load_x_y(); snp_dict = get_snp_dict()
	x0, x1 = x[:len(x)/2], x[len(x)/2:]
	fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12,12))
	index = np.arange(3); bar_width = 0.25; opacity = 0.4
	plt.setp(axes, xticks=list(index+bar_width), xticklabels=['0', '1', '2'])
	for ind, i in enumerate(ilist[:Nsubplot]):
		ax = axes.flat[ind]
		_x0 = np.array([count for _, count in sorted(Counter(x0[:,i]).items())])
		_x1 = np.array([count for _, count in sorted(Counter(x1[:,i]).items())])
		_xd = _x0-_x1
		bar1 = ax.bar(index, _x0, bar_width, alpha=opacity, color='g',label='X0')  
		bar2 = ax.bar(index+bar_width, _x1, bar_width, alpha=opacity, color='r', label='X1')
		bar3 = ax.bar(index+2*bar_width, _xd, bar_width, alpha=opacity, color='k', label='D')
		ax.set_xlabel(snp_dict.get(i))
		legend([bar1,bar2,bar3],[u'健康',u'患病',u'差值'])
	plt.tight_layout()
	plt.savefig('../figures/snp_difference.png')


def plot_snp_importance():
	'''
		绘图位点权重
	'''
	ind_dict = {}; snp_dict = get_snp_dict()
	for ind, pvalue, _ in [line.split(u',') for line in open('../data/important_snp_pvalue_OR.csv','r').read().decode('utf-8').split(u'\n')]:
		ind_dict[ind] = [pvalue]
	for ind, importance, noiseimp in [line.split(u',') for line in open('../data/important_snp_importance_noiseimp.csv','r').read().decode('utf-8').split(u'\n')]:
		ind_dict[ind] += [importance, noiseimp]
	for ind in ind_dict.keys():
		ind_dict[ind] += [snp_dict.get(int(ind))]
	with open('../figures/table_snp_importance.csv','w') as outfile:
		outfile.write('Rank\tSNP name\tP-value\tImportance * 10-3\tBoruta index\n')
		for i, (pvalue, importance, noiseimp, snpname) in enumerate(sorted(ind_dict.values(),key=lambda x:x[1],reverse=True)):
			outfile.write(u'{0}\t{1}\t{2:.8}\t{3:.6}\t{4}\n'.format(i+1,snpname,pvalue,importance,noiseimp).encode('utf-8'))

	snp_stat, gene_stat = statistic_candidate_gene(do_print=False)

	plt.figure(figsize=(6,5)); plt.xlim(-10, 200)
	x = sorted([snp[2] for snp in snp_stat.values()])[::-1]
	plt.plot(range(len(x)),x,'kx',alpha=1.0)
	plt.xlabel('SNP Rank')
	plt.ylabel('Importance $\\times 10^{-3}$')
	plt.savefig('../figures/snp_rank_importance.png')

	plt.figure(figsize=(6,5));
	x = [snp[2] for snp in snp_stat.values()]
	y = [snp[4] for snp in snp_stat.values()]
	plt.plot(x,y,'kx',alpha=1.0)
	plt.xlabel('Importance $\\times 10^{-3}$')
	plt.ylabel('p-value')
	plt.savefig('../figures/snp_importance_pvalue.png')


def plot_snp_num_accuracy(data):
	'''
		绘图位点数量与准确率关系
	'''
	plt.figure(figsize=(6,5));
	x, y = zip(*data)
	plt.plot(x,y,'kx--',alpha=1.0)
	plt.xlabel('Number of SNP')
	plt.ylabel('F-measure')
	plt.savefig('../figures/snp_num_accuracy.png')


if __name__ == '__main__':
	#### 检验个体独立 ####
	# display_person_distance()

	#### 位点显著检验 ####
	# significance_test(p_threshold=0.05)

	#### 加噪特征抽取 ####
	# feature_selection()
	# feature_selection(do_plot=True)

	#### 位点相关评估 ####
	# snp_validation(classifier='SVM_rbf')
	# snp_validation(classifier='SVM_linear')
	# snp_validation(classifier='DecisionTree')
	# snp_validation(classifier='ExtraTrees')
	# snp_validation(classifier='RandomForest')
	# snp_validation(classifier='SGDClassifier_L1')
	# snp_validation(classifier='SGDClassifier_L2')
	# snp_validation(classifier='AdaBoost')
	# snp_validation(classifier='GradientBoosting')
	# snp_validation(classifier='BernoulliNB')
	# snp_validation(classifier='LogisticRegression_L1')
	# snp_validation(classifier='LogisticRegression_L2')

	#### 绘图位点分布 ####
	# plot_snp_difference([2937,8379,7736,1540,9423,8588,5936,3771,79,4525,961,4931])

	#### 绘图位点权重 ####
	# plot_snp_importance()

	#### 绘图位点数量与准确率关系 ####
	# plot_snp_num_accuracy([(1,0.5487),(10,0.5969),(20,0.6500),(40,0.7164),(60,0.7324),(80,0.7399)])

	pass
