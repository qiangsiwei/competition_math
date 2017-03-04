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


def statistic_candidate_gene(do_print=True):
	'''
		统计可疑位点所在基因
	'''
	snp_dict = get_snp_dict()
	gene_dict, gene_inverse_dict = get_gene_dict()
	gene_stat = {}; snp_stat = {}
	for line in open('../data/important_snp_importance_noiseimp.csv','r').read().decode('utf-8').split(u'\n'):
		ind, importance, noiseimp = line.split(u',')
		gene = gene_inverse_dict.get(snp_dict.get(int(ind)))[0]
		snp_stat[ind] = [ind,gene,float(importance),float(noiseimp)]
	for line in open('../data/important_snp_pvalue_OR.csv','r').read().decode('utf-8').split(u'\n'):
		ind, pvalue, OR = line.split(u',')
		snp_stat[ind] += [float(pvalue), float(OR)]

	for ind, (ind,gene,importance,noiseimp,pvalue,OR) in snp_stat.iteritems():
		gene_stat[gene] = gene_stat.get(gene,[])+[(ind,importance)]

	if do_print:
		print 'suspicious gene:', len(gene_stat)
		for gene, array in sorted(gene_stat.items(),key=lambda item:len(item[1]),reverse=True):
			print gene, len(gene_dict.get(gene)), len(array), max([importance for _, importance in array]), sorted(array,key=lambda x:x[1],reverse=True)
	else:
		return snp_stat, gene_stat


def plot_snp_mutual_information(topN=60):
	'''
		绘制可疑位点间互信息
	'''
	from sklearn.metrics import normalized_mutual_info_score
	x, y = load_x_y()
	snp_dict = get_snp_dict()
	gene_dict, gene_inverse_dict = get_gene_dict()
	snp_index = np.array([int(line.split(u',')[0]) for line in open('../data/important_snp_pvalue_OR.csv','r').read().decode('utf-8').split(u'\n')])
	x = x[:,snp_index][:,:topN]; mutual_information = np.zeros((len(x[0]),len(x[0])))
	for i in xrange(len(x[0])):
		for j in xrange(i+1,len(x[0])):
			mutual_information[i,j] = mutual_information[j,i] = normalized_mutual_info_score(x[:,i],x[:,j])
	# correlation = sorted([(snp_index[i],snp_index[j],mutual_information[i,j]) for i in xrange(len(x[0])) for j in xrange(len(x[0]))],key=lambda x:x[2],reverse=True)
	# for snp1, snp2, mi in correlation:
	# 	gene1, gene2 = (gene_inverse_dict[snp_dict[snp]][0] for snp in (snp1, snp2))
	# 	if snp1 != snp2 and gene1 == gene2:
	# 		print snp1, snp2, mi
	fig = plt.figure(figsize=(8,6))
	ax = fig.add_subplot(111)
	(X, Y) = meshgrid(range(topN), range(topN))
	cset1 = pcolormesh(X, Y, mutual_information, cmap=cm.get_cmap("OrRd"))
	plt.xlim(0,topN-1); plt.ylim(0,topN-1)
	colorbar(cset1)
	plt.xlabel('SNP Index')
	plt.ylabel('SNP Index')
	plt.savefig('../figures/gene_snp_mutual_information.png')


def select_most_probable_gene(topN=30, Min=1):
	'''
		筛选最可能基因
	'''
	snp_stat, gene_stat = statistic_candidate_gene(do_print=False)
	print 'Number of gene:\t', sum([1 for snps in gene_stat.values() if len(snps) >= Min])
	return [int(snp[0]) for snps in gene_stat.values() for snp in snps if len(snps) >= Min]


def select_most_probable_gene_regression(topN=20, do_plot=True, classifer='LR'):
	'''
		筛选最可能基因
	'''
	import statsmodels.api as sm
	from sklearn.linear_model import Ridge
	from sklearn.linear_model import RidgeClassifier
	from sklearn.linear_model import LogisticRegression
	from sklearn.cross_validation import KFold
	from sklearn.preprocessing import OneHotEncoder

	if classifer == 'LR':
		clf = LogisticRegression(penalty="l1")
		# clf = LogisticRegression(penalty="l2")
	elif classifer == 'RC':
		clf = RidgeClassifier()
	else:
		raise Exception('Classifer not supported!')

	x, y = load_x_y()
	snp_dict = get_snp_dict(key_is_snp=True)
	gene_dict, gene_inverse_dict = get_gene_dict()
	snp_stat, gene_stat = statistic_candidate_gene(do_print=False)

	gene_R2 = []
	for gene in gene_stat:
		x_gene = x[:,[ind for ind, _ in gene_stat[gene]]]
		x_gene = OneHotEncoder().fit_transform(x_gene).toarray()
		clf.fit(x_gene, y)
		SSres = (clf.predict(x_gene) != y).sum(); SStot = 500; R2 = 1-1.*SSres/SStot
		gene_R2.append((gene, R2))
		# logit = sm.Logit(y, x_gene)
		# result = logit.fit()
		# gene_R2.append((gene, result.prsquared))

	if do_plot:
		with open('../figures/table_gene_importance_linear.csv','w') as outfile:
			outfile.write('Rank\tGene name\tR2\tNumber of suspicious SNP\tHighest importance of suspicious SNP * 10-3\n')
			for i, (gene, R2) in enumerate(sorted(gene_R2, key=lambda x:x[1], reverse=True)):
				outfile.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(i+1, re.sub('.dat','',gene), R2, len(gene_stat[gene]), max([snp_importance for _,snp_importance in gene_stat[gene]])))
		
		if do_plot == 'gene_rank_importance':
			plt.figure(figsize=(6,5)); plt.xlim(-10, 200)
			x = [R2 for _, R2 in sorted(gene_R2, key=lambda x:x[1], reverse=True)]
			plt.plot(range(len(x)),x,'kx',alpha=1.0)
			plt.xlabel('Gene rank')
			plt.ylabel('Coefficient of determination')
			plt.savefig('../figures/gene_rank_importance.png')

		if do_plot == 'gene_snp_relation':
			data = [(len(gene_stat[gene]), max([snp_importance for _,snp_importance in gene_stat[gene]])) for gene, _ in sorted(gene_R2, key=lambda x:x[1], reverse=True)]
			x, y = zip(*data)
			plt.plot(x[:topN],y[:topN],'ro',alpha=0.8)
			plt.plot(x[topN:],y[topN:],'ko',alpha=0.2)
			plt.xlabel('Number of suspicious SNP')
			plt.ylabel('Highest importance of suspicious SNP $\\times 10^{-3}$')
			plt.savefig('../figures/gene_snp_relation.png')
	else:
		return [gene for gene,_ in sorted(gene_R2, key=lambda x:x[1], reverse=True)][:topN]


def gene_validation(Nfeature=20, classifier=''):
	'''
		基因相关评估
	'''
	from sklearn import svm
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.ensemble import AdaBoostClassifier
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.ensemble import GradientBoostingClassifier
	from sklearn.ensemble import ExtraTreesClassifier
	from sklearn.linear_model import SGDClassifier
	from sklearn.linear_model import RidgeClassifier
	from sklearn.linear_model import RidgeClassifierCV
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

	snp_potential = select_most_probable_gene()
	snp_dict = get_snp_dict(key_is_snp=True)
	gene_dict, gene_inverse_dict = get_gene_dict()
	gene_candidate = select_most_probable_gene_regression(do_plot=False)
	snp_candidate = set([snp_dict[snp[1]] for gene in gene_candidate for snp in gene_dict[gene]])&set(snp_potential)
	snp_candidate = np.array(list(snp_candidate))
	print 'Number of SNP:\t', len(snp_candidate), snp_candidate

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


def plot_statistic_gene_snps():
	'''
		绘制基因可疑位点分布
	'''
	snp_stat, gene_stat = statistic_candidate_gene(do_print=False)
	x = sorted([len(snps) for gene, snps in gene_stat.items()],reverse=True)
	plt.figure(figsize=(6,5)); plt.xlim(1, 13)
	plt.hist(x, 12, facecolor='green', alpha=0.5)
	plt.xlabel('Number of Suspicious SNPs')
	plt.ylabel('Number of Genes')
	plt.savefig('../figures/gene_suspicious_snps.png')


if __name__ == '__main__':
	#### 统计可疑位点所在基因 ####
	# statistic_candidate_gene()

	#### 绘制可疑位点间互信息 ####
	# plot_snp_mutual_information()

	#### 筛选最可能基因 ####
	# select_most_probable_gene_regression(do_plot='gene_rank_importance')
	# select_most_probable_gene_regression(do_plot='gene_snp_relation')

	#### 基因相关评估 ####
	# gene_validation(classifier='SVM_rbf')
	# gene_validation(classifier='SVM_linear')
	# gene_validation(classifier='DecisionTree')
	# gene_validation(classifier='ExtraTrees')
	# gene_validation(classifier='RandomForest')
	# gene_validation(classifier='SGDClassifier_L1')
	# gene_validation(classifier='SGDClassifier_L2')
	# gene_validation(classifier='AdaBoost')
	# gene_validation(classifier='GradientBoosting')
	# gene_validation(classifier='BernoulliNB')
	# gene_validation(classifier='LogisticRegression_L1')
	# gene_validation(classifier='LogisticRegression_L2')

	#### 绘制基因可疑位点分布 ####
	# plot_statistic_gene_snps()

	pass
