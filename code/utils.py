# -*- encoding:utf-8 -*-

import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def load_x_y(featurefile='../data/genotype_encode.dat', labelfile='../data/phenotype.txt'):
	'''
		加载数据
	'''
	if os.path.isfile('../data/x.npy'): 
		x = np.load('../data/x.npy')
	else:
		x = np.array([map(lambda x:int(x), line.strip().decode('utf-8').split(' ')) for line in list(fileinput.input(featurefile))[1:]])
		np.save('../data/x.npy', x)
	if os.path.isfile('../data/y.npy'): 
		y = np.load('../data/y.npy')
	else:
		y = np.array([int(line.strip().decode('utf-8')) for line in fileinput.input(labelfile)])
		np.save('../data/y.npy', y)
	return x, y


def get_snp_dict(featurefile='../data/genotype_encode.dat', outfile='../data/snp_names.dat', return_dict=True, key_is_snp=False):
	'''
		解析位点数据
	'''
	if return_dict:
		if key_is_snp:
			return {snp:ind for ind, snp in enumerate(open(featurefile,'r').readline().strip().decode('utf-8').split())}
		else:
			return {ind:snp for ind, snp in enumerate(open(featurefile,'r').readline().strip().decode('utf-8').split())}
	else:
		with open(outfile,'w') as output:
			for ind, snp in enumerate(open(featurefile,'r').readline().strip().decode('utf-8').split()):
				output.write('{0}\t{1}\n'.format(ind,snp).encode('utf-8')) 


def get_gene_dict(geneinfodir='../data/gene_info/*.dat'):
	'''
		解析基因数据
	'''
	gene_dict = {}; gene_inverse_dict = {}
	for filename in glob.glob(geneinfodir):
		gene = os.path.basename(filename)
		for ind, snp in enumerate(open(filename,'r').read().strip().decode('utf-8').split(u'\n')):
			gene_dict[gene] = gene_dict.get(gene,[])+[(ind, snp)]
			gene_inverse_dict[snp] = (gene, ind)
	return gene_dict, gene_inverse_dict


def get_gene_set(snp_name=''):
	'''
		获取同基因的所有位点
	'''
	gene_dict, gene_inverse_dict = get_gene_dict()
	print gene_inverse_dict.get(snp_name)
	print gene_dict.get(gene_inverse_dict.get(snp_name)[0])


def plot_gene_distribution():
	'''
		绘制基因分布
	'''
	gene_dict, gene_inverse_dict = get_gene_dict()
	gene_snp_count = sorted([len(snps) for snps in gene_dict.values()],reverse=True)
	plt.plot(xrange(len(gene_snp_count)), gene_snp_count, alpha=0.4)
	plt.xlabel('Number of SNPs')
	plt.ylabel('Number of genes')
	plt.savefig('../figures/snp_gene_distribution.png')


if __name__ == '__main__':
	#### 加载数据 ####
	# load_x_y()

	#### 解析位点数据 ####
	# get_snp_dict()

	#### 解析基因数据 ####
	# get_gene_dict()

	#### 获取同基因的所有位点 ####
	# get_gene_set(snp_name=u'rs2273298') # 2937th SNP

	#### 绘制基因分布 ####
	# plot_gene_distribution()

	pass
