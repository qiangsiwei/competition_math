# -*- encoding:utf-8 -*-

import fileinput


def SNP_encode(genotypefile='../data/genotype.dat', outfile='../data/genotype_encode.dat'):
	'''
		位点编码
	'''
	for line in fileinput.input(genotypefile):
		if fileinput.lineno() == 1:
			Stats, StatsByCol = {}, [{} for _ in xrange(len(line.strip().decode('utf-8').split()))]
		else:
			for i, s in enumerate(line.strip().decode('utf-8').split()):
				Stats[s] = Stats.get(s,0)+1; StatsByCol[i][s] = StatsByCol[i].get(s,0)+1
	fileinput.close()
	for _s,_n in sorted(Stats.items(),key=lambda x:x[1],reverse=True):
		print 'SNP:{0}\tfreq:{1}'.format(_s,_n)
	ZEROs = [sorted([(_snp, _freq) for _snp, _freq in _dict.items() if _snp[0]==_snp[1]],\
				key=lambda x:x[1],reverse=True)[0][0] for _dict in StatsByCol]
	with open(outfile,'w') as output:
		for line in fileinput.input(genotypefile):
			if fileinput.lineno() == 1:
				output.write(line)
			else:
				SNPs = line.strip().decode('utf-8').split()
				assert len(SNPs) == len(ZEROs)
				output.write('{0}\n'.format(u' '.join(map(lambda x:u'{0}'.format(x), \
								[0 if snp == zero else 1 if snp[0]!=snp[1] else 2 \
									for snp, zero in zip(SNPs, ZEROs)]))).encode('utf-8'))


if __name__ == '__main__':
	#### 位点编码 ####
	# SNP_encode()

	pass
