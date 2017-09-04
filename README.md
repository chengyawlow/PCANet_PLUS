# PCANet_PLUS
MATLAB Codes 


	Title   : "Stacking PCANet+: An Overly Simplified ConvNets Baseline for Face Recognition", accepted by IEEE Signal Processing Letters, August 2017. 
	Authors : C. Y. Low, A. B. J. Teoh, K. A. Toh
	Affl.   : Yonsei University, Seoul, South Korea
	Email   : {chengyawlow, bjteoh, katoh}@yonsei.ac.kr
	
	Be noted that we modified from the PCANet implementation provided by the authors as follows:
	T. H. Chan, K. Jia, S. Gao, J. Lu, Z. Zeng, and Y. Ma, "PCANet: a simple deep learning baseline for image classification?" IEEE Trans. Image Process., vol. 24, no. 12, pp. 5017-5032, Dec. 2015.
	URL : http://mx.nthu.edu.tw/~tsunghan/Source%20codes.html
	
	****************************************.
	
	To reproduce the reported results (for FERET only):
	
	1. Please email us for "FERET_I_128_128.mat".
	
	2. Run FERET_PCANet_PLUS_MAIN.
	
	The pre-learned PCA filter are provided in PCANet_PLUS_PRE_LEARNED_FILTERS.
	
	However, to trigger PCA filter learning, please set LEARNING_FLAG = 1 (by DEFAULT, LEARNING_FLAG = 0). 
	
	The sample results are as follows :
	
	
	% ***** PERFORMANCE SUMMARY ***** 
	% 
	% recogRate_CD = 	<<<<< Rank-1 Identification Rate (%) W/O WPCA
	% 
	%    93.6402   88.6598   74.7922   65.3846   80.6192	<<<<< Layer-1
	%    95.3975   99.4845   86.9806   85.4701   91.8332	<<<<< Layer-2
	%    97.4895  100.0000   90.1662   88.4615   94.0293 	<<<<< Layer-3
	% 
	% 
	% recogRate_CD_wPCA =	<<<<< Rank-1 Identification Rate (%) W/ WPCA 1190
	% 
	%    99.4979   99.4845   92.6593   88.8889   95.1327	<<<<< Layer-1
	%    99.3305  100.0000   95.4294   94.4444   97.3011	<<<<< Layer-2	
	%    99.2469  100.0000   96.2604   94.4444   97.4879	<<<<< Layer-3
	%    99.4142  100.0000   96.9529   96.5812   98.2371	<<<<< Fea. Conc. for Layer-1, 2, 3.
	% 
	% 
	% PCANet = 
	% 
	%   struct with fields:
	% 
	%               NumStages: 3
	%	            PatchSize: [7 7 9]
	%              NumFilters: [8 16 16]
	%                      BN: 2
	%                BN_DESCR: 'Z-Score Norm.'
	%                  BN_REG: 0
	%                      NL: 1
	%                NL_DESCR: 'MeanPool'
	%           NL_PoolWindow: [3 3 3]
	%           NL_PoolStride: [1 1 1]
	%           HistNumFeaMap: 8
	%     HistBlkOverlapRatio: 0
	%            HistImgSz_PX: [128 128]
	%           HistBlkSz_BLK: [8 8]
	%            HistBlkSz_PX: [16 16]
	%           LEARNING_FLAG: 1
	%                 TR_NORM: 0
	%                     DIM: [16384 32768 32768 81920]
	%               wPCA_FLAG: 1
	%                  k_wPCA: 1190

	
