
% Title   : "Stacking PCANet+: An Overly Simplified ConvNets Baseline for Face Recognition", accepted by IEEE Signal Processing Letters, August 2017. 
% Authors : C. Y. Low, A. B. J. Teoh, K. A. Toh
% Affl.   : Yonsei University, Seoul, South Korea
% Email   : {chengyawlow, bjteoh, katoh}@yonsei.ac.kr
% 	
% Be noted that we modified from the PCANet implementation provide by the authors as follows:
% T. H. Chan, K. Jia, S. Gao, J. Lu, Z. Zeng, and Y. Ma, "PCANet: a simple deep learning baseline for image classification?" IEEE Trans. Image Process., vol. 24, no. 12, pp. 5017-5032, Dec. 2015.
% URL : http://mx.nthu.edu.tw/~tsunghan/Source%20codes.html

function [ V ] = PCANet_PLUS_LearnFilterBank( X, PS, NUM_FILT, PCANet ) 

    %% Initialize PCA filter ensemble parameters
    NUM_IMG = numel( X );
    NUM_IMG_CHAN = size( X{ 1 }, 3 );
 
    X_PATCHES_COV = zeros( NUM_IMG_CHAN * PS .^ 2, NUM_IMG_CHAN * PS .^2 );

    %% Learning PCA filter ensemble V
    for IMG_ID = 1 : NUM_IMG
        
        X_PATCHES = im2col_BN_REG( X{ IMG_ID }, [ PS, PS ], PCANet );
        
        if IMG_ID == 1
            NUM_PATCHES = size( X_PATCHES, 2 );
        end
        
        X_PATCHES_COV = X_PATCHES_COV + X_PATCHES * X_PATCHES';
                       
    end

    X_PATCHES_COV = X_PATCHES_COV ./ ( NUM_IMG * NUM_PATCHES );

    [ V, D ] = eig( X_PATCHES_COV );
    [ D, IND ] = sort( diag( D ), 'descend' );

    %% Extract principal eigen-vectors & eigen-values, with respect to NUM_FILT     
    V = V( :, IND( 1 : NUM_FILT ) );  
    D = D( 1 : NUM_FILT );
    
    %% Display D
    D
    
    %% Clear ALL, except V
    clearvars -except V;
    pause( 0.0001 );

end



 



