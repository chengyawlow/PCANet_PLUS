
% Title   : "Stacking PCANet+: An Overly Simplified ConvNets Baseline for Face Recognition", accepted by IEEE Signal Processing Letters, August 2017. 
% Authors : C. Y. Low, A. B. J. Teoh, K. A. Toh
% Affl.   : Yonsei University, Seoul, South Korea
% Email   : {chengyawlow, bjteoh, katoh}@yonsei.ac.kr
% 	
% Be noted that we modified from the PCANet implementation provide by the authors as follows:
% T. H. Chan, K. Jia, S. Gao, J. Lu, Z. Zeng, and Y. Ma, "PCANet: a simple deep learning baseline for image classification?" IEEE Trans. Image Process., vol. 24, no. 12, pp. 5017-5032, Dec. 2015.
% URL : http://mx.nthu.edu.tw/~tsunghan/Source%20codes.html

function [ X_OUT ] = PCANet_PLUS_Output( X_IN, PS, NUM_FILT, V, PCANet )
% function [ X_OUT ] = PCANet_PLUS_Output( X_IN, PS, NUM_FILT, BN, BN_REG, V )

    % Initialize PCA filter ensemble parameters    
    [ X_H, X_W, NUM_IMG_CHAN ] = size( X_IN{ 1 } );
    
    NUM_IMG = numel( X_IN );
    ZP_DIM = ( PS - 1 ) / 2;  
    
    X_OUT = cell( NUM_IMG, 1 );

    %% Convolve X_TR with V, accordingly
    for IMG_ID = 1 : NUM_IMG
    
        X = zeros( X_H + PS - 1, X_W + PS - 1, NUM_IMG_CHAN );
        X( ( ZP_DIM + 1 ) : end - ZP_DIM,( ZP_DIM + 1 ) : end - ZP_DIM,: ) = X_IN{ IMG_ID };

        X_PATCHES = im2col_BN_REG( X, [ PS, PS ], PCANet );
        
        % Convolution Output
        X_OUT_TEMP = [];
        
        for FILT_ID = 1 : NUM_FILT    
            TEMP = V( :, FILT_ID )' * X_PATCHES;
            TEMP = reshape( TEMP, X_H, X_W );
            X_OUT_TEMP = cat( 3, X_OUT_TEMP, TEMP );
            clear TEMP;           
        end

        X_OUT{ IMG_ID } = X_OUT_TEMP;
        X_IN{ IMG_ID } = [];
    
        clear X_OUT_TEMP;
        pause(0.0001);
    
    end
 
    %% Clearvars -except X_OUT
    clearvars -except X_OUT;

end



