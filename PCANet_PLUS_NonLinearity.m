
% Title   : "Stacking PCANet+: An Overly Simplified ConvNets Baseline for Face Recognition", accepted by IEEE Signal Processing Letters, August 2017. 
% Authors : C. Y. Low, A. B. J. Teoh, K. A. Toh
% Affl.   : Yonsei University, Seoul, South Korea
% Email   : {chengyawlow, bjteoh, katoh}@yonsei.ac.kr
% 	
% Be noted that we modified from the PCANet implementation provide by the authors as follows:
% T. H. Chan, K. Jia, S. Gao, J. Lu, Z. Zeng, and Y. Ma, "PCANet: a simple deep learning baseline for image classification?" IEEE Trans. Image Process., vol. 24, no. 12, pp. 5017-5032, Dec. 2015.
% URL : http://mx.nthu.edu.tw/~tsunghan/Source%20codes.html

function [ X_NL, X_NL_DIM ] = PCANet_PLUS_NonLinearity( X, PCANet, Stage_ID )
    
    %% Set NUM_IMG, NL
    NUM_IMG = numel( X );
    NL = PCANet.NL;

    for IMG_ID = 1 : NUM_IMG
        
        
        %% if IMG_ID == 1, initialize NL Parameters
        if IMG_ID == 1
                        
            NL_PoolWindow = PCANet.NL_PoolWindow( Stage_ID );
            NL_PoolStride = PCANet.NL_PoolStride( Stage_ID );    
    
            NUM_IMG = numel( X );
            
            X_NL = cell( NUM_IMG, 1 );
            
            [ X_DIM( 1 ), X_DIM( 2 ), NUM_IMG_CHAN ] = size( X{ 1 } );
   
            if NL == 1
                X_NL_DIM = [ floor( ( X_DIM(1) - NL_PoolWindow ) / NL_PoolStride ) + 1, floor( ( X_DIM(2) - NL_PoolWindow ) / NL_PoolStride ) + 1 ];
                assert( NL_PoolWindow ~=0 && NL_PoolStride ~= 0 );
            else
                X_NL_DIM = X_DIM;
            end
                      
        end
        
        %% Perform non-linearity on X, with respect to NL
        if NL == 0
        
            X_NL_TEMP = X{ IMG_ID };
              
        elseif NL == 1
            
            X_NL_TEMP = zeros( X_NL_DIM(1), X_NL_DIM(2), NUM_IMG_CHAN ); 
            
            for IMG_CHAN_ID = 1 : NUM_IMG_CHAN
                                
                X_PATCHES = im2col( X{ IMG_ID }( :, :, IMG_CHAN_ID ), [ NL_PoolWindow, NL_PoolWindow ], 'sliding' );
                
                if IMG_ID == 1 && IMG_CHAN_ID == 1 
                    NUM_PATCHES_BLK = ( X_DIM( 1 ) - NL_PoolWindow ) + 1;
                    NUM_BLK = size( X_PATCHES, 2 ) / NUM_PATCHES_BLK;
                end

                X_POOLED = [];
                for BLK_ID = 1 : NL_PoolStride : NUM_BLK
                    X_PATCHES_TEMP = X_PATCHES( :, ( BLK_ID - 1 ) * NUM_PATCHES_BLK + 1 : ( BLK_ID - 1 ) * NUM_PATCHES_BLK + NUM_PATCHES_BLK );
                    X_PATCHES_TEMP = X_PATCHES_TEMP( :, 1 : NL_PoolStride : NUM_PATCHES_BLK  );
                    X_POOLED = cat( 2, X_POOLED, X_PATCHES_TEMP );
                end
                            
                X_POOLED = mean( X_POOLED, 1 ); 

                X_NL_TEMP( :, :, IMG_CHAN_ID ) = reshape( X_POOLED, X_NL_DIM );
                
                clear X_POOLED X_PATCHES_TEMP X_PATCHES;
                
            end
                
        end
        
        X_NL{ IMG_ID } = X_NL_TEMP;
        X{ IMG_ID } = [];
        clear X_NL_TEMP;
        
    end
                 
    %% Clear all, except FeaMap_Pooled
    clearvars -except X_NL X_NL_DIM;
    pause(0.0001);
    
end

