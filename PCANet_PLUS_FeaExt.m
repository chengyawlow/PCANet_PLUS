
% Title   : "Stacking PCANet+: An Overly Simplified ConvNets Baseline for Face Recognition", accepted by IEEE Signal Processing Letters, August 2017. 
% Authors : C. Y. Low, A. B. J. Teoh, K. A. Toh
% Affl.   : Yonsei University, Seoul, South Korea
% Email   : {chengyawlow, bjteoh, katoh}@yonsei.ac.kr
% 	
% Be noted that we modified from the PCANet implementation provide by the authors as follows:
% T. H. Chan, K. Jia, S. Gao, J. Lu, Z. Zeng, and Y. Ma, "PCANet: a simple deep learning baseline for image classification?" IEEE Trans. Image Process., vol. 24, no. 12, pp. 5017-5032, Dec. 2015.
% URL : http://mx.nthu.edu.tw/~tsunghan/Source%20codes.html

% X : a cell structure, with each cell contains a 3D matrix

function [ X_HIST ] = PCANet_PLUS_FeaExt( X, V, PCANet )
        
    %% Initialize X_HIST
    X_HIST = [];
        
    %% Extract PCANet_PLUS features with respect to V, PCANet
    for Stage_ID = 1 : PCANet.NumStages
        
        % Convolve X with V, with respect to Stage_ID
        [ X ] = PCANet_PLUS_Output( X, PCANet.PatchSize( Stage_ID ), PCANet.NumFilters( Stage_ID ), V{ Stage_ID }, PCANet );

        % Perform LBP-like feature encoding       
        X_HIST = cat( 1, X_HIST, PCANet_PLUS_HashingHist( X{ : }, PCANet ) ); 
                        
        %% Apply non-linearity to feature maps, if Stage_ID ~= PCANet.NumStages
        if Stage_ID ~= PCANet.NumStages 

            if PCANet.NL ~= 0
                
                [ X, X_NL_DIM ] = PCANet_PLUS_NonLinearity( X, PCANet, Stage_ID );

                % Update PCANet.HistImgSz_PX, PCANet.HistBlkSz_PX, with respect to X_NL_DIM
                PCANet.HistImgSz_PX = X_NL_DIM;
                PCANet.HistBlkSz_PX = cat( 2, floor( PCANet.HistImgSz_PX( 1 ) / PCANet.HistBlkSz_BLK( 1 ) ), floor( PCANet.HistImgSz_PX( 2 ) / PCANet.HistBlkSz_BLK( 2 ) ) ); 
               
            end

        end
        
        pause( 0.00001 );
                
    end
    
    %% Clear all, except X_HIST
    clearvars -except X_HIST    

end

