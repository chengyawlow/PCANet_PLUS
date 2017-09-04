
% Title   : "Stacking PCANet+: An Overly Simplified ConvNets Baseline for Face Recognition", accepted by IEEE Signal Processing Letters, August 2017. 
% Authors : C. Y. Low, A. B. J. Teoh, K. A. Toh
% Affl.   : Yonsei University, Seoul, South Korea
% Email   : {chengyawlow, bjteoh, katoh}@yonsei.ac.kr
% 	
% Be noted that we modified from the PCANet implementation provide by the authors as follows:
% T. H. Chan, K. Jia, S. Gao, J. Lu, Z. Zeng, and Y. Ma, "PCANet: a simple deep learning baseline for image classification?" IEEE Trans. Image Process., vol. 24, no. 12, pp. 5017-5032, Dec. 2015.
% URL : http://mx.nthu.edu.tw/~tsunghan/Source%20codes.html

function FERET_PCANet_PLUS_MAIN
  
    % clear all;
    clc;
    
    fprintf('\n');
    fprintf(' -----------------------------------------------------------\n');
    fprintf('                   FERET_PCANet_PLUS_MAIN                   \n');
    fprintf(' -----------------------------------------------------------\n');    
    
    %% Locate PCANet_PLUS_UTILITIES, PCANet_PLUS_FILTERS
    addpath('PCANet_PLUS_UTILITIES');
        
    %% Initialize PCANet Parameters    
    PCANet_NumStages = 3;
    PCANet_PatchSize = [ 7, 7, 9 ];
    PCANet_NumFilters = [ 8, 16, 16 ];
    
    PCANet.NumStages = PCANet_NumStages;
    PCANet.PatchSize = PCANet_PatchSize;
    PCANet.NumFilters = PCANet_NumFilters;

    % Set Other PCANet Parameters
    % BN : Batch Normalization
    % BN = 0 : No BN
    % BN = 1 : Zero-Mean
    % BN = 2 : Z-Score
    BN = 2;
    BN_REG = 0;
    if BN == 1
        BN_REG = 0;
    end
    PCANet.BN = BN;
    if PCANet.BN == 1
        PCANet.BN_DESCR = 'Zero-Mean Norm.';
    elseif PCANet.BN == 2
        PCANet.BN_DESCR = 'Z-Score Norm.';
    end
    % PCANet.BN_FLAG = BN_FLAG;   
    PCANet.BN_REG = BN_REG;  
        
    % Configure PCANet_NL, i.e., non-linearity
    % PCANet_NL = 0 : Linaer
    % PCANet_NL = 1 : MeanPool
    PCANet_NL = 1;
    PCANet.NL = PCANet_NL;
    if PCANet.NL == 0
        PCANet.NL_DESCR = 'Linear';
    elseif PCANet.NL == 1
        PCANet.NL_DESCR = 'MeanPool';
    end
           
    PCANet_NL_PoolWindow = 0;
    PCANet_NL_PoolStride = 0;
    if PCANet.NL == 1
        PCANet_NL_PoolWindow = 3;
        PCANet_NL_PoolStride = 1;
    end 
    PCANet.NL_PoolWindow = ones( 1, PCANet.NumStages ) .* PCANet_NL_PoolWindow;
    PCANet.NL_PoolStride = ones( 1, PCANet.NumStages ) .* PCANet_NL_PoolStride;
    
    % -----------------------------------------------
    % Set Histogram Feature Encoding Parameters
    % DEFAULT : PCANet.HistNumFeaMap = 8
    PCANet.HistNumFeaMap = 8;
    % Initialize PCANet.HistBlockSize in BLOCK, will be converted to PIXELS
    PCANet.HistBlkOverlapRatio = 0;
    PCANet.HistImgSz_PX = [ 128, 128 ];
    PCANet.HistBlkSz_BLK = [ 8, 8 ];
    PCANet.HistBlkSz_PX = [ 0, 0 ];
    
    % Set Other PCANet Parameters
    DIM = zeros( 1, PCANet.NumStages );
    PCANet.DIM = DIM;
    
    % Set LEARNING_FLAG
    % if LEARNING_FLAG = 1, trigger PCA filter learning stage
    % otherwise, load pre-learned PCA filters 
    LEARNING_FLAG = 0;
    
    % Trigger TiedRank ( TR ) Normalization
    TR_NORM = 0;
    PCANet.TR_NORM = TR_NORM;   
              
    % Set wPCA_FLAG
    wPCA_FLAG = 1;
    k_wPCA = 0;
    if wPCA_FLAG == 1
        k_wPCA = 1190;
    end
    PCANet.wPCA_FLAG = wPCA_FLAG;
    PCANet.k_wPCA = k_wPCA;
                    
    %% Validate if length( PCANet.NumFilters ) ~= PCANet.NumStages
    if length( PCANet.NumFilters ) ~= PCANet.NumStages || length( PCANet.PatchSize ) ~= PCANet.NumStages
        fprintf( '\n' );
        fprintf( 'ERROR : numel( PCANet.PatchSize / PCANet.NumFilters / PCANet.NL_PoolStride ) ~= PCANet.NumStages ... !\n' );
        fprintf( '\n' );
        return;
    end
    
    %% Load FERET images
    load ( 'FERET_I_128_128.mat ' );
    
    H = fa.h;
    W = fa.w;    
   
    %% Finalize and display PCANet Parameters
    % Reset HistImgSz_PX 
    PCANet.HistImgSz_PX = cat( 2, H, W );
    
    % Reset PCANet.HistBlockSize in PIXELS
    PCANet.HistBlkSz_PX = cat( 2, floor( H / PCANet.HistBlkSz_BLK(1) ), floor( W / PCANet.HistBlkSz_BLK(2) ) );
   
    DIM_TEMP = prod( PCANet.HistBlkSz_BLK ) * 2.^ PCANet.HistNumFeaMap;
    for Stage_ID = 1 : PCANet.NumStages
        DIM( Stage_ID ) = ( PCANet.NumFilters( Stage_ID ) / PCANet.HistNumFeaMap ) * DIM_TEMP;
    end
    
    DIM = cat( 2, DIM, sum( DIM ) );
    PCANet.DIM = DIM;    
    
    % Display PCANet Parameters
    PCANet 
           
    %% Learn or load PCA filters, V
    % Define PCANet_FILTER_FOLD
    PCANet_FILTER_FOLD = strcat( pwd, '\PCANet_PLUS_PRE_LEARNED_FILTERS\' );
    if LEARNING_FLAG == 1
        PCANet_FILTER_FOLD = strcat( pwd, '\' );
    end
    PCANet_FILTER_FOLD
        
    % Define PCANet_FILTERS_TAG
    PCANet_FILTERS_TAG = [ 'FERET_PCANet_PLUS_2' ];    
    if PCANet.BN == 1
        PCANet_FILTERS_TAG = [ 'FERET_PCANet_PLUS_2_ZM' ];
    elseif PCANet.BN == 2
        PCANet_FILTERS_TAG = [ 'FERET_PCANet_PLUS_2_ZS' ];
    end
    
    if PCANet.BN_REG ~= 0
        PCANet_FILTERS_TAG = cat( 2, PCANet_FILTERS_TAG, '_REG', num2str( PCANet.BN_REG ) );
    end
   
    if PCANet.NL == 1
        PCANet_FILTERS_TAG = cat( 2, PCANet_FILTERS_TAG, '_MeanPool' );
    end

    for Stage_ID = 1 : PCANet.NumStages
        
        TAG_TEMP = [ '_S', num2str( Stage_ID ), 'PS', num2str( PCANet.PatchSize( Stage_ID ) ), 'NF', num2str( PCANet.NumFilters( Stage_ID ) ) ]; 
        if PCANet.NL == 1
            TAG_TEMP = [ TAG_TEMP, 'PoW', num2str( PCANet.NL_PoolWindow( Stage_ID ) ), 'PoS', num2str( PCANet.NL_PoolStride( Stage_ID ) ) ];
        end
        PCANet_FILTERS_TAG = cat( 2, PCANet_FILTERS_TAG, TAG_TEMP );
        clear TAG_TEMP;
        pause(0.0001);
        
    end 
    
    PCANet_FILTERS_TAG 

    % Trigger PCA filter learning 
    if LEARNING_FLAG == 1
        X_TR = double( train.X ) ./ 255;
        % X_TR = double( fa.X ) ./ 255;
        FERET_PCANet_PLUS_FilterBank_MAIN( PCANet, X_TR, H, W, PCANet_FILTERS_TAG )
    end
    
    % Load pre-learned PCA Filters, V 
    load( [ PCANet_FILTER_FOLD, PCANet_FILTERS_TAG, '.mat' ], 'V' ); 
    
    %% Extract PCANet features from FERET - FA Images 
    fprintf( '\n' );
    fprintf( '**********' );
    fprintf( '\n' );
    
    fprintf( '\n' );
    fprintf( 'EXTRACTING PCANet+ FEATURES FROM FERET - FA IMAGES ... ' );
    fprintf('\n');   
        
    X_TR = double( fa.X ) ./ 255;
    Y_TR = fa.y;
    X_TR_H = fa.h;
    X_TR_W = fa.w;

    % Convert columnar X_TR into cells 
    X_TR = mat2imgcell( X_TR, X_TR_H, X_TR_W, 'gray' );
    
    X_PCANet_TR = zeros( DIM( end ), numel( X_TR ) );
            
    for IMG_ID = 1 : numel( X_TR )
                
        if IMG_ID == 1 || mod( IMG_ID, 100 ) == 0 || IMG_ID == numel( X_TR )
            fprintf( '\n' ); 
            fprintf( 'PROCESSING IMG ID : %d', IMG_ID );
            fprintf( '\n' );
        end
        
        [ X_PCANet_TR_TEMP ] = PCANet_PLUS_FeaExt( X_TR( IMG_ID ), V, PCANet );

        X_PCANet_TR( :, IMG_ID ) = X_PCANet_TR_TEMP;
        
        clear X_PCANet_TR_TEMP;
        pause( 0.0001 );
                
    end
   

    X_PCANet_TR_SZ = size( X_PCANet_TR )
    Y_PCANet_TR = Y_TR;
       
    clear X_TR Y_TR;
    pause( 0.0001 );   
    
    %% Apply wPCA to X_PCANet_TR on FERET FA
    Stages_MAX = PCANet.NumStages;
    if PCANet.NumStages > 1
        Stages_MAX = PCANet.NumStages + 1;
    end
    
    if wPCA_FLAG == 1
        
        fprintf( '\n' );
        fprintf( 'LEARNING wPCA FROM FERET - FA ... \n' );
        fprintf( '\n' );  
        
        X_PCANet_TR_wPCA = cell( 1, Stages_MAX );
        X_PCANet_TR_MEAN = cell( 1, Stages_MAX );
        eigen_wPCA = cell( 1, Stages_MAX );
        
        % Learn FULLY-CONNECTED wPCA Parameters from FERET - TR Features, with respect to Stage_ID
        for Stage_ID = 1 : Stages_MAX          
            
            % X_PCANet_TR_TEMP = X_PCANet_TR( 1 : sum( DIM( 1 : Stage_ID ) ), : );
            if Stage_ID == 1
                X_PCANet_TR_TEMP = X_PCANet_TR( 1 : DIM( 1 ), : );  
            elseif Stage_ID > 1 && Stage_ID ~= Stages_MAX 
                X_PCANet_TR_TEMP = X_PCANet_TR( sum ( DIM( 1 : Stage_ID - 1 ) ) + 1 : sum ( DIM( 1 : Stage_ID ) ), : );
            elseif Stage_ID == Stages_MAX
                X_PCANet_TR_TEMP = X_PCANet_TR;
            end
            
            X_PCANet_TR_TEMP = normc( X_PCANet_TR_TEMP );
            % X_PCANet_TR_TEMP = zscore( X_PCANet_TR_TEMP );
            
            [ X_PCANet_TR_MEAN{ Stage_ID } , eigen_wPCA{ Stage_ID }, ~ ] = wPCA_FAST_GENERALIZED( X_PCANet_TR_TEMP, k_wPCA, 1 ); 
            
            X_PCANet_TR_wPCA_TEMP = bsxfun( @minus, X_PCANet_TR_TEMP, X_PCANet_TR_MEAN{ Stage_ID } );
            X_PCANet_TR_wPCA{ Stage_ID } = eigen_wPCA{ Stage_ID } * X_PCANet_TR_wPCA_TEMP;
                                            
            clear X_PCANet_TR_TEMP X_PCANet_TR_wPCA_TEMP;
            pause( 0.0001 );
          
        end
        
    end
        
    %% PCANet Testing on FERET FB, FC, DUP I, DUP II    
    X_TT_DESCR = { 'FB', 'FC', 'DUP I', 'DUP II' };
    X_TT_ALL = { fb, fc, dup1, dup2 };
        
    recogRate_CD = zeros( PCANet.NumStages, numel( X_TT_ALL ) );
    recogRate_CD_wPCA = zeros( PCANet.NumStages, numel( X_TT_ALL ) );
    if PCANet.NumStages > 1
        recogRate_CD_wPCA = zeros( Stages_MAX, numel( X_TT_ALL ) );
    end
            
    for X_TT_ID = 1 : numel( X_TT_ALL )
        
        fprintf( '\n' );
        fprintf( '**********' );
        fprintf( '\n' );
        
        fprintf( '\n' );
        fprintf( [ 'EXTRACT PCANet+ FEATURES FROM FERET - ', X_TT_DESCR{ X_TT_ID } ] );
        fprintf( '\n' );
        
        X_TT = double( X_TT_ALL{ X_TT_ID }.X ) ./ 255;
        Y_TT = X_TT_ALL{ X_TT_ID }.y;
        X_TT_H = X_TT_ALL{ X_TT_ID }.h;
        X_TT_W = X_TT_ALL{ X_TT_ID }.w;
        
        % Convert columnar X_TT into cell representation
        X_TT = mat2imgcell( X_TT, X_TT_H, X_TT_W, 'gray' ); 
        
        X_PCANet_TT = zeros( DIM( end ), numel( X_TT ) );
        
        for IMG_ID = 1 : numel( X_TT )
            
            if IMG_ID == 1 || mod( IMG_ID, 100 ) == 0 || IMG_ID == numel( X_TT )
                fprintf( '\n' );
                fprintf( 'PROCESSING IMG ID : %d', IMG_ID );
                fprintf( '\n' );
            end
            
            [ X_PCANet_TT_TEMP ] = PCANet_PLUS_FeaExt( X_TT( IMG_ID ), V, PCANet );
            
            X_PCANet_TT( :, IMG_ID ) =  X_PCANet_TT_TEMP;
            
            clear X_PCANet_TT_TEMP;
            pause( 0.0001 );

        end
        
        X_PCANet_TT_SZ = size( X_PCANet_TT )
        Y_PCANet_TT = Y_TT;
        
        %% Calculate recogRate_CD, recogRate_SD
        fprintf( '\n' );
        fprintf( [ 'CALCULATING recogRate ( W/O wPCA ) for FERET - ', X_TT_DESCR{ X_TT_ID }, '... ' ] );
        fprintf( '\n' ); 
                
        for Stage_ID = 1 : PCANet.NumStages
        
            if Stage_ID == 1
                X_PCANet_TR_TEMP = X_PCANet_TR( 1 : DIM( Stage_ID ), : );  
                X_PCANet_TT_TEMP = X_PCANet_TT( 1 : DIM( Stage_ID ), : );  
            elseif Stage_ID > 1 
                X_PCANet_TR_TEMP = X_PCANet_TR( sum ( DIM( 1 : Stage_ID - 1 ) ) + 1 : sum ( DIM( 1 : Stage_ID ) ), : ); 
                X_PCANet_TT_TEMP = X_PCANet_TT( sum ( DIM( 1 : Stage_ID - 1 ) ) + 1 : sum ( DIM( 1 : Stage_ID ) ), : );  
            end
            
            X_PCANet_TR_TEMP = normc( X_PCANet_TR_TEMP );  
            X_PCANet_TT_TEMP = normc( X_PCANet_TT_TEMP ); 
            
            [ recogRate_CD( Stage_ID, X_TT_ID ) ] = recognitionRate_CosineDistance( X_PCANet_TR_TEMP, X_PCANet_TT_TEMP, Y_PCANet_TR, Y_PCANet_TT );
            
            clear X_PCANet_TR_TEMP X_PCANet_TT_TEMP;
            pause( 0.0001 );
            
        end
        
        % Display recogRate_CD
        recogRate_CD
           
        %% Apply wPCA to X_PCANet_TT
        if wPCA_FLAG == 1
                       
            % Calculate recogRate_CD, recogRate_SD
            fprintf('\n');
            fprintf( [ 'CALCULATING recogRate ( W/ wPCA ) for FERET - ', X_TT_DESCR{ X_TT_ID }, '... \n ' ] );
                       
            % Apply FULLY-CONNECTED wPCA Parameters to FERET - TT Features, with respect to Stage_ID
            for Stage_ID = 1 : Stages_MAX
                
                X_PCANet_TR_wPCA_TEMP = X_PCANet_TR_wPCA{ Stage_ID }; 
            
                if Stage_ID == 1
                    X_PCANet_TT_TEMP = X_PCANet_TT( 1 : DIM( Stage_ID ), : );  
                elseif Stage_ID > 1 && Stage_ID ~= Stages_MAX
                    X_PCANet_TT_TEMP = X_PCANet_TT( sum ( DIM( 1 : Stage_ID - 1 ) ) + 1 : sum ( DIM( 1 : Stage_ID ) ), : );  
                elseif Stage_ID == Stages_MAX
                    X_PCANet_TT_TEMP = X_PCANet_TT;
                end
                
                X_PCANet_TT_TEMP = normc( X_PCANet_TT_TEMP );
                
                X_PCANet_TT_wPCA_TEMP = bsxfun( @minus, X_PCANet_TT_TEMP, X_PCANet_TR_MEAN{ Stage_ID } );
                X_PCANet_TT_wPCA_TEMP = eigen_wPCA{ Stage_ID } * X_PCANet_TT_wPCA_TEMP;
                
                [ recogRate_CD_wPCA( Stage_ID, X_TT_ID ) ] = recognitionRate_CosineDistance( X_PCANet_TR_wPCA_TEMP, X_PCANet_TT_wPCA_TEMP, Y_PCANet_TR, Y_PCANet_TT );
                                             
                clear X_PCANet_TR_wPCA_TEMP;
                clear X_PCANet_TT_wPCA_TEMP X_PCANet_TT_TEMP;
                pause( 0.0001 );

            end
            
            % Display recogRate_CD
            recogRate_CD_wPCA
            
        end
                
    end
    
    %% Display recogRate_CD, recogRate_SD
    fprintf( '\n' );
    fprintf( ' ***** PERFORMANCE SUMMARY ***** ' );
    fprintf( '\n'); 
    
    recogRate_CD = cat( 2, recogRate_CD, mean( recogRate_CD, 2 ) )
    recogRate_CD_wPCA = cat( 2, recogRate_CD_wPCA, mean( recogRate_CD_wPCA, 2 ) )
        
    %% Display PCANet Parameters
    PCANet
      
end

% ***** PERFORMANCE SUMMARY ***** 
% 
% recogRate_CD =
% 
%    93.6402   88.6598   74.7922   65.3846   80.6192
%    95.3975   99.4845   86.9806   85.4701   91.8332
%    97.4895  100.0000   90.1662   88.4615   94.0293
% 
% 
% recogRate_CD_wPCA =
% 
%    99.4979   99.4845   92.6593   88.8889   95.1327
%    99.3305  100.0000   95.4294   94.4444   97.3011
%    99.2469  100.0000   96.2604   94.4444   97.4879
%    99.4142  100.0000   96.9529   96.5812   98.2371
% 
% 
% PCANet = 
% 
%   struct with fields:
% 
%               NumStages: 3
%               PatchSize: [7 7 9]
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
