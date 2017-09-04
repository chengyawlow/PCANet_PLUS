
% Title   : "Stacking PCANet+: An Overly Simplified ConvNets Baseline for Face Recognition", accepted by IEEE Signal Processing Letters, August 2017. 
% Authors : C. Y. Low, A. B. J. Teoh, K. A. Toh
% Affl.   : Yonsei University, Seoul, South Korea
% Email   : {chengyawlow, bjteoh, katoh}@yonsei.ac.kr
% 	
% Be noted that we modified from the PCANet implementation provide by the authors as follows:
% T. H. Chan, K. Jia, S. Gao, J. Lu, Z. Zeng, and Y. Ma, "PCANet: a simple deep learning baseline for image classification?" IEEE Trans. Image Process., vol. 24, no. 12, pp. 5017-5032, Dec. 2015.
% URL : http://mx.nthu.edu.tw/~tsunghan/Source%20codes.html

function [ X_HIST ] = PCANet_PLUS_HashingHist( X, PCANet )

    %% Initialize Parameters   
    % Set feature encoding weights W for binary to decimal conversion
    W = 2.^( PCANet.HistNumFeaMap - 1 : -1 : 0 );
    NUM_HIST = size( X, 3 ) / PCANet.HistNumFeaMap;
    X_HIST = [];
       
    %% Perform LBP-like feature encoding
    for HIST_ID = 1 : NUM_HIST 
        
        T = 0;
        for HIST_FEA_MAP_ID = 1 : PCANet.HistNumFeaMap
            T = T + W( HIST_FEA_MAP_ID ) * Heaviside( X( :, :, ( HIST_ID - 1 ) * PCANet.HistNumFeaMap + HIST_FEA_MAP_ID ) );
        end
                        
        % Perform LBP-alike histograming for each local block in T
        STR = PCANet.HistBlkSz_PX;
        FEA_MAP_SZ = size( T );
        if PCANet.HistBlkOverlapRatio ~= 0
            STR = round( ( 1 - PCANet.HistBlkOverlapRatio ) * PCANet.HistBlkSz_PX );
        end
                
        % T : COLUMNAR
        T = im2col_general( T, PCANet.HistBlkSz_PX, STR );
                      
        % X_HIST & X_HIST_TEMP : COLUMNAR
        X_HIST_TEMP = histc( T, ( 0 : 2 ^ PCANet.HistNumFeaMap - 1 )' );
        
        % Trim X_HIST_TEMP, with respect to PCANet.HistBlkSz_BLK
        if PCANet.HistBlkOverlapRatio ~= 0  
            X_HIST_TEMP = spp( X_HIST_TEMP, FEA_MAP_SZ, STR, PCANet );
        end
                       
        % Trigger TiedRank ( TR ) Normalization
        if PCANet.TR_NORM == 1
            X_HIST_TEMP = TiedRank_Normalization( X_HIST_TEMP );
        elseif PCANet.TR_NORM == 0
            % X_HIST_TEMP = sqrt( X_HIST_TEMP );
            X_HIST_TEMP = normc( sqrt( X_HIST_TEMP ) );
        end
                   
        X_HIST = cat( 2, X_HIST, X_HIST_TEMP );
        clear X_HIST_TEMP;
        
    end
    
    %% Perform Feature Vectorization on X_HIST
    X_HIST = vec( X_HIST );
    
    %% Clear all, except X_HIST
    clearvars -except X_HIST

end

%% Binary Quantization
function X = Heaviside( X ) 
    X = sign( X );
    X( X <= 0 ) = 0;
end

%% Feature Vectorization
function X = vec( X ) 
    X = X(:);
end

%% SPP Encoding
% function beta = spp( blkwise_fea, sam_coordinate, ImgSize, pyramid )
function beta = spp( blkwise_fea, ImgSize, stride, PCANet )

    x_start = ceil( PCANet.HistBlkSz_PX(2) / 2 );
    y_start = ceil( PCANet.HistBlkSz_PX(1) / 2 );
    x_end = floor( ImgSize(2) - PCANet.HistBlkSz_PX(2) / 2 );
    y_end = floor( ImgSize(1) - PCANet.HistBlkSz_PX(1) / 2 );
                
    sam_coordinate = [...
                    kron( x_start : stride : x_end, ones( 1, length( y_start : stride : y_end ) ) ); 
                    kron( ones( 1, length( x_start : stride : x_end ) ), y_start : stride : y_end ) ];

    [dSize, ~] = size(blkwise_fea);

    img_width = ImgSize(2);
    img_height = ImgSize(1);

    % spatial levels
    pyramid = PCANet.SpatialPyramid_LEVEL;
    pyramid_Levels = length(pyramid);
    pyramid_Bins = pyramid.^2;
    tBins = sum(pyramid_Bins);

    beta = zeros(dSize, tBins);
    % beta = [];
    cnt = 0;

    for i1 = 1:pyramid_Levels
    
        Num_Bins = pyramid_Bins(i1);
    
        wUnit = img_width / pyramid(i1);
        hUnit = img_height / pyramid(i1);
    
        % find to which spatial bin each local descriptor belongs
        xBin = ceil(sam_coordinate(1,:) / wUnit);
        yBin = ceil(sam_coordinate(2,:) / hUnit);
        idxBin = (yBin - 1)*pyramid(i1) + xBin;
    
        for i2 = 1 : Num_Bins     
            cnt = cnt + 1;
            sidxBin = find( idxBin == i2 );
            if isempty(sidxBin)
                continue;
            end      
            % beta( :, cnt ) = max( blkwise_fea( :, sidxBin ), [], 2 );
            beta( :, cnt ) = mean( blkwise_fea( :, sidxBin ), 2 );
            % beta( :, cnt ) = sum( blkwise_fea( :, sidxBin ), 2 );
        end
        
    end
    
end

