
function [ X_PATCHES ] = im2col_BN_REG( X, PS, PCANet )

    %% Initialize parameters
    NUM_CHANNEL = size( X, 3 );
    
    BN = PCANet.BN;
    BN_REG = PCANet.BN_REG;
        
    X_PATCHES = [];
    
    %% Perform patch-mean removal, by image ( feature map ).    
    for CHANNEL_ID = 1 : NUM_CHANNEL
        
        X_PATCHES_TEMP = im2colstep( X( :, :, CHANNEL_ID ), PS );
        
        if BN == 1
            X_PATCHES_TEMP = bsxfun( @minus, X_PATCHES_TEMP, mean( X_PATCHES_TEMP, 1 ) );
        elseif BN == 2 
            if BN_REG == 0
                X_PATCHES_TEMP = zscore( X_PATCHES_TEMP );
            elseif BN_REG ~= 0
                X_PATCHES_MEAN = mean( X_PATCHES_TEMP, 1 );
                X_PATCHES_STD = std( X_PATCHES_TEMP, [], 1 );
                X_PATCHES_STD( X_PATCHES_STD <= BN_REG ) = 1;
                % X_PATCHES_STD = sqrt( var( X_PATCHES_TEMP, [], 1 ) + BN_REG );
                X_PATCHES_TEMP = bsxfun( @rdivide, bsxfun( @minus, X_PATCHES_TEMP, X_PATCHES_MEAN ), X_PATCHES_STD );
            end
        end
                       
        X_PATCHES = cat( 1, X_PATCHES, X_PATCHES_TEMP );
        
        clear X_PATCHES_TEMP;
        
    end
       
    %% Clear all, except X_PATCHES
    clearvars -except X_PATCHES;
    
end