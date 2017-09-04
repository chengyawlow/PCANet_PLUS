
function [ MAT ] = imgcell2mat( X, X_H, X_W )

    N = numel(X); 
    MAT = zeros( X_H * X_W, N );
    
    for i = 1 : N
        MAT( :, i ) = reshape( X{i}, X_H * X_W, 1 );
    end

end


