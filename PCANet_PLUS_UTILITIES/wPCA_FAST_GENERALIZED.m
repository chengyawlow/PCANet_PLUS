
% UPDATED ON OCTOBER 16, 2015

% REMARK : X_TR IS A COLUMNAR MATRIX, I.E., EACH COLUMN OF X_TR IS A FEATURE VECTOR
% wPCA_FAST_GENERALIZED FAST PCA, I.E., THE STANDARD TURK-PENTLAND EIGENFACES METHOD.
% projected onto the subDim-dimensional subspace found by PCA.
% 
% REFERENCE : http://www.face-rec.org/algorithms/
% SOURCE CODE : http://www.face-rec.org/source-codes/

function [ X_TR_MEAN, eigen_wPCA, eigen_PCA ] = wPCA_FAST_GENERALIZED( X_TR, kPCA, wPCA )
        
    %% Parameter Initialization
    % Turk-Pentland Trick's Validation
    [ numDim, numImg ] = size(X_TR);
    assert( numImg <= numDim );
    % sz_X_TR = size(X_TR) 
  
    % rglEpsilon = 0.000000005;
    rglEpsilon = 0.00001;

    %% Traditional PCA
    % Step 1: Formulate zero-mean X_TR
    X_TR_MEAN = mean(X_TR, 2);
    % X_TR_MEAN = zeros( size(X_TR, 1), 1 );
    assert(numel(X_TR_MEAN) == size(X_TR, 1));

    % B = repmat(A,M,N) creates a large matrix B consisting of an M-by-N
    % tiling of copies of A. The size of B is [size(A,1)*M, size(A,2)*N].
    X_TR = X_TR - repmat(X_TR_MEAN, 1, size(X_TR, 2));
    
    % Job Queue Interruption
    % pause(0.00001);
    
    % Step 2: Compute eigenFaces and eigenValues
    % [U, S, V] = svd(X, 0) produces the "economy size" decomposition. 
    % If X is m-by-n with m > n, then only the first n columns of U 
    % are computed and S is n-by-n. 
    % [ eigFaces_PCA, eigValues_PCA, ~ ] = svd(X_TR); 
    % [ eigFaces_PCA, eigValues_PCA, ~ ] = svdecon(X_TR); 
    % [eigFaces_PCA, eigValues_PCA ] = eig(cov(X_TR'));
    
    % TURK-PENTLAND TRICK-1
    % [ eigFaces_PCA, eigValues_PCA ] = eig( X_TR' * X_TR );
    [ eigFaces_PCA, eigValues_PCA, ~ ] = svd( X_TR' * X_TR );
    % [ eigFaces_PCA, eigValues_PCA, ~ ] = svds( X_TR' * X_TR, size( X_TR, 2 ) );
    [ eigValues_PCA, eigValues_IND ] = sort( abs(diag(eigValues_PCA)), 'descend' );
    
    % TURK-PENTLAND TRICK-2
    eigFaces_PCA = X_TR * eigFaces_PCA( :, eigValues_IND );
    % size(eigenFaces_PCA) = numDim_TR x numImg_TR
    eigFaces_PCA = eigFaces_PCA( :, 1 : kPCA );
    % Ensure eigFaces_PCA are of unit norm
    eigFaces_PCA = normc(eigFaces_PCA);
    eigFaces_PCA = eigFaces_PCA';
    assert(size(eigFaces_PCA, 1) == kPCA);
    
    % Return eigFaces_PCA as eigen_PCA
    % eigen_PCA = eigFaces_PCA;
    eigen_PCA = [];
    
    % size(eigenValues_PCA) = numImg_TR x numImg_TR
    % eigValues_PCA = diag(eigValues_PCA);
    % eigValues_PCA = diag(eigValues_PCA.^2);
    eigValues_PCA = eigValues_PCA( 1 : kPCA );
   
    %% Whitening PCA
    % X_TR_PCA = eigenFaces_PCA * X_TR;
    eigValues_wPCA = diag( 1 ./ sqrt( eigValues_PCA + rglEpsilon ) );
   
    if wPCA == 0
         eigValues_wPCA = 1;
    end
    
    % cov(X_TR_wPCA) = eyes(N)
    eigen_wPCA = eigValues_wPCA * eigFaces_PCA;
    
	%% Clear all, EXCEPT X_TR_MEAN, eigen_wPCA, eigen_PCA
	clearvars -except X_TR_MEAN eigen_wPCA eigen_PCA
         
end