
% UPDATED ON APRIL 22, 2015

function [ recogRate ] = recognitionRate_CosineDistance( GA, PB, USR_ID_GA, USR_ID_PB )

    % GA = full(GA);
    % PB = full(PB);
    
    distCosine = pdist2( PB', GA', 'Cosine' );
    assert( size(distCosine, 1) == numel(USR_ID_PB) );
    assert( size(distCosine, 2) == numel(USR_ID_GA) );
        
    % Calculate recogRate
    [ ~, USR_ID_PB_EST ] = min( distCosine, [], 2 );
         
    USR_ID_PB_ORI = USR_ID_PB;
    USR_ID_PB_EST = USR_ID_GA(USR_ID_PB_EST); 
           
    recogRate = sum( USR_ID_PB_ORI == USR_ID_PB_EST ) / numel( USR_ID_PB ) * 100;
    
    %% Clear all, except recogRate
    clearvars -except recogRate;
    pause(0.001);
                
end


