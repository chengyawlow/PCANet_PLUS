    
function FERET_PCANet_PLUS_FilterBank_MAIN_ONLINE( PCANet, X_TR, X_HEIGHT, X_WIDTH, PCANet_FILTERS_TAG )
  
    % close all; 
    % clc;
    
    % fprintf('\n');
    % fprintf(' -----------------------------------------------------------\n');
    % fprintf('          FERET_PCANet_PLUS_FilterBank_MAIN_ONLINE          \n');
    % fprintf(' -----------------------------------------------------------\n');
    
    %% Load FERET - TR
    % load ( 'FERET_I_128_128.mat ' );
    
    % X_TR = train.X ./ 255;
    % Y_TR = train.y;
    % X_HEIGHT = 128;
    % X_WIDTH = 128;
   
    %% Convert columnar X_TR into cells 
    X_TR = mat2imgcell( X_TR, X_HEIGHT, X_WIDTH, 'gray' );
        
    X_TR_MAX = numel( X_TR );
                
    clear fa fb fc dup1 dup2;
    pause( 0.001 );
    
    %% Learn PCA Filters from X_TR with respect to PCANet Parameters
    V = cell( PCANet.NumStages, 1 ); 
   
    for Stage_ID = 1 : PCANet.NumStages
    
        fprintf( '\n' );
        fprintf( '***** LEARNING PCA FILTERS FOR STAGE ID : %d *****', Stage_ID );
        fprintf( '\n' ); 
        
        % Validate X_TR, with respect to Stage_ID
        assert( numel( X_TR ) == X_TR_MAX );

        % Learn PCA filter banks, with respect to Stage_ID & PCANet parameters
        V{ Stage_ID } = PCANet_PLUS_LearnFilterBank_ONLINE( X_TR, PCANet.PatchSize( Stage_ID ), PCANet.NumFilters( Stage_ID ), PCANet );
        
        % Estimate the PCA feature maps only, if it is NOT the last Stage_ID
        if Stage_ID ~= PCANet.NumStages 
            
            % fprintf( '\n' );
            fprintf( 'STAGE %d : EXTRACTING PCANet FEA. MAPS ... ', Stage_ID );
            fprintf( '\n' ); 
                       
            [ X_TR ] = PCANet_PLUS_Output_ONLINE( X_TR, PCANet.PatchSize( Stage_ID ), PCANet.NumFilters( Stage_ID ), V{ Stage_ID }, PCANet );
            
            if PCANet.NL ~= 0
                [ X_TR, ~ ] = PCANet_PLUS_NonLinearity_ONLINE( X_TR, PCANet, Stage_ID );
            end

        end
        
    end

    %% Save PCA Filters, V    
    % fprintf( '\n' ); 
    % fprintf( 'SAVING PCANet FILTER ENSEMBLES ... ' );
    % fprintf( '\n' ); 
       
    save( [ PCANet_FILTERS_TAG, '.mat' ], 'PCANet', 'V' ); 
             
    %% Clear all
    clear all;
    
end

