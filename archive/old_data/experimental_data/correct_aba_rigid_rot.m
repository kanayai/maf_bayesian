function [coords_1_corr] = correct_aba_rigid_rot(coords_0,coords_1)
%CORRECT_ABA_RIGID_ROT  Summary of this function goes here
%   Detailed explanation goes here
%  Inputs:
%       coords_0: undeformed coordinates
%       coords_1: deformed coordinates
%  Outpus:
%       coords_1_corr: deformed coordinates corrected for rigid motion
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    c_0 = mean(coords_0);
    c_1 = mean(coords_1);
    
    coords_0_c = coords_0 - c_0;
    coords_1_c = coords_1 - c_1;

    S = coords_0_c' * coords_1_c;  % covariance matrix 
    [U, Sigma, V] =  svd(S);       % # singular value decomposition matrix
    R = V * U';                    % # rotation matrix

    T = coords_1(1,:)' - R * coords_0(1,:)'; %#transformation matrix
    
%     coords_1_corr = zeros(length(coords_0),3); % # initialise
%     for k = 1:length(coords_0)
%         coords_1_corr(k,:) = R \ (coords_1(k,:)' - T);
%     end

    coords_1_corr = (R \ (coords_1' - T))';
end

