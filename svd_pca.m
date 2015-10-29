% Weighted sample mean:
mu = classreg.learning.internal.wnanmean(x, vWeights);

case 'svd' % Use SVD to compute
    % Center the data if 'Centered' is true.
    if vCentered
        x = bsxfun(@minus,x,mu);
    end
    
    [U,sigma, coeff, wasNaN] = localSVD(x, n,...
        vEconomy, vWeights, vVariableWeights);
    if nargout > 1
        score =  bsxfun(@times,U,sigma');
        latent = sigma.^2./DOF;
        if nargout > 3
            tsquared = localTSquared(score,latent,DOF,p);
        end
        %Insert NaNs back
        if any(wasNaN)
            score = internal.stats.insertnan(wasNaN, score);
            if nargout >3
                tsquared = internal.stats.insertnan(wasNaN,tsquared);
            end
        end
    end
    
    if DOF < p
        % When 'Economy' value is true, nothing corresponding to zero
        % eigenvalues should be returned.
        if vEconomy
            coeff(:, DOF+1:end) = [];
            if nargout > 1
                score(:, DOF+1:end)=[];
                latent(DOF+1:end, :)=[];
            end
        elseif nargout > 1
        % otherwise, eigenvalues and corresponding outputs need to pad
        % zeros because svd(x,0) does not return columns of U corresponding
        % to components of (DOF+1):p.
            score(:, DOF+1:p) = 0;
            latent(DOF+1:p, 1) = 0;
        end
    end
    
%----------------Subfucntions--------------------------------------------

function [U,sigma, coeff, wasNaN] = localSVD(x, n,...,
    vEconomy, vWeights, vVariableWeights)
% Compute by SVD. Weights are supplied by vWeights and vVariableWeights.

% Remove NaNs missing data and record location
[~,wasNaN,x] = internal.stats.removenan(x);
if n==1  % special case because internal.stats.removenan treats all vectors as columns
    wasNaN = wasNaN';
    x = x';
end

% Apply observation and variable weights
vWeights(wasNaN) = [];
OmegaSqrt = sqrt(vWeights);
PhiSqrt = sqrt(vVariableWeights);
x = bsxfun(@times, x, OmegaSqrt');
x = bsxfun(@times, x, PhiSqrt);
    
if vEconomy
    [U,sigma,coeff] = svd(x,'econ');
else
    [U,sigma, coeff] = svd(x, 0);
end

U = bsxfun(@times, U, 1./OmegaSqrt');
coeff = bsxfun(@times, coeff,1./PhiSqrt');

if n == 1     % sigma might have only 1 row
    sigma = sigma(1);
else
    sigma = diag(sigma);
end
end


function tsquared = localTSquared(score, latent,DOF, p)
% Subfunction to calulate the Hotelling's T-squared statistic. It is the
% sum of squares of the standardized scores, i.e., Mahalanobis distances.
% When X appears to have column rank < r, ignore components that are
% orthogonal to the data.

if isempty(score)
    tsquared = score;
    return;
end

r = min(DOF,p); % Max possible rank of x;
if DOF > 1
    q = sum(latent > max(DOF,p)*eps(latent(1)));
    if q < r
        warning(message('stats:pca:ColRankDefX', q)); 
    end
else
    q = 0;
end
standScores = bsxfun(@times, score(:,1:q), 1./sqrt(latent(1:q,:))');
tsquared = sum(standScores.^2, 2);
end


