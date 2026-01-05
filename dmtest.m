function [dm_stat, p_value, h] = dmtest(e1, e2, h_max, type)
% DMTEST - Diebold-Mariano test for comparing predictive accuracy
%
% Input arguments:
%   e1     - forecast error vector from model 1
%   e2     - forecast error vector from model 2
%   h_max  - maximum lag order (optional, default is automatically selected)
%   type   - loss function type:
%            type = 1: squared error loss
%            otherwise: absolute error loss
%
% Output arguments:
%   dm_stat - DM test statistic
%   p_value - p-value of the test
%   h       - hypothesis test result (1 = reject the null, 0 = do not reject)

% Check input arguments
if nargin < 3 || isempty(h_max)
    h_max = max(1, floor(length(e1)^(1/3))); % default lag order
end

if length(e1) ~= length(e2)
    error('The error vectors must have the same length.');
end

% Remove NaN values
valid_idx = ~isnan(e1) & ~isnan(e2);
e1 = e1(valid_idx);
e2 = e2(valid_idx);

T = length(e1);
if T < 10
    error('Sample size is too small; at least 10 observations are required.');
end

% Compute loss differential series
if type == 1
    % Squared error loss
    d = e1.^2 - e2.^2;
else
    % Absolute error loss
    d = abs(e1) - abs(e2);
end

% Compute sample mean of loss differential
d_bar = mean(d);

% Compute sample variance (using 1/n normalization)
gamma_0 = var(d, 1);

if h_max == 0
    % Standard DM test without autocorrelation adjustment
    var_d = gamma_0 / T;
else
    % Newey-West type variance estimation
    % Compute autocovariances
    gamma_sum = 0;
    for h = 1:h_max
        if h < T
            % Autocovariance at lag h
            gamma_h = sum((d(1:T-h) - d_bar) .* (d(1+h:T) - d_bar)) / T;
            % Bartlett weight
            weight = 1 - h / (h_max + 1);
            gamma_sum = gamma_sum + 2 * weight * gamma_h; % factor 2 due to symmetry
        end
    end
    
    % Long-run variance
    long_run_var = gamma_0 + gamma_sum;
    
    % Ensure positivity of variance
    if long_run_var <= 0
        long_run_var = gamma_0;
        warning('Non-positive long-run variance detected; using sample variance instead.');
    end
    
    var_d = long_run_var / T;
end

% Compute DM statistic
dm_stat = d_bar / sqrt(var_d);

% Compute one-sided p-value
% p_value = 2 * (1 - normcdf(abs(dm_stat))); % two-sided version
p_value = 1 - normcdf(dm_stat);

% Hypothesis test result (5% significance level)
h = p_value < 0.05;

fprintf('DM statistic: %.4f, p-value: %.4f, lag order: %d\n', ...
        dm_stat, p_value, h_max);

end
