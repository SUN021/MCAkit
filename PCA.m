function [Fhat,Lhat]=PCA(X,r)

% X is the T by N matrix of observed variables 
% r is the number of estimated number of factors
% Fhat is the T by r matrix of estimated PCA factors
% Lhat is the N by r matrix of estimated PCA factor loadings

[T,N]=size(X);
[Y,W,V]=svd(X);
Fhat= Y(:,1:r)*sqrt(T);
Lhat= X' * Fhat/T;
%[Fhat,Lhat]=normalize(Fhat1,Lhat1);