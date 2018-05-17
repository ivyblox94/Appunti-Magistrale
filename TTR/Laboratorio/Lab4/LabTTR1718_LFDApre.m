% Lezione di lab TTR  -24/04/2017- LDA  - Marco Cristani

% Code the FLDA algorithm for the multiclass case.
% Apply FLDA as we have seen previously with PCA on the IRIS dataset, 
% showing the first two extracted features. Compare with PCA.






clear all
close all
load 'irisSet.mat'

[d,N] = size(X); %number dim/feat, number points
% 1: mean
u = mean(X,2);
% 2: centering
h = ones(1,N);
B = X-u*h; % translate the points by the means

% 3: covariance matrix
C = 1/(N-1) * B*B'; 

% 4: eigenvalues/eigenvectors
[V,D] = eig(C);

% 5: sort eigenvalues
D=diag(D);
[D,ind]=sort(D,'descend');
V = V(:,ind); % V contains the e's organized by columns (the first column 
%               has the "largest" eigenvector and so on so forth)

% Two dimensions
% 6: transform matrix
V12 = V(:,1:2);
% 7: transformation
W = V12'*B;
% 8: plot on two dimensions
figure, scatter(W(1,:),W(2,:),[],l)
set(gcf,'name','PCA 2D projection');

% Apply LFDA
clear all
load 'irisSet.mat'

[d,N] = size(X);
K = max(l); % numero classi in gioco;

% 1. determino le classi Ck
for k = 1:K
    a = find (l == k);
    Ck{k} = X(:,a);
end
% 2. determino le medie
for k = 1:K
    mk{k} = mean(Ck{k},2);
end
% 3. determino la numerosità della classe
for k = 1:K
    [d, Nk(k)] = size(Ck{k});
end
% 4. determino le within class covariance
for k = 1:K
    S{k} = 0;
    for i = 1:Nk(k)
        S{k} = S{k} + (Ck{k}(:,i)-mk{k})*(Ck{k}(:,i)-mk{k})';
    end
    S{k} = S{k}./Nk(k);
end
Swx = 0;
for k = 1:K
    Swx = Swx + S{k};
end

% 5. determino la between class covariance
% 5.1 determino la media totale
m = mean(X,2);
Sbx = 0;
for k=1:K
    Sbx = Sbx + Nk(k)*((mk{k} - m)*(mk{k} - m)');
end
Sbx = Sbx/K;

MA = inv(Swx)*Sbx;

% eigenvalues/eigenvectors
[V,D] = eig(MA);

% 5: transform matrix
W = V(:,1:2);

% 6: transformation
Y = W'*X;

% 7: plot
figure, scatter(Y(1,:),Y(2,:),[],l)
set(gcf,'name','LFDA 2D projection');

