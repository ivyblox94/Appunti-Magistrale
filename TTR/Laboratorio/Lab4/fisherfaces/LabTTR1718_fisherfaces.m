% Lab session TTR  -19/4/2017- Fisherfaces - Marco Cristani
 
clear all;
dire    =    '.\images'; % ORL face database
list    =   dir(strcat(dire,'\*.bmp'));
M       =   size(list,1);
tmp     =   imread(strcat(dire,'\',list(1).name));
[r,c]   =   size(tmp);
TMP     =   zeros(r*c,M);
for i=1:M
    tmp = imread(strcat(dire,'\',list(i).name));
    TMP(:,i)= tmp(:);
end
TMP                     =   double(TMP);
media                   =   mean(TMP,2);
AA(:,:)                 =   TMP-repmat(media,1,M);
[U,lambda]              =   eigen_training(AA);

T           =   200;
X           =   U(:,1:T)'*AA; % projection;
l           =   reshape(repmat([1:40],10,1),400,1);

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
% 4. determino le within class scatter
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
A = V(:,1:39);

% 6: transformation
Y = A'*X;

% 7: plot
figure, scatter(Y(1,:),Y(2,:),[],l)
for i=1:M
    text(Y(1,i),Y(2,i),num2str(l(i)))
end



