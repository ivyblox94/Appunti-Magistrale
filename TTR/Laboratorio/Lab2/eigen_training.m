function [U,lambda]=eigen_training(A)

M                   =   size(A,2);
N                   =   size(A,1);
L                   =   A'*A;
% computing eigenvalues of A'A
[vettori,valori]    =   eig(L);
valori              =   diag(valori);

[lambda, ind]    =   sort(valori,'descend');
vettori          =   vettori(:,ind);

%getting back the M eigenvectors of AA'
U                   =   A*vettori;

%In order to be sure to have vectors of norm 1 
for i=1:M
    U(:,i)=U(:,i)/norm(U(:,i));
end

