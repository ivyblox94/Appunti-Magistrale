clear all
close all

dire    =    '.\images'; % ORL face database
list    =   dir(strcat(dire,'\*.bmp'));
M       =   size(list,1);
tmp     =   imread(strcat(dire,'\',list(1).name));
[r,c]   =   size(tmp);
TMP     =   zeros(r*c,M);
figure;
for i=1:M
    tmp = imread(strcat(dire,'\',list(i).name));
    TMP(:,i)= tmp(:);
end
TMP                     =   double(TMP);
hrec   = figure; set(gcf,'name','FACE RECOGNITION');
label  = reshape(repmat([1:40],10,1),400,1);
num_instances = 10;
num_identities = 40;
confmat = zeros(num_identities,num_identities); % confusion matrix
num_iter = num_instances;
test_i = 0;
for iter = 1:num_instances
    probe_i = iter:num_instances:M;
    probe = TMP(:,probe_i);
    newTMP = TMP;
    newTMP(:,probe_i) = [];
    newlabel = label;
    newlabel(probe_i) = [];
    media                   =   mean(newTMP,2);
    newA                    =   newTMP-repmat(media,1,size(newTMP,2));
    [U,lambda]              =   eigen_training(newA);
    % Deciding the number of components to be taken
    T           =  200;% M-num_identities;
    omega_g       =   U(:,1:T)'*newA; % projection of the gallery;
    omega_p     =   U(:,1:T)'*(probe-repmat(media,1,num_identities)); % projection of the probe
    %%%%% ADDING LFDA
    %     
    l           =   reshape(repmat([1:40],10,1),400,1);
    X = omega_g;
    [d,N] = size(X);
    K = max(newlabel); % numero classi in gioco;
    
    % 1. determino le classi Ck
    for k = 1:K
        a = find (newlabel == k);
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
    omega_g = A'*omega_g;
    omega_p = A'*omega_p;
    %%%%% END ADDING LFDA
    num_neigh = 5;
    for j = 1:num_identities
        test_i = test_i + 1;
        class = knnclassify(omega_p(:,j)',omega_g',newlabel,num_neigh);
        subplot(211); imagesc(reshape(TMP(:,probe_i(j)),r,c)); title('probe');...
        colormap gray, axis image;
        figure(hrec); subplot(212); imagesc(reshape(TMP(:,class*10-1),r,c));colormap gray, axis image;
        accuracy(test_i)=class==j;
        title(strcat('matched?:',num2str(accuracy(test_i))))
        fprintf('run %i|%i: accuracy = %i\n',test_i,M,accuracy(test_i));
        confmat(j,class)=confmat(j,class)+1;
       drawnow
    pause(0.02);  
    end
end
sum(accuracy)/400