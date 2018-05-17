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
    T           =  360;% M-num_identities;
    omega_g       =   U(:,1:T)'*newA; % projection of the gallery;
    omega_p     =   U(:,1:T)'*(probe-repmat(media,1,num_identities)); % projection of the probe
    num_neigh = 1;
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