% Exercize 11 Face recognition by eigenfaces
% Design a recognition system which does the following things:
% 1) It takes a gallery set of identities (The 41 identities that we have, 10 images for each identity)
% 2) It pulls out from these identities a single image (for each identity)
% that will be used as test image afterwards
% 3) It learns eigenfaces with the remaining "gallery" images (9 per identity), and projects the gallery in the few dimensional
% space
% 4) For each test or "probe" image, apply the eigenface recognition algorithm
% 5) Evaluate the overall accuracy
% 6) Repeat the previous steps, by changing the test image, 10 times in
% total

% Note: for classifying an image, use the KNN algorithm, with a number of
% neighbors at your choice

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
label  = reshape(repmat([1:41],10,1),410,1);
num_instances = 10;
confmat = zeros(10,41); % confusion matrix
num_iter = num_instances;
test_i = 0;
for iter = 1:1
    probe_i = 401:410;
    probe = TMP(:,probe_i);
    newTMP = TMP;
    newTMP(:,probe_i) = [];
    newlabel = label;
    newlabel(probe_i) = [];
    media                   =   mean(newTMP,2);
    newA                    =   newTMP-repmat(media,1,size(newTMP,2));
    [U,lambda]              =   eigen_training(newA);
    % Deciding the number of components to be taken
    T           =   10;
    omega_g       =   U(:,1:T)'*newA; % projection of the gallery;
    omega_p     =   U(:,1:T)'*(probe-repmat(media,1,10)); % projection of the probe
    num_neigh = 3;
    for j = 1:10
        test_i = test_i + 1;
        %Non valuto il più vicino ma i k più vicini. Quindi avendone 10 per
        %ogni classe mi auguro che i primi k vicini siano tutti uguali
        class = knnclassify(omega_p(:,j)',omega_g',newlabel,num_neigh);
        subplot(211); imagesc(reshape(TMP(:,probe_i(j)),r,c)); title('probe');...
        colormap gray, axis image;
        figure(hrec); subplot(212); imagesc(reshape(TMP(:,class*10-1),r,c));colormap gray, axis image;
        accuracy(test_i)=class==j;
        title(strcat('matched?:',num2str(accuracy(test_i))))
        fprintf('run %i|%i: accuracy = %i\n',test_i,M,accuracy(test_i));
        confmat(j,class)=confmat(j,class)+1;
       drawnow
    pause(0.5);  
    end
end
sum(accuracy)/410
