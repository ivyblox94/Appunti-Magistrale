% ---------------------------------------
% Esercize 6- face recognition
% ---------------------------------------
clc
clear all
close all

fprintf(' ---------------\n')
fprintf(' Esercize 6\n')
fprintf(' ---------------\n')

label = [1 1 1 1 1 1 1 2 2 2 2 2];
%The goal will be that of instantiate two gaussian classifiers which will
%be used to classify test images, guessing one of the two labels: male or
%female, in a maximum a posteriori sense.

trainSet1=[1:5];
Man = [];
trainSet2= [8:9];
Woman=[];

for i=trainSet1
    img = rgb2gray(imread(strcat(num2str(i),'.jpg')));
    img = img(1:4:end,1:4:end);
    Man=[Man,img(:)];
    
end

Mu1 = mean(Man,2);%La media è un'immagine;
Sigma1 = std(single(Man)');

for i=trainSet2
    img = rgb2gray(imread(strcat(num2str(i),'.jpg')));
        img = img(1:4:end,1:4:end);

    Woman=[Woman, img(:)];
    
end

Mu2 = mean(Woman,2);%La media è un'immagine;
Sigma2 = std(single(Woman)');

rows = size(img,1);
cols = size(img,2);

h_Mean_Std = figure;
set(gcf,'name','mean and variance, robust approach')
colormap gray;
subplot 221; imagesc(reshape(Mu1,rows,cols)); title('Mean class 1'); 
subplot 222; imagesc(reshape(Mu2,rows,cols)); title('Mean class 2'); 
subplot 223; imagesc(reshape(Sigma1,rows,cols)); title('Std class 1'); colorbar
subplot 224; imagesc(reshape(Sigma2,rows,cols)); title('Std class 2'); colorbar


C1=[];
C2=[];
testtot = setdiff([1:12],[trainSet1, trainSet2]);
for i=testtot
    test = rgb2gray(imread(strcat(num2str(i),'.jpg')));
        test = test(1:4:end,1:4:end);

    test = test(:);
    LK1 = sum(log(normpdf(double(test),double(Mu1),double(Sigma1'+eps))));
    LK2 = sum(log(normpdf(double(test),double(Mu2),double(Sigma2'+eps))));
    if LK1 >LK2
        C1 = [C1,i];
    else
        C2 = [C2,i];
    end
end
latoMan=ceil(sqrt(length(C1)));
latoWoman=ceil(sqrt(length(C2)));
figure;
set(gcf,'name','C1 classification, new mean estimation')
for i=1:length(C1)
    
    subplot(latoMan,latoMan,i);
    test = rgb2gray(imread(strcat(num2str(C1(i)),'.jpg')));
    imagesc(test);axis off; colormap gray;
    
end


figure;
set(gcf,'name','C2 classification, nuova mean estimation')
for i=1:length(C2)
    subplot(latoWoman,latoWoman,i);
    test = rgb2gray(imread(strcat(num2str(C2(i)),'.jpg')));
    imagesc(test);axis off; colormap gray;
end   


classif = label.*0;
classif(C1)=1;
classif(C2)=2;
goodtest = find(classif~=0);
confmat = zeros(2,2);
for j=1:length(goodtest)
    el=goodtest(i);
    confmat(label(el), classif(el))=confmat(label(el), classif(el)) + 1;
end