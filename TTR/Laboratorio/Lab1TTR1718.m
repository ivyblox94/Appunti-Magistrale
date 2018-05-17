% Lezione di lab TTR  -14/3/2018- Legge di Bayes - Marco Cristani
close all;
clear all;
clc
% ---------------
% Exercize 1
% ---------------
fprintf(' ---------------\n')
fprintf(' Exercize 1\n')
fprintf(' ---------------\n')
%-- sample visualization
mu1         =   1;%Poiché mu e sigma sono numeri, siamo in dimensionalità 1, i.e. identifichiamo dei valori che stanno sulla retta
sigma1      =   2;

mu2         =   20;
sigma2      =   3;

num_s       =   5000;% #campioni

%fprintf('generating 1D samples...');
[Sample_set1] =   normrnd(mu1,sigma1,num_s,1); %"generami dei campioni con questa distribuzione e mettimi i risultati in un vettore 1xnum_s"
[Sample_set2] =   normrnd(mu2,sigma2,num_s,1);

%fprintf('OK.\n');

%Come fare intuitivamente a capire che sono distribuiti in modo normale?
%Con l'istogramma
[histo1,c1]=hist( Sample_set1,32);
[histo2,c2]=hist( Sample_set2,32);

h0=figure;
set(gcf,'name','Histogram visualization')
subplot(211); bar(c1,histo1);
subplot(212); bar(c2,histo2);


Data        =   [Sample_set1;Sample_set2];

%-- Plot campioni

maxData     =   max(Data);
minData     =   min(Data);


h_G1d = figure; hold on;
h_S1 = scatter(Sample_set1,zeros(num_s,1),10,'b');
h_S2 = scatter(Sample_set2,zeros(num_s,1),10,'r');



axis([minData-1,maxData+1,0,2])
set(gcf,'name','Campioni Gaussiani')
set(gca,'ytick',[]);
h_m1 = line([mu1;mu1],[0;2]); 
set(h_m1,'color',[0,0,1]);
text(mu1+0.05,1,'\mu1','color',[0,0,1])

h_m2 = line([mu2;mu2],[0;2]);
set(h_m2,'color',[1,0,0]);
text(mu2+0.05,1,'\mu2','color',[1,0,0])

h_leg = legend([h_S1(1);h_S2(1);h_m1;h_m2],{strcat('N1: \mu1=',num2str(mu1),' \sigma1=',num2str(sigma1));...
    ,strcat('N2: \mu2=',num2str(mu2),' \sigma2=',num2str(sigma2));...
    '\mu1';...
    '\mu2'});


fprintf('mu1 = %3.5f; dev-std1 =%3.5f\n',mu1,sigma1)
fprintf('mu1 = %3.5f; dev-std1 =%3.5f\n',mu2,sigma2)

%-- Mean and standard deviation estimation

%fprintf('Estimate of mean and std...')
est_mu1 = mean(Sample_set1);
est_sigma1 = std(Sample_set1);

est_mu2 = mean(Sample_set2);
est_sigma2 = std(Sample_set2);
%fprintf('OK.\n');


fprintf('estimated mu1 = %3.5f; estimated dev-std1 =%3.5f\n',est_mu1,est_sigma1)
fprintf('estimated mu2 = %3.5f; estimated dev-std2 =%3.5f\n',est_mu2,est_sigma2)

%-- Plot of the estimates 
set(h_G1d,'name','Campioni Gaussiani, confronto medie reali e stimate')
h_estm1 = line([est_mu1;est_mu1],[0;2]); 
set(h_estm1,'color',[0,0,1],'LineStyle','--');
text(est_mu1+0.05,1.5,'est. \mu1','color',[0,0,1])

h_estm2 = line([est_mu2;est_mu2],[0;2]);
set(h_estm2,'color',[1,0,0],'LineStyle','--');
text(est_mu2+0.05,1.5,'est. \mu2','color',[1,0,0])

delete(h_leg)

h_leg = legend([h_S1(1);h_S2(1);h_m1;h_m2;h_estm1;h_estm2],{strcat('N1: \mu1=',num2str(mu1),' \sigma1=',num2str(sigma1));...
    ,strcat('N2: \mu2=',num2str(mu2),' \sigma2=',num2str(sigma2));...
    ' \mu1';...
    ' \mu2';...
    'est. \mu1';...
    'est. \mu2'});

%-- Errore stime
%fprintf('Calcolo errore stime...');
err_mu1 = abs(est_mu1-mu1);
err_sigma1 = abs(est_sigma1-sigma1);

err_mu2 = abs(est_mu2-mu2);
err_sigma2 = abs(est_sigma2-sigma2);
%fprintf('OK \n');

fprintf('mu1 estimation error = %3.5f; dev-std1 estimation error =%3.5f\n',err_mu1,err_sigma1);
fprintf('mu2 estimation error  = %3.5f; dev-std2 estimation error  =%3.5f\n',err_mu2,err_sigma2);

fprintf('END! Press a button to continue!\n')
% pause

% ---------------
% Exercise 2
% ---------------
% I dati vengono presi come l'insieme del set 1 e 2
fprintf(' ---------------\n')
fprintf(' Exercise 2\n')
fprintf(' ---------------\n')

Data        =   [Sample_set1;Sample_set2];

% compute the likelihhod w.r.t. a given parametrization
mu3 = 10.5; %10.5;
dev_std3 = 9.8; %5;

lik_ind3 = normpdf(Data,mu3,dev_std3);%In matlab rappresenta la likelihood, 
%i.e. P(x|w) con w=<mu, sigma> e x distribuita come N(mu, sigma).

h3 = figure;
scatter(Data,lik_ind3);

lik_tot3 = prod(lik_ind3); % very probably, I will get 0 from this. Why?

llik_tot3 = sum(log(lik_ind3));

%Now, try to fit the data with a single Gaussian distribution that fits the
%best the data. What is this number (there is no correct choice, this exercize
% will make you able to reason in terms of log likelihood probabilities)

% In other to perform a more systematic analysis, let's compute a double
% for looking for the best pair of parameters for my data
itmu = 0;
for mu3=-5:0.1:20
    itmu = itmu +1;
    itsigma = 0;
    for dev_std3=0.1:0.1:20
        itsigma = itsigma +1;
        lik_ind3 = normpdf(Data,mu3,dev_std3);
        llik_tot3 = sum(log(lik_ind3));
        res(itmu,itsigma)=llik_tot3 ;
        mu(itmu) = mu3;
        sigma(itsigma) = dev_std3;
    end
end
figure; imagesc(res);
[llmax, indmax] = max(res(:))
[I,J] = ind2sub(size(res),indmax);
maxmean = mu(I);
maxsigma = sigma(J);
h3 = figure;
lik_ind3 = normpdf(Data,maxmean,maxsigma);
scatter(Data,lik_ind3);
% ---------------
% Exercise 3
% ---------------
fprintf(' ---------------\n')
fprintf(' Exercise 3\n')
fprintf(' ---------------\n')

%Normal distribution, multivariate
clear all;
A= normrnd(0,2,10000,2);%Sto utilizzando una distribuzione monovariata per 
%arrivare ad una distribuzione multivariata diagonale. In pratica posso
%usare normrnd perché considero che la matrice di covarianza abbia solo
%sigma^2 sulla diagonale.

h1 = figure;
scatter(A(:,1),A(:,2));
axis equal

% The points which have been sampled above are distributed following a
% particular multivariate normal distribution. Which are the parameters of
% this multivariate normal distribution (in terms of mean and covariance matrix)?
% mu = [mu1,mu2]; Sigma = [cov11,cov12; cov21,cov22];

Mu_est=mean(A); %Me lo aspetto simile a mu=0;
Cov_est=cov(A); %Me lo aspetto simile a sigma^2*I
% Injecting a small covariance between variables

%NB: se la covarianza è negativa vuol dire che mano a mano che uno cresce
%l'altro decresce, il contrario se è positiva, se è 0 è indipendente.
A2dc = mvnrnd([0,0],[4,-0.9;-0.9,4],10000);
h2 = figure;
scatter(A2dc(:,1),A2dc(:,2));
axis equal
% Se A=[1 0]^T e voglio usarla come proiezione, allora se proietto N(mu,
% sigma) ottengo N(A*mu, A^T*Sigma*A)=N(0,4) e ottengo una distribuzione
% monodimensionale.

%Noto che se uso come matrice A2 anziché A, A^T*Sigma*A=4 sempre, in quanto
%la covarianza mi dice solo come sono distribuiti tra loro i dati e non
%come sono distribuiti in funzione della media. 
pause

% ---------------------------------------
% Esercize 4 - face recognition
% ---------------------------------------
fprintf(' ---------------\n')
fprintf(' Esercize 4\n')
fprintf(' ---------------\n')

%The goal will be that of instantiate two gaussian classifiers which will
%be used to classify test images, guessing one of the two labels: male or
%female, in a maximum a posteriori sense.

A=rgb2gray(imread('1.jpg'));

%Ho ridotto le dimensioni dell'immagine in una 148x224. Ho 7 immagini per
%lui, 5 per lei. Se rappresentassi in uno spazio multidimensionale i punti,
%ovvero in R^148x224=R^33K, posso augurarmi che siano clusterizzati in
%sottoinsiemi diversi del multispazio.

%Se cercassi la gaussiana del tizio, essa sarebbe N_tizio=<mu, Sigma> con
%mu 33Kx1, Sigma 33Kx33K. Posso vagamente ottimizzare facendo Sigma_tilde
%matrice diagonale -> N_tizio=<mu, Diag*I> con Diag 1x33K.

%Voglio visualizzare i parametri -> li tratto come immagini
A=A(1:4:end,1:4:end);
rows = size(A,1);
cols = size(A,2); 

% I can downsample my images, but still the parameter space of the gaussian
% distribution will be huge (530432). I simplify the problem, making the
% covariance matrix as diagonal. IN that case, the cavariance matrix would
% have 530432 non zero elements

figure; 
set(gcf,'name','Mean and variance, fast approach')
colormap gray; subplot 221 ; imagesc(A); 
title('Mean class 1');
Mu1 = A(:);%La media è un'immagine;
Sigma1 = ones(rows*cols,1).*10;
subplot 222 ; imagesc(reshape(Sigma1,rows,cols)); colorbar; 
title('Sigma class 1');

B=rgb2gray(imread('12.jpg'));
B=B(1:4:end,1:4:end);
subplot 223; imagesc(B); 
title('Mean class 2');
Mu2 = B(:);
Sigma2 = ones(rows*cols,1).*10;
subplot 224 ; imagesc(reshape(Sigma2,rows,cols)); colorbar; 
title('Sigma class 2');

% Now, the task would be that of classifying the remaining images using
% maximum a posteriori (...) law. The problem is that we do not have the a
% priori probability, so we will use just the likelihood probability
% (implying to have flat prior distribution, so 0.5 for each class)

lista = dir('*.jpg');
numelem = length(lista);
C1=[];
C2=[];
good = setdiff([1:numelem],[1,12]);
for i=good
    test = rgb2gray(imread(strcat(num2str(i),'.jpg')));
    test = test(1:4:end,1:4:end);
    test=test(:);
    LK1 = sum(log(normpdf(double(test),double(Mu1),double(Sigma1))));
    LK2 = sum(log(normpdf(double(test),double(Mu2),double(Sigma2))));
    if LK1 >LK2
        C1 = [C1,i];
    else
        C2 = [C2,i];
    end
end


latoC1=ceil(sqrt(length(C1)));
latoC2=ceil(sqrt(length(C2)));
figure;
set(gcf,'name','Class 1')
count=1;
for i=1:length(C1)
    
    subplot(latoC1,latoC1,i);
    test = rgb2gray(imread(strcat(num2str(C1(i)),'.jpg')));
    imagesc(test);axis off; colormap gray;
    
end

figure;
set(gcf,'name','Class 2')
count=1;
for i=1:length(C2)
    
    subplot(latoC2,latoC2,i);
    test = rgb2gray(imread(strcat(num2str(C2(i)),'.jpg')));
    imagesc(test);axis off; colormap gray;
    
end    

% ---------------------------------------
% Esercize 5 - computing accuracy
% ---------------------------------------
fprintf(' ---------------\n')
fprintf(' Esercize 5\n')
fprintf(' ---------------\n')

% At this point, we have done the classification, and we need to compute
% how "good" we have been in classifying. To this sake, we use the accuracy
% measure, which is =(N. correct class./ N classifications)
label = [1 1 1 1 1 1 1 2 2 2 2 2];
countc = 0;
count = length(label)-2;
for i=C1;
    if label(i)==1
        countc = countc + 1;
    end
end

for i=C2;
    if label(i)==2
        countc = countc + 1;
    end
end

accuracy = countc/count;


% -------------------------------------------------
% Exercise 6 - "robust" face recognition
% -------------------------------------------------
fprintf(' ---------------\n')
fprintf(' Exercise 6\n')
fprintf(' ---------------\n')

Vol1 = [];
Vol2 = [];

train1 = [1:5];
train2 = [8:10];
traintot = [train1,train2];

for i=train1
    train = rgb2gray(imread(strcat(num2str(i),'.jpg')));
    train = train(1:4:end,1:4:end);
    Vol1 = [Vol1,train(:)];
end
for i=train2
    train = rgb2gray(imread(strcat(num2str(i),'.jpg')));
    train = train(1:4:end,1:4:end);
    Vol2 = [Vol2,train(:)];
end


Eff_mean1 = mean(Vol1,2);
Eff_sigma1 = std(single(Vol1)');
Eff_mean2 = mean(Vol2,2);
Eff_sigma2 = std(single(Vol2)');

h_Mean_Std = figure;
set(gcf,'name','mean and variance, robust approach')
colormap gray;
subplot 221; imagesc(reshape(Eff_mean1,rows,cols)); title('Mean class 1'); 
subplot 222; imagesc(reshape(Eff_mean2,rows,cols)); title('Mean class 2'); 
subplot 223; imagesc(reshape(Eff_sigma1,rows,cols)); title('Std class 1'); colorbar
subplot 224; imagesc(reshape(Eff_sigma2,rows,cols)); title('Std class 2'); colorbar

 
C1=[];
C2=[];
testtot = setdiff([1:12],traintot);
for i=testtot
    test = rgb2gray(imread(strcat(num2str(i),'.jpg')));
    test = test(1:4:end,1:4:end);
    test = test(:);
    LK1 = sum(log(normpdf(double(test),double(Eff_mean1),double(Eff_sigma1'+eps))));
    LK2 = sum(log(normpdf(double(test),double(Eff_mean2),double(Eff_sigma2'+eps))));
    if LK1 >LK2
        C1 = [C1,i];
    else
        C2 = [C2,i];
    end
    i
end
latoC1=ceil(sqrt(length(C1)));
latoC2=ceil(sqrt(length(C2)));
figure;
set(gcf,'name','C1 classification, new mean estimation')
for i=1:length(C1)
    
    subplot(latoC1,latoC1,i);
    test = rgb2gray(imread(strcat(num2str(C1(i)),'.jpg')));
    imagesc(test);axis off; colormap gray;
    
end


figure;
set(gcf,'name','C2 classification, nuova mean estimation')
for i=1:length(C2)
    subplot(latoC2,latoC2,i);
    test = rgb2gray(imread(strcat(num2str(C2(i)),'.jpg')));
    imagesc(test);axis off; colormap gray;
end   

% -------------------------------
% Esercizio 3 - BG subtraction
% -------------------------------
fprintf(' ---------------\n')
fprintf(' Exercize 5\n')
fprintf(' ---------------\n')
vidObj = VideoReader('piramide.avi');
C_info = aviinfo('piramide.avi');
C = rgb2gray(read(vidObj,91));
figure; subplot(211); imagesc(C); colormap gray; axis off; set(gcf,'name','Sottrazione del background: stima grezza della scena');
subplot(212); imagesc(C); colormap gray; axis off;
Mu_BG = double(C(:));
sigma_BG = 16;
figure; set(gcf,'name','Sottrazione del background');
for t=1:C_info.NumFrames
    test = rgb2gray(read(vidObj,t));
    res  = test(:)<Mu_BG-3*sigma_BG | test(:)>Mu_BG+3*sigma_BG;
    g(t)= test(151,45);
    subplot(211);
    imagesc(reshape(res,C_info.Height,C_info.Width)); axis image;
    colormap gray;
    subplot(212);
    imagesc(reshape(uint8(res),C_info.Height,C_info.Width).*...
        test); axis image; pause(0.05); 
end
% Draw the signal which represents a given pixel (you select) through time
figure;
plot(g);
set(gcf,'name','Time evolution of a given pixel');
axis([0,91,0,255])
% The exercise here allows you to sense how the foreground classification
% does operate. When the pixel signal is changing beyond its gaussian
% distribution, the threshold operation is switched on, and the pixel is
% considered as foreground

% Q: Considering the videosequence above, how it could be possible to push the
% foreground silhouettes to be more compact (no hole inside)?
% The answer regards just two numbers to be changed
% A: You need to change the "3"s, which are the span of the gaussian
% distribution I check fro outliers (which is my foreground). If I
% decrement such number I will be more sensible (but also more noisy, considering false positives)





% ---------------------------------------------------------------------------------
% Exercize 4 - Gaussian manipulations, 1D and ND
% ---------------------------------------------------------------------------------

% The first Gaussian
clear all;
A= normrnd(0,2,10000,2);
h1 = figure;
scatter(A(:,1),A(:,2));
axis equal

% The second Gaussian
A2d = mvnrnd([0,0],[4,0;0,4],10000);
h2 = figure;
scatter(A2d(:,1),A2d(:,2));
axis equal

% Q: What is the formal difference between these two distribustions? 
% A: They are equal.... understand the why...



% Injecting a small covariance
A2dc = mvnrnd([0,0],[1,-sqrt(2)/2;-sqrt(2)/2,1],10000);
h2 = figure;
scatter(A2dc(:,1),A2dc(:,2));
axis equal

cova = cov(A2dc);
[v,a]=eig(cova); % che cosa risulta, senza guardare il risultato?

% ESERCIZIO: Provate con distribuzioni 3D, visualizzate

% -----------------------------------------------
% Esercizio 8 - Funzioni discriminanti lineari
% -----------------------------------------------


num_s=100;

Pw1=0.5;
Pw2=0.5;
%Noto che se diminuisco la probabilità a priori della 1, la mia soglia si
%avvicina di più alla media della prima perché sarà più difficile che un
%elemento appartenga a quella classe; al contrario se la probabilità più
%alta è la prima. Se Pw1==Pw2 la soglia sarà esattamente nella media delle
%medie.
Mu1 = 3;%
Mu2 = 0;%

sigma=1.5;

Sample_set1 =   normrnd(Mu1,sigma,num_s,1);
Sample_set2 =   normrnd(Mu2,sigma,num_s,1);

Data = [Sample_set1 ; Sample_set2];
% 
% 
% % Disegno le funzioni discriminanti per la due classi di campioni Gaussiani
% 
W1  = (1/sigma^2).*Mu1;
w10 = -(1/(2*sigma^2))*(Mu1'*Mu1) +log(Pw1);
G1 = sum(repmat(W1,1,num_s*2).*Data',1) + ones(1,num_s*2)*w10;


W2  = (1/sigma^2).*Mu2;
w20 = -(1/(2*sigma^2))*(Mu2'*Mu2) +log(Pw2);
G2 = sum(repmat(W2,1,num_s*2).*Data',1) + ones(1,num_s*2)*w20;
% 
% % Confine di decisione
W  = Mu1-Mu2;
X0 = 0.5*(Mu1+Mu2)-((sigma^2)./((Mu1-Mu2)'*(Mu1-Mu2))).*log(Pw1/Pw2).*(Mu1-Mu2);
% 
% 
h_CBayesG1D = figure; hold on;
[Ord,index_ord] = sort(Data,'ascend');
h_S1 = scatter(Sample_set1,zeros(num_s,1),10,'b');
h_S2 = scatter(Sample_set2,zeros(num_s,1),10,'r');
h_G1 = plot(Ord,G1(index_ord),'b');
h_G2 = plot(Ord,G2(index_ord),'r');
set(gcf,'name','Classificatore Bayesiano 2 classi 1D')
h_BayesG1D=line([X0;X0],[min([G1';G2']),max([G1';G2'])],'color',[0,0,0]);
%legend([h_G1;h_G2;h_BayesG1D],{'g_1(x)';'g_2(x)';'x_0'});
legend([h_S1(1);h_S2(1);h_G1;h_G2],{strcat('N1: \mu1=',num2str(Mu1),' \sigma1=',num2str(sigma));...
    strcat('N2: \mu2=',num2str(Mu2),' \sigma2=',num2str(sigma));...
    'g_1(x)';'g_2(x)'});

fprintf('FINE! \n')






