% TTR Lab session -27/03/2018- EIGENFACES - Marco Cristani 
clear all;
close all;
% -------------------------------------------------------------
% Eigenfaces
% -------------------------------------------------------------

dire    =    '.\images'; % ORL face database (plus my images!!!)
list    =   dir(strcat(dire,'\*.bmp'));
M       =   size(list,1);
tmp     =   imread(strcat(dire,'\',list(1).name));
[r,c]   =   size(tmp);
TMP     =   zeros(r*c,M);
figure;
for i=1:M
    tmp = imread(strcat(dire,'\',list(i).name));
        imagesc(tmp); axis image; colormap gray;
        drawnow;
        pause(0.01);
        TMP(:,i)= tmp(:);
        mont(:,:,1,i)= tmp;
end
figure;
montage(mont);
set(gcf,'name','Gallery');

TMP                     =   double(TMP);
media                   =   mean(TMP,2);
vari                    =   var(TMP,[],2);
figure;
set(gcf,'name','mean face');
mimage = reshape(media,r,c);
vimage = reshape(vari, r, c);
subplot(1,2,1); imagesc(mimage);
subplot(1,2,2); imagesc(vimage);
colormap gray, axis image;


A(:,:)                  =   TMP-repmat(media,1,M);
[U,lambda]              =   eigen_training(A);

%% NB: 
% U è il numero di pixel X gli autovalori, lambda contiene gli  autovalori 
% ordinati in maniera decresente. 
% La varianza ci mostra in quali punti c'è maggiore discontinuità.


%Exercize 1
% write the eigen_training function, which takes the mean-subtracted matrix
% of faces, and apply the eigenfaces algorithm. Remember, the algorithm is
% the same as PCA, but instead of the scatter matrix (AA') we have that
% projection in M-dimensions A'A. U should contain the ordered
% eigenvectors, lambda the corresponding eigenvalues.

% Exercize 2 
% show the eigenvectors in a figure as they were images (in fact we are talking)
% about "eigenfaces". For convenience, show just the first 25 eigefaces
% figure;
% for i=1:25
%     imagesc(reshape(U(:,i),r,c)); colormap gray; axis image; colorbar;
%     pause(0.5);
% end

%Un'immagine è una somma pesata di tutti questi elementi; infatti abbiamo
%visto che è data da X = mean + sum(_i=1^400 e_i'*a_i)
%1: capelli biondi; 
%2: capelli scuri;
%3: pelle chiara;
%4: se l'illuminazione viene da sinistra;
%5: se l'illuminazione viene da destra;
% ....
% Da notare che le basse frequenze sono più definite nella prima che è
% quella più importante. Mentre nell'immagine 25 l'immagine frequenziale è
% più alta. Questo a prova del fatto che le prime immagini definiscono ad
% esempio la forma del volto o comunque connotati più generici, mentre
% procedendo acquisiamo dati più specifici come ad esempio la presenza e
% non di occhiali o l'attaccatura dei capelli.

% Visualizing the first 25 eigenfaces
figure;
for i=1:25
    subplot(5,5,i); imagesc(reshape(U(:,i),[r,c]));
    colormap gray; axis image; axis off; title(num2str(i)); colorbar
end

% Exercize 3 
% show the eigenvectors that are far in the ordering (for example, starting from the 300th). 
% For convenience, show just 25 eigefaces
figure;
for i=1:25
    subplot(5,5,i); imagesc(reshape(U(:,i+300),[r,c]));
    colormap gray; axis image; axis off; title(num2str(i+300)); colorbar
end
% Exercize 4 
% show the eigenvalues and the amount of cumulative variance they capture

figure;
subplot(211)
plot(lambda); title('Eigenvalues');
hold on;
scatter([1:M],lambda);
%Se volessimo modellare le immagini così da avere l'80% dell'informazione,
%dovremmo tenere all'incirca le prime 50 immagini.

subplot(212)
y = cumsum(lambda)/sum(lambda);
plot(y); title('Modelled Information')
hold on;
scatter([1:M],y);

%Exercize 5 Do 2D projection of faces! Show the projected faces on a two
%dimensional space.

% Per fare la proiezione di un'immagine su una particolare direzione, devo
% ricavare a_i = e_i^T*X
% Dobbiamo proiettare le facce su uno spazio bidimensionale.

a_12 = U(:,1:2)'*A(:,:);
figure; scatter(a_12(1,:),a_12(2,:));




% % % 2D
T           =   2;
omega       =   U(:,1:T)'*A; % proiezione;
figure; set(gcf,'name','Visualizzazione 2D dei volti');
scatter(omega(1,:),omega(2,:));
l           =   reshape(repmat([1:41],10,1),410,1);
for i=1:M
    text(omega(1,i)+25,omega(2,i)+25,num2str(l(i)))
end
% 
% for i=1:M
%     text(omega(1,i)+25,omega(2,i)+25,num2str(i))
% end

% exercize 6
% I'm considering that you have your personal data in the dataset, so 10
% images. You need to compute, for each of these 10 images the distances
% wrt all the other images in the low/dimensional space, for example using
% a number of eigenvectors that model 90% of the variance.

% Then, for each of the image, you can compute the subject which is the
% closest one. For each of the 10 images you will have a subject which is
% the closest one

% In order to select just the most similar one, you need to evaluate a
% majority criterion over the 10 most similar subjects you had.

% Exercize 8
%For each identity, compute which one contains the highest variance in the
%low dimensional space.

for i =1:M/10
    good = find(l==i);
    omega_good = omega(:,good);
    cova=cov(omega_good');
    spread(i) = sum(sqrt(diag(cova)));
end

[maxs, max_i] = max(spread);

%Quello con massimo spread è 10 perché in alcune foto la disparità di posa
%e illuminazione è molto più differente


% Exercize 9
% In the cloud of points in my few-dimensional space (400/410), I want to
% compute maximum distance and the minimum distance holding between the w
% projections of different people. Visualize the pair of images corresponding 
%to the minimum and maximum distance.

% T = 410;
% omega       =   U(:,1:T)'*A; % proiezione; 410xsn * 410 features
% 
% for i=1:T
%     for j =1:T
%         D(i,j)= norm(omega(:,i)' - omega(:,j)');
%     end
% end
% dista = mean(D);
% [maxx, indM] = max(dista);
% dist = dist + diag(ones(1,T) * Inf);
% [minn indm] = min(dista);

% T           =   410;
% omega       =   U(:,1:T)'*A;
% num_id = M/10;
% dista = zeros (M,M); %matrice di distanza tra le varie immagini
% 
% for i=1:num_id    %ciclo sul numero di identità
%     good = find(l==i);
%     adv_id = setdiff(1:M, good);   %fa la differenza insiemistica, perché 
%                             %può essere che le miei immagini si assomiglino
%                             % di più tra loro rispetto a quelle di un altro
%                             %tizio    
%     for ej = 1:length(good)
%         j = good(ej);
%            for ek = length(adv_id)  %popolo la matrice
%                k = adv_id(ek);
%                dista (j,k) = norm(omega(:,j)'-omega(:,k)');
%            end
%            for ek = 1:length(good)
%                k = good(ek);
%                dista(j,k)=-1;
%            end
%     end
% end
% figure; imagesc(dista); set(gcf,'name', 'Distanza tra proiezioni');colorbar;
T           =   410;
omega       =   U(:,1:T)'*A; % my projection;
for i=1:M
    for j=1:M
        dista(i,j)= norm(omega(:,i)'-omega(:,j)');

    end
end
% 
% for ei=1:M
%     i = fix(ei/10) + 1;
%     for j=1:M
%         if (fix(j/10) == i-1 || j/10== i)
%         dista(i,j)= NaN;
%         end
% 
%     end
% end
lim = 1;
while lim<=T
    for i = lim:lim+9
        for j=lim:lim+9
            dista(i,j)=NaN;
        end
    end
    lim = lim+10;
end
    
figure; imagesc(dista); set(gcf,'name','Distances among projections')
colorbar;
dista = dista + diag(ones(1,M)*inf);
[distmin,Imin]= min(dista(:));
[sim_i,sim_j]=ind2sub([M,M],Imin);
[distmax,Imax]= max(dista(dista~=inf));
[dsim_i,dsim_j]=ind2sub([M,M],Imax);
figure; set(gcf,'name','Similarita nello spazio 50D = spazio orig?');
subplot(2,2,1); imagesc(reshape(TMP(:,sim_i),r,c));axis image
title(strcat(num2str(sim_i),'-simile-')); colormap gray
subplot(2,2,2); imagesc(reshape(TMP(:,sim_j),r,c));axis image
title(strcat(num2str(sim_j),'-simile-')); colormap gray
subplot(2,2,3); imagesc(reshape(TMP(:,dsim_i),r,c));axis image
title(strcat(num2str(dsim_i),'-dissimile-')); colormap gray
subplot(2,2,4); imagesc(reshape(TMP(:,dsim_j),r,c));axis image
title(strcat(num2str(dsim_j),'-dissimile-')); colormap gray

% Exercize 9
% Reconstruct an image of your dataset that has -NOT- used to build the
% eigenspace. How is its reconstruction, in terms of similarity with the
% original image?

clear U
clear lambda
clear A
ind_img                 =   110; % this is the image I want to keep out;
% good                    =   setdiff([1:M], ind_img); % subtracting it from the original dataset
TMPnew                  =   TMP;
TMPnew(:,ind_img)       =   []; %obtaining a different dataset with my image which is absent here
media                   =   mean(TMPnew,2);
A(:,:)                  =   TMPnew-repmat(media,1,M-1);
[U,lambda]              =   eigen_training(A);

omega = [];
for t=1:410
    T              =   t;
    omegatmp       =   U(:,T)'*(TMP(:,ind_img)-media); % proiezione;
    omega          =   [omega; omegatmp];
    rec = media + U(:,1:T)*omega;
    subplot(1,3,1); imagesc(reshape(TMP(:,ind_img),r,c)); colormap gray
    title('originale'); axis image
     subplot(1,3,2); imagesc(reshape(rec,r,c)); colormap gray
    title(strcat('ricostruita con ',num2str(t),'-autovettori')); axis image
    subplot(1,3,3); imagesc(reshape(U(:,t),r,c)); colormap gray
    title(strcat('eigenface n:',num2str(t),'|a_i=',num2str(omega(t)))); axis image
    drawnow; pause (0.01); 
end


ind_img = 405; %Immagine I che voglio far venir fuori
tmpNew = TMP;
media = mean(tmpNew, 2);
A(:,:) = tmpNew-repmat(media, 1,M);%sottraggo al dataset la media per ottenere la matrice A
[U, lambda] = eigen_training(A);
omega = [];
for t=1:410
    T = t;
    omegatmp = U(:,T)'*(TMP(:,ind_img)-media); %faccio la proiezione della mia immagine
    omega = [omega; omegatmp]; %Omega è il vettore che voglio ricostruire
    %lo faccio usando la formula media + sommatoria
    rec = media + U(:,1:T)*omega;
    subplot(1,3,1); imagesc(reshape(tmpNew(:,ind_img),r,c)); colormap gray
    title('originale'); axis image

    subplot(1,3,2); imagesc(reshape(rec,r,c)); colormap gray
    title(strcat('ricostruita con ', num2str(t),' autovettori')); axis image

    subplot(1,3,3); imagesc(reshape(U(:,T),r,c)); colormap gray
    title(strcat('autovettore ',num2str(omega(t)))); axis image

    pause(0.05)
end
 % Exercize 10
% Reconstruct an image of subject i by using an eigenspace whose eigenvectors have been
% created -WITHOUT ANY OF THE IMAGES OF SUBJECT i- and check the quality of the reconstruction 

clear U
clear lambda
clear A
ind_img                 =   409; % this is the image I want to keep out;
% good                    =   setdiff([1:M], ind_img); % subtracting it from the original dataset
TMPnew                  =   TMP;
TMPnew(:,ind_img)       =   []; %obtaining a different dataset with my image which is absent here
media                   =   mean(TMPnew,2);
A(:,:)                  =   TMPnew-repmat(media,1,M-1);
[U,lambda]              =   eigen_training(A);

omega = [];
rico= 409;
for t=1:409
    T              =   t;
    omegatmp       =   U(:,T)'*(TMP(:,rico)-media); % proiezione;
    omega          =   [omega; omegatmp];
    rec = media + U(:,1:T)*omega;
    subplot(1,3,1); imagesc(reshape(TMP(:,rico),r,c)); colormap gray
    title('originale'); axis image
     subplot(1,3,2); imagesc(reshape(rec,r,c)); colormap gray
    title(strcat('ricostruita con ',num2str(t),'-autovettori')); axis image
    subplot(1,3,3); imagesc(reshape(U(:,t),r,c)); colormap gray
    title(strcat('eigenface n:',num2str(t),'|a_i=',num2str(omega(t)))); axis image
    drawnow; pause (0.01); 
end

 

clear U
clear lambda
clear A
ind_img                 =   401:410; % this is the image I want to keep out;
% good                    =   setdiff([1:M], ind_img); % subtracting it from the original dataset
TMPnew                  =   TMP;
TMPnew(:,ind_img)       =   []; %obtaining a different dataset with my image which is absent here
media                   =   mean(TMPnew,2);
A(:,:)                  =   TMPnew-repmat(media,1,M-10);
[U,lambda]              =   eigen_training(A);

omega = [];
rico= 401;
for t=1:400
    T              =   t;
    omegatmp       =   U(:,T)'*(TMP(:,rico)-media); % proiezione;
    omega          =   [omega; omegatmp];
    rec = media + U(:,1:T)*omega;
    subplot(1,3,1); imagesc(reshape(TMP(:,rico),r,c)); colormap gray
    title('originale'); axis image
     subplot(1,3,2); imagesc(reshape(rec,r,c)); colormap gray
    title(strcat('ricostruita con ',num2str(t),'-autovettori')); axis image
    subplot(1,3,3); imagesc(reshape(U(:,t),r,c)); colormap gray
    title(strcat('eigenface n:',num2str(t),'|a_i=',num2str(omega(t)))); axis image
    drawnow; pause (0.01); 
end
 


 % THE VERY LAST EXERCIZE!!!
 % MAD!!!
 % Reconstruct an image of WHATEVER YOU WANT (please, not a face!) with
 % your eigenspace. Evaluate the reconstruction. This is an example of
 % HALLUCINATION.
 
 
 omega = [];
rico= 110;
mad  = double(imresize(rgb2gray(imread('shoes.jpg')),[112,92]));
for t=1:390
    T              =   t;
    omegatmp       =   U(:,T)'*(mad(:)-media); % proiezione;
    omega          =   [omega; omegatmp];
    rec = media + U(:,1:T)*omega;
    subplot(1,3,1); imagesc(mad); colormap gray
    title('originale'); axis image
     subplot(1,3,2); imagesc(reshape(rec,r,c)); colormap gray
    title(strcat('ricostruita con ',num2str(t),'-autovettori')); axis image
    subplot(1,3,3); imagesc(reshape(U(:,t),r,c)); colormap gray
    title(strcat('eigenface n:',num2str(t),'|a_i=',num2str(omega(t)))); axis image
    drawnow; pause (0.01); 
end
 

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
num_identities = 41;
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
    T           =   M-num_identities;
    omega_g       =   U(:,1:T)'*newA; % projection of the gallery;
    omega_p     =   U(:,1:T)'*(probe-repmat(media,1,num_identities)); % projection of the probe
    num_neigh = 3;
    for j = 1:num_identities
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
