clc
clear all
close all


dire    =    '.\images'; % ORL face database (plus my images!!!)
list    =   dir(strcat(dire,'\*.jpg'));
for i=1:10
   tmp = imread(strcat(dire,'\',list(i).name));
   tmp = imcrop(tmp);
   tmp = rgb2gray(tmp);
   tmp = imresize(tmp, [112,92]);
   imwrite(tmp, strcat('.\images\', num2str(400+i), '.bmp'));
   
end