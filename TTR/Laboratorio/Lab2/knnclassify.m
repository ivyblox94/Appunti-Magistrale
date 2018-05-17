function class = knnclassify(test,train,label,K);

[num_c,num_el] = size(train);

repe = repmat(test,num_c,1);
dist = sqrt(sum((repe-train).^2,2));

[thrash,ind]= sort(dist,'ascend');

label_ind = label(ind);

label_ind = label_ind(1:K);

num_classi = max(label);
for i=1:num_classi;
    win(i)= sum(double(label_ind == i));
end

[thrash, class]= max(win);


