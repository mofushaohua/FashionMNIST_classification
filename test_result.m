% This file Outputs the fashion MNIST test file to a csv file

load mnistclassify_weights;
load fashionMNISTtest;
testdata = D/255;

err=0;
[testnumcases testnumdims]=size(testdata);
N=testnumcases;
testdata = [testdata ones(N,1)];
w1probs = 1./(1 + exp(-testdata*w1)); w1probs = [w1probs  ones(N,1)];
w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
targetout = exp(w3probs*w_class);
targetout = targetout./repmat(sum(targetout,2),1,10);

[I J]=max(targetout,[],2);

towrite = [[0:9999]' J-1];
csvwrite('result.csv',towrite);