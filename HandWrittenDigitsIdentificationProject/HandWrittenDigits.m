% UBITname    :   dhirenbh
% UBIT number :   50170084

%% 
%%Fetching 60000 testing images
%{
fid = fopen('train-images.idx3-ubyte', 'r', 'b');
header = fread(fid, 1, 'int32');
count = fread(fid, 1, 'int32');
disp(count);
h = fread(fid, 1, 'int32');
w = fread(fid, 1, 'int32');
disp(h);
disp(w);
imgs = zeros([h w count]);
for i = 1 : count
     for y=1:h
         imgs(y,:,i) = fread(fid, w, 'uint8');
     end
end
 imgs = double(imgs);
 for i=1:size(imgs, 3)
     imgs(:,:,i) = imgs(:,:,i)./255.0;
 end
fclose(fid);
%% 
%%Fetching 60000 testing labels
fid = fopen('train-labels.idx1-ubyte', 'r', 'b');
header = fread(fid, 1, 'int32');
count = fread(fid, 1, 'int32');
labels = fread(fid, count, 'uint8');
fclose(fid);
%}
%imageTrain = loadMNISTImages('train-images.idx3-ubyte');  
 %labelTrain = loadMNISTLabels('train-labels.idx1-ubyte');  
 %imageTest = loadMNISTImages('t10k-images.idx3-ubyte');  
 %labelTest = loadMNISTLabels('t10k-labels.idx1-ubyte');   
%%
%Logistic regression method
%Wlr = zeros(785,10);
%for i = 1 : 10
 %   Wlr(:,i) = randi(10,785,1);
%end
m = 1;
Wlr = rand(785,10);
Wlr = Wlr_temp;
%BETA  = 0.01;

precision = 0.001;
eta = 0.01;
G_E = zeros(785,10);
E_prev = 100;
E = 0;
 E_lg = zeros(60000,1);
for o = 1 : 1
    disp('o');
     disp(o);
  % eta = 0.01;
for i = 1 : 60000
    index = 1;
    processed_img = zeros(785,1);
    processed_img(index,1) = 1;
    %index = index + 1;
    %for j = 1 : 28
     %   for l = 1 : 28
      %      processed_img(index,1) = imgs(j,l,i); 
       %     index = index + 1;
       % end
    %end
    for j = 1 : 784
    processed_img(j+1,1) = imageTrain(j,i); 
    end
    t = zeros(10,1);
    t(labelTrain(i,1)+1,1) = 1;
    
    a = zeros(10,1);
    %denom = 0;
    for k = 1 : 10
        a(k,1) = (transpose(Wlr(:,k))*processed_img) ;
        %denom = denom + exp(a(k,1));
    end
   %max_a = max(a);
    %a = a / max_a;
    %a = a *BETA;
    denom = sum(exp(a));
    
    y = zeros(10,1);
    for k = 1 : 10
        y(k,1) = (exp(a(k,1)))/(denom);
    end
   % y = sigmf(a, [BETA, 0]);
     G_E = zeros(785,10);
    for k = 1 : 10
        G_E(:,k) = G_E(:,k) + (y(k,1)-t(k,1))*processed_img; 
    end
    
    for k = 1 : 10
        E = E - (t(k,1)*log(y(k,1)));
    end
    E_lg(i,1) = E;
    
    if E < precision
    %    break;
    end
    if mod(i,m) == 0
        if i~= m
        if E < E_prev
            eta = eta + (0.05*eta);
        else
            eta = eta*0.5;
            if eta <0.001
                eta = 0.01;
            end
            Wlr = temp_Wlr;
            %continue;
        end
        end
        %eta = 0.001;
        
        temp_Wlr = Wlr;      
        E_prev = E;
        E = 0;
        %G_E = zeros(785,10);
    end
       % eta = 0.001;
      for k = 1 : 10    
        Wlr(:,k) = Wlr(:,k) - eta* G_E(:,k);        
        end 
        
    
    
end

 plotx = [];
ploty = [];
for i = 59001 : 60000
    ploty(i,1) = E_lg(i,1);
    plotx(i,1) = i-59000;
end
plot(plotx,ploty);
title('Error Vs Iteration Graph');
xlabel('Iterations while training');
ylabel('Error');

misclass = 0;
for i = 1 : 10000
    index = 1;
    processed_img = zeros(785,1);
    processed_img(index,1) = 1;
    index = index + 1;
    %for j = 1 : 28
     %   for l = 1 : 28
      %      processed_img(index,1) = test_imgs(j,l,i); 
       %     index = index + 1;
       % end
    %end
    for j = 1 : 784
    processed_img(j+1,1) = imageTest(j,i); 
    end
    t = zeros(10,1);
    t(labelTest(i,1)+1,1) = 1;
    
    a = zeros(10,1);
    %denom = 0;
    for k = 1 : 10
        a(k,1) = (transpose(Wlr(:,k))*processed_img) ;
        %denom = denom + exp(a(k,1));
    end
    %max_a = max(a);
    %a = a / max_a;
    %a= a*BETA;
    denom = sum(exp(a));
    y = zeros(10,1);
    for k = 1 : 10
        y(k,1) = (exp(a(k,1)))/(denom);
    end
     
    max_y = find(y == max(y));
    if max_y ~= (labelTest(i,1)+1)
        misclass = misclass + 1;
    end
    %%if(y(test_labels(i,1)+1,1) < 0.5)
      %%  misclass = misclass + 1;
    %%end
end
disp('misclass : ');
disp(misclass);
end


Wlr_temp = Wlr;
blr = Wlr(1,:);
Wlr = zeros(784,10);
for i = 2:785
    Wlr(i-1,:)=Wlr_temp(i,:);
end

%%Neural network method
trainSize = size(imageTrain);  
 testSize = size(imageTest); 
 N = trainSize(2);  
 TOTAL_HU = 700;  
 TOTAL_IN = trainSize(1);  
 TOTAL_OUT = 10;  
 MAX_ITERATION = 1;  
 m1 = 1;
 E_cur = 0;
 BETA = 0.01; % Scaling factor in sigmoid function  
 eta = 0.01; % Learning rate  
 %% Initialize the weights from Uniform(0, 1)  
 %w1 = rand(TOTAL_IN + 1, TOTAL_HU); % Weight for layer-1, including bias unit  
 %w2 = rand(TOTAL_HU + 1, TOTAL_OUT); % Weight for layer-2. including bias unit  
 w1_temp = zeros(TOTAL_IN + 1, TOTAL_HU); % Weight for layer-1, including bias unit  
 w2_temp = zeros(TOTAL_HU + 1, TOTAL_OUT);
 %% Temporary matrix to store weight updates and activations  
 x1 = zeros(TOTAL_IN + 1, 1);  
 x2 = zeros(TOTAL_HU + 1, 1);  
 x3 = zeros(TOTAL_OUT, 1);   
 e2 = zeros(TOTAL_HU + 1, 1);  
 e3 = zeros(TOTAL_OUT, 1);    
 A = eye(TOTAL_OUT);  
 success = zeros(MAX_ITERATION, 1); 
 E = zeros(60000,1);
 %% Training  
 for t = 1:MAX_ITERATION  
   fprintf('Iteration %d\n', t);  
   perm = randperm(N);  
   for p = 1:N  
     if (mod(p, 1000) == 0)  
       fprintf('\t%d\n', p);  
     end  
     % Propogate forward      
     x1 = imageTrain(:, p);  
     x1 = [1; x1];
     x2 = logsig(w1'*x1); 
     x2 = [1; x2]; 
     x3 = logsig(w2' * x2);
     t1 = zeros(10,1);
      t1(labelTrain(p,1)+1,1) = 1; 
      e3 = x3 - t1 ;
     e2 =  (x2) .* (1 - (x2)).* (w2 * e3);
     
     e2 = e2(2:TOTAL_HU+1);  
     % Update weight  
     
     
     for k = 1 : 10
        E(p,1) = E(p,1) - (t1(k,1)*log(x3(k,1)));
     end
     E_cur = E_cur + E(p,1); 
     
      if mod(p,m1) == 0
        if p~= m1
        if E_cur < E_prev
            eta = eta + (0.05*eta);
        else
            eta = eta*0.5;
            if eta <0.001
                eta = 0.01;
            end
            w1=w1_temp;
            w2 = w2_temp;        
            %continue;
        end
        end
        E_prev = E_cur;
        E_cur = 0;
        w1_temp = w1;
        w2_temp = w2;
      end
     
     w2 = w2 - eta * (x2 * e3');
     w1 = w1 - eta * (x1 * e2'); 
      
   end  
   
   plotx = [];
ploty = [];
for i = 1 : 1000
    ploty(i,1) = E(i,1);
    plotx(i,1) = i;
end
plot(plotx,ploty);

   %% Check training error  
   success(t) = 0;  
   for i = 1:N  
     x1 = imageTrain(:, i); 
      x1 = [1; x1];
     x2 = logsig(w1' * x1);
     x2 = [1; x2];
      x3 = logsig(w2' * x2);
    
     [dummy, m] = max(x3);  
     if (m ~= labelTrain(i) + 1)  
       success(t) = success(t) + 1;  
     end  
   end  
   success(t) = (success(t) /60000) * 100;
 end  

 testSuccess = 0;  
 for i = 1:testSize(2)  
   x1 = imageTest(:, i); 
   x1 = [1; x1];
   x2 = logsig(w1' * x1);
   x2 = [1; x2];
   x3 = logsig(w2' * x2); 
   [dummy, m] = max(x3);  
   if (m ~= labelTest(i) + 1)  
     testSuccess = testSuccess + 1;  
   end  
 end  
 testSuccess = testSuccess /100;
 Wnn1 = zeros(TOTAL_IN,TOTAL_HU);
 for i = 2 : 785
     Wnn1(i-1,:)=w1(i,:);
 end
 bnn1 = zeros(1,TOTAL_HU);
 bnn1(1,:)=w1(1,:);
 Wnn2 = zeros(TOTAL_HU,TOTAL_OUT);
 for i = 2 : TOTAL_HU+1
     Wnn2(i-1,:) = w2(i,:);
 end
 bnn2 = zeros(1,TOTAL_OUT);
 bnn2(1,:)=w2(1,:);
 
 
 %Convolutional neural network method
 cnn.layers = {
 struct('type', 'i') %input layer
 struct('type', 'c', 'outputmaps', 10, 'kernelsize', 7) %convolution layer
 struct('type', 's', 'scale', 2) %sub sampling layer
};

cnn = cnnsetup(cnn,imgs ,labels_a);

opts.alpha = 0.1; % learning rate
opts.batchsize = 10;
opts.numepochs = 10;

cnn = cnntrain(cnn, imgs,labels_a, opts);

[er, bad] = cnntest(cnn, test_imgs, test_labels_a);


