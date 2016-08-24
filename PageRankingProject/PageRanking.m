%%
% Real data implementation
%Loading dataa from input file and making proper input and output matrix
%{
%This block of commented code is used to load 69623 x 46 data from text
%file for first time only... i have saved that in workspace...
h = waitbar(0,'Please wait...');
steps = 69623;
for step = 1:steps
    
fileimport = importdata('Querylevelnorm.txt',' ');
t_real = fileimport.textdata(:,1);
for i = 1 : length(fileimport.textdata(:,1))
    for j = 3 : 48
        c=strsplit(num2str(cell2mat(fileimport.textdata(i,j))),':');
       samples_real(i,j-2) = c(1,2); 
    end
     waitbar((step/steps),h,'Completion till now')
end
end
close(h)
%}

%% 
%Real data started
%Initialization
E = [];
min = 10000;
no_of_iter = 30;
samples = [];
input = [];
input_val = [];
data_to_train = 0.8 * length(x_real(:,1));
samples = x_real;
for i = 1 : data_to_train
   input(i,:)= samples(i,:);
end
for i = 1 : 6962
   input_val(i,:)= samples(i+55698,:);
end
whole_sigma = cov(input).*eye(46);
%whole_sigma_val = cov(input_val).*eye(46);
whole_sigma = whole_sigma + (eye(46)/100);
for M1_temp = 25 : no_of_iter
mu1_temp = [];
Sigma1_temp = [];
basis_temp = [];
w1_temp=[];

%% Finding mu and sigma for each cluster
cluster_ids = [];
cluster_ids = kmeans(input,M1_temp-1);
row_ind = [];
for i = 1 : M1_temp-1
    row_ind(1,i) = 0;
end
temp = [];
for i = 1 : data_to_train
    row_ind(1,cluster_ids(i,1)) = row_ind(1,cluster_ids(i,1)) + 1;
  for j = 1 : length(input(1,:))
     temp(row_ind(1,cluster_ids(i,1)),j,cluster_ids(i,1)) = input(i,j);
  end
end
for i = 1 : M1_temp-1
    temp1 = [];
    for j = 1 : length(temp(:,1,i))
        sum = 0;
        for k = 1 : length(temp(j,:,i))
            sum = sum + temp(j,k,i);
        end
        if sum ~= 0
            temp1 = [temp1 ; temp(j,:,i)];
        end
    end
   
    for k = 1 : length(temp1(1,:))
              mu1_temp(k,i) = mean(temp1(:,k)); 
    end
   
    
            %  mu2_temp(:,i) = mean(temp1);
             Sigma1_temp(:,:,i) = whole_sigma;
          %  Sigma1_temp(:,:,i) = cov(temp1);
            % Sigma2_temp(:,:,i) = Sigma2_temp(:,:,i).*eye(10);  
end    

%% finding basis function array
for i = 1 : data_to_train
    for j = 1 : length(mu1_temp(1,:))
        basis_temp(i,j) = exp((-1/2)*(input(i,:)-transpose(mu1_temp(:,j)))*inv(whole_sigma)*transpose((input(i,:)-transpose(mu1_temp(:,j)))));
    end
end


%lambda2 = error_sum / EwW;
basis_temp = [ones(length(basis_temp(:,1)),1) basis_temp];
basis_temp_val = [];
for i = 1 : 6962
    for j = 1 : length(mu1_temp(1,:))
        basis_temp_val(i,j) = exp((-1/2)*(input_val(i,:)-transpose(mu1_temp(:,j)))*inv(whole_sigma)*transpose((input_val(i,:)-transpose(mu1_temp(:,j)))));
    end
end
basis_temp_val = [ones(length(basis_temp_val(:,1)),1) basis_temp_val];

k=0;
lambda1_temp = 0;
%for lambda2_temp = 0.1 : 2 
while lambda1_temp <= 0.5
    lambda1_temp = lambda1_temp + 0.1;
    k=k+1;
%lambda2=0;
%% Recalculating w2 and y according to lambda2 value
%for ii = 1 : 200
output = [];
for i = 1 : data_to_train
   output(i,1)= t_real(i,1);
end
I = eye(M1_temp);
w1_temp = inv((lambda1_temp*I)+(transpose(basis_temp)*basis_temp))*transpose(basis_temp)*output;
%% Recalculating linear regression output according to new w2 value
y = [];
for i = 1 : data_to_train
    sum = 0;
    for j = 1 : length(mu1_temp(1,:))+1
        sum = sum + w1_temp(j,1)*basis_temp(i,j) ;
    end
    y(i,:) = sum;
end


%%
%Finding validation Error
%% finding basis function array

y_val = [];
for i = 1 : 6962
    sum = 0;
    for j = 1 : length(mu1_temp(1,:))+1
        sum = sum + w1_temp(j,1)*basis_temp_val(i,j) ;
    end
    y_val(i,:) = sum;
end

EwW = 0;
for i = 1 : M1_temp
    EwW = EwW + ((w1_temp(i,1)^2)/2);
end
EwW = lambda1_temp*EwW;
error_sum = 0;
for i = 1 : 6962
    error_sum = error_sum + ((t_real(i+55698,1)-y_val(i,1))^2);
end
error_sum = error_sum/2;
valid_sqr_err = error_sum;
validPer1_temp = sqrt((2*(error_sum))/6962);
%% Recalculating Root mean square error

E(M1_temp,k) = error_sum + (lambda1_temp * EwW);

error_sum = 0;
for i = 1 : data_to_train
    error_sum = error_sum + ((t_real(i,1)-y(i,1))^2);
end
error_sum = error_sum/2;
trainPer1_temp = sqrt((2*(error_sum))/55698);

if (validPer1_temp < min)
    lambda1 = [];
    mu1 = [];
    mu1 = zeros(length(mu1_temp(:,1)),1);
    Sigma1 = [];
    Sigma1(:,:,1) = eye(length(mu1_temp(:,1)),length(mu1_temp(:,1)));
    w1 = [];
    basis = [];
%min = E(M2_temp,k);
min = validPer1_temp;
trainPer1 = trainPer1_temp;
validPer1 = validPer1_temp;
    M1=M1_temp;
    lambda1 = lambda1_temp;
    mu1 =[mu1  mu1_temp];
    for i = 1 : length(Sigma1_temp(1,1,:))
        Sigma1(:,:,i+1) = Sigma1_temp(:,:,i);
    end
 w1 = w1_temp;
    basis = basis_temp;
    basis_val = [];
    basis_val = basis_temp_val;
    y_act_real = [];
    y_act_real = y;
    y_act_val_real = [];
    y_act_val_real = y_val;
end
end
end


%Gradient-Descent Implementation

M1_temp = M1;

%%
%Initialization

w01 = [];
w01_temp = [];

for i = 1 : M1_temp
    w01(i,1) = 20;
end
w01_temp = w01;

E = 0;
dw1 = [];
eta1 = [];
ijk = 0;
ini_eta1 = 1;
while ijk <2
ijk  = ijk + 1;
y = [];
y_val = [];

for i = 1 : data_to_train
precision = 0.001;

    sum = 0;
    for j = 1 : length(mu1(1,:))
       sum = sum + w01_temp(j,1)*basis(i,j) ;
    end
    error_sum_prev = (((t_real(i,1)-sum)^2)/2);
    E= E+1;
   dw1(:,E) = zeros(M1,1);
    eta1(1,E) = zeros(1,1);
%while no_iter<max_ope
   % ini_eta2 = 1/(no_iter);
    gradient_step = [];
    change = [];
    
   change = (( output(i,1)- (transpose(w01_temp)*transpose(basis(i,:))) )*transpose(basis(i,:))) ;
   gradient_step = ini_eta1*(change -  (lambda1*w01_temp)); 
   dw1(:,E) = dw1(:,E) + gradient_step;
   eta1(1,E) = eta1(1,E) + ini_eta1;  
   %if(mean(abs(change)) < precision)
    %   break;
   %end
   w01_temp_prev = [];
   w01_temp_prev = w01_temp;
   w01_temp = w01_temp + gradient_step;
   no_iter = no_iter + 1;
   
      sum = 0;
    for j = 1 : length(mu1(1,:))
        sum = sum + w01_temp(j,1)*basis(i,j) ;
    end
    error_sum = (((t_real(i,1)-sum)^2)/2);
    if(error_sum < precision)
    %if(mean(abs(change))<precision)
    %w02_temp = w02_temp - gradient_step;
        %break;
    end
    if(error_sum < error_sum_prev)
    %if(mean(abs(change))<error_sum_prev) 
    ini_eta1 = ini_eta1 + (0.05*ini_eta1);
    else
        ini_eta1 = ini_eta1/2;
        %w02_temp = w02_temp_prev;
        continue;
    end
   error_sum_prev = error_sum;
   %error_sum_prev = mean(abs(change));
%end

y(i,:) = sum;
end

for i = 1 : 6962
    sum = 0;
    for j = 1 : length(mu1(1,:))
        sum = sum + w01_temp(j,1)*basis_val(i,j) ;
    end
    y_val(i,:) = sum;
end

error_sum = 0;
for i = 1 : 6962
    error_sum = error_sum + ((t_real(i+55698,1)-y_val(i,1))^2);
end
error_sum = error_sum/2;

end


%Real data over

%% 
%Synthetic data started
%Initialization
E = [];
min = 10000;
no_of_iter = 61;
samples = [];
input = [];
input_val = [];
input_test = [];
data_to_train = 0.8 * length(x(1,:));
samples = transpose(x);
for i = 1 : data_to_train
   input(i,:)= samples(i,:);
end
for i = 1 : 200
   input_val(i,:)= samples(i+1600,:);
end
for i = 1 : 200
   input_test(i,:)= samples(i+1800,:);
end
whole_sigma = cov(input).*eye(10);
whole_sigma_val = cov(input_val).*eye(10);
%whole_sigma = whole_sigma/10;
for M2_temp = 2 : no_of_iter
mu2_temp = [];
Sigma2_temp = [];
basis_temp = [];
w2_temp=[];

%% Finding mu and sigma for each cluster
cluster_ids = [];
cluster_ids = kmeans(input,M2_temp-1);
row_ind = [];
for i = 1 : M2_temp-1
    row_ind(1,i) = 0;
end
temp = [];
for i = 1 : data_to_train
    row_ind(1,cluster_ids(i,1)) = row_ind(1,cluster_ids(i,1)) + 1;
  for j = 1 : length(input(1,:))
      temp(row_ind(1,cluster_ids(i,1)),j,cluster_ids(i,1)) = input(i,j);
  end
end
for i = 1 : M2_temp-1
    temp1 = [];
    for j = 1 : length(temp(:,1,i))
        sum = 0;
        for k = 1 : length(temp(j,:,i))
            sum = sum + temp(j,k,i);
        end
        if sum ~= 0
            temp1 = [temp1 ; temp(j,:,i)];
        end
    end
    
    for k = 1 : length(temp1(1,:))
              mu2_temp(k,i) = mean(temp1(:,k)); 
    end
    
            %  mu2_temp(:,i) = mean(temp1);
             Sigma2_temp(:,:,i) = whole_sigma; 
            % Sigma2_temp(:,:,i) = Sigma2_temp(:,:,i).*eye(10);  
end    

%% finding basis function array
for i = 1 : data_to_train
    for j = 1 : length(mu2_temp(1,:))
        basis_temp(i,j) = exp((-1/2)*(input(i,:)-transpose(mu2_temp(:,j)))*inv(whole_sigma)*transpose((input(i,:)-transpose(mu2_temp(:,j)))));
    end
end


%lambda2 = error_sum / EwW;
basis_temp = [ones(length(basis_temp(:,1)),1) basis_temp];
basis_temp_val = [];
for i = 1 : 200
    for j = 1 : length(mu2_temp(1,:))
        basis_temp_val(i,j) = exp((-1/2)*(input_val(i,:)-transpose(mu2_temp(:,j)))*inv(whole_sigma)*transpose((input_val(i,:)-transpose(mu2_temp(:,j)))));
    end
end
basis_temp_val = [ones(length(basis_temp_val(:,1)),1) basis_temp_val];

basis_temp_test = [];
for i = 1 : 200
    for j = 1 : length(mu2_temp(1,:))
        basis_temp_test(i,j) = exp((-1/2)*(input_test(i,:)-transpose(mu2_temp(:,j)))*inv(whole_sigma)*transpose((input_test(i,:)-transpose(mu2_temp(:,j)))));
    end
end
basis_temp_test = [ones(length(basis_temp_test(:,1)),1) basis_temp_test];

k=0;
lambda2_temp = 0;
%for lambda2_temp = 0.1 : 2 
while lambda2_temp <= 0.1
    lambda2_temp = lambda2_temp + 0.01;
    k=k+1;
%lambda2=0;
%% Recalculating w2 and y according to lambda2 value
%for ii = 1 : 200
output = [];
for i = 1 : data_to_train
   output(i,1)= t(i,1);
end
I = eye(M2_temp);
w2_temp = inv((lambda2_temp*I)+(transpose(basis_temp)*basis_temp))*transpose(basis_temp)*output;
%% Recalculating linear regression output according to new w2 value
y = [];
for i = 1 : data_to_train
    sum = 0;
    for j = 1 : length(mu2_temp(1,:))+1
        sum = sum + w2_temp(j,1)*basis_temp(i,j) ;
    end
    y(i,:) = sum;
end


%%
%Finding validation Error
%% finding basis function array

y_val = [];
for i = 1 : 200
    sum = 0;
    for j = 1 : length(mu2_temp(1,:))+1
        sum = sum + w2_temp(j,1)*basis_temp_val(i,j) ;
    end
    y_val(i,:) = sum;
end
y_test = [];
for i = 1 : 200
    sum_test = 0;
    for j = 1 : length(mu2_temp(1,:))+1
        sum_test = sum_test + w2_temp(j,1)*basis_temp_test(i,j) ;
    end
    y_test(i,:) = sum_test;
end

EwW = 0;
for i = 1 : M2_temp
    EwW = EwW + ((w2_temp(i,1)^2)/2);
end
EwW = lambda2_temp*EwW;
error_sum = 0;
for i = 1 : 200
    error_sum = error_sum + ((t(i+1600,1)-y_val(i,1))^2);
end
error_sum = error_sum/2;
valid_sqr_err = error_sum;
validPer2_temp = sqrt((2*(error_sum))/200);

error_sum_test = 0;
for i = 1 : 200
    error_sum_test = error_sum_test + ((t(i+1800,1)-y_test(i,1))^2);
end
error_sum_test = error_sum_test/2;
testPer2_temp = sqrt((2*(error_sum_test))/200);
%% Recalculating Root mean square error

E(M2_temp,k) = error_sum + (lambda2_temp * EwW);

error_sum = 0;
for i = 1 : data_to_train
    error_sum = error_sum + ((t(i,1)-y(i,1))^2);
end
error_sum = error_sum/2;
trainPer2_temp = sqrt((2*(error_sum))/1600);

if (validPer2_temp < min)
    lambda2 = [];
    mu2 = [];
    mu2 = zeros(length(mu2_temp(:,1)),1);
    Sigma2 = [];
    Sigma2(:,:,1) = eye(length(mu2_temp(:,1)),length(mu2_temp(:,1)));
    w2 = [];
    basis = [];
%min = E(M2_temp,k);
min = validPer2_temp;
trainPer2 = trainPer2_temp;
validPer2 = validPer2_temp;
testPer2 = testPer2_temp;
    M2=M2_temp;
    lambda2 = lambda2_temp;
    mu2 =[mu2  mu2_temp];
    for i = 1 : length(Sigma2_temp(1,1,:))
        Sigma2(:,:,i+1) = Sigma2_temp(:,:,i);
    end
    w2 = w2_temp;
    basis = basis_temp;
    basis_val = [];
    basis_val = basis_temp_val;
    y_act = [];
    y_act = y;
    y_act_val = [];
    y_act_val = y_val;
end
end
end


%Gradient-Descent Implementation

M2_temp = M2;

%%
%Initialization

w02 = [];
w02_temp = [];

for i = 1 : M2_temp
    w02(i,1) = 5;
end
w02_temp = w02;

E = 0;
dw2 = [];
eta2 = [];
ijk = 0;
ini_eta2 = 1;
while ijk <20
ijk  = ijk + 1;
y = [];
y_val = [];

for i = 1 : data_to_train
precision = 0.001;

    sum = 0;
    for j = 1 : length(mu2(1,:))
       sum = sum + w02_temp(j,1)*basis(i,j) ;
    end
    error_sum_prev = (((t(i,1)-sum)^2)/2);
    E= E+1;
   dw2(:,E) = zeros(M2,1);
    eta2(1,E) = zeros(1,1);
%while no_iter<max_ope
   % ini_eta2 = 1/(no_iter);
    gradient_step = [];
    change = [];
    
   change = (( output(i,1)- (transpose(w02_temp)*transpose(basis(i,:))) )*transpose(basis(i,:))) ;
   gradient_step = ini_eta2*(change -  (lambda2*w02_temp)); 
   dw2(:,E) = dw2(:,E) + gradient_step;
   eta2(1,E) = eta2(1,E) + ini_eta2;  
   %if(mean(abs(change)) < precision)
    %   break;
   %end
   w02_temp_prev = [];
   w02_temp_prev = w02_temp;
   w02_temp = w02_temp + gradient_step;
   no_iter = no_iter + 1;
   
      sum = 0;
    for j = 1 : length(mu2(1,:))
        sum = sum + w02_temp(j,1)*basis(i,j) ;
    end
    error_sum = (((t(i,1)-sum)^2)/2);
    if(error_sum < precision)
    %if(mean(abs(change))<precision)
    %w02_temp = w02_temp - gradient_step;
        %break;
    end
    if(error_sum < error_sum_prev)
    %if(mean(abs(change))<error_sum_prev) 
    ini_eta2 = ini_eta2 + (0.05*ini_eta2);
    else
        ini_eta2 = ini_eta2/2;
        %w02_temp = w02_temp_prev;
        continue;
    end
   error_sum_prev = error_sum;
   %error_sum_prev = mean(abs(change));
%end

y(i,:) = sum;
end

for i = 1 : 200
    sum = 0;
    for j = 1 : length(mu2(1,:))
        sum = sum + w02_temp(j,1)*basis_val(i,j) ;
    end
    y_val(i,:) = sum;
end

error_sum = 0;
for i = 1 : 200
    error_sum = error_sum + ((t(i+1600,1)-y_val(i,1))^2);
end
error_sum = error_sum/2;

end

%Synthetic data over

