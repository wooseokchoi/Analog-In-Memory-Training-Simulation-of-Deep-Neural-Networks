clear;
%% load datasets
testset_inputs=loadMNISTImages('t10k-images-idx3-ubyte');
testset_labels=loadMNISTLabels('t10k-labels-idx1-ubyte');
trainset_inputs=loadMNISTImages('train-images-idx3-ubyte');
trainset_labels=loadMNISTLabels('train-labels-idx1-ubyte');
image_resolution = 784; % input image size

%% network parameters
NumNeurons = [image_resolution 256 128 10];   % Number of neurons for each neuron layers
neuron_type = 'sigmoid';  % 'sigmoid', 'relu' is available
error_function = 'mean_squared_error'; % 'mean_squared_error', 'cross_entropy_error' is available
LR = 0.04; % learning rate 
    LR_decay_period = 1; % per epoch
    LR_decay_rate = 0.5; % 1: no LR decay
epoch_number = 30; accuracy = zeros(epoch_number,1); E = zeros(epoch_number,1);
batch_size = 1; % 1: stochastic gradient decent
trainset_length = 5000; % maximum length: 60000
    trainset_inputs = trainset_inputs(:, 1:trainset_length);
    trainset_labels = trainset_labels(:, 1:trainset_length);

%% device parameters
update_mode = 2; % Stochastic
write_noise = 0.2;

device.wmax = 1; 
device.wmin = -1; 
device.dw_min = 0.002; 
device.dw_min_std = 0; % c2c variation
device.w_max_dtod = 0; % d2d variation
device.w_min_dtod = 0; % d2d variation
device.up_down = 0;
device.up_down_dtod = 0; % d2d variation
device.dw_min_dtod = 0; % d2d variation

device.Nstates = round((device.wmax-device.wmin) / device.dw_min);
len_bit = round( LR / (device.dw_min) ); % the length of stochastic bitstream for fully parallel crossbar updates

%% weight initialization
array = W_init(NumNeurons,device);
w12 = device.up_down + (randn(array.w12.size)*write_noise); % around the symmetry point
w23 = device.up_down + (randn(array.w23.size)*write_noise);
w34 = device.up_down + (randn(array.w34.size)*write_noise);

w12ref = w12 + (randn(array.w12.size)*write_noise);
w23ref = w23 + (randn(array.w23.size)*write_noise);
w34ref = w34 + (randn(array.w34.size)*write_noise);

%% pre-setting
if update_mode == 1
    fprintf('Update mode -- Deterministic \n');  
else
    fprintf('Update mode -- Stochastic \n');
end
fprintf('Neuron Function -- %s \n', neuron_type);  

fprintf('System parameters -- Input: %d Epoch: %d LR: %.2f trainset: %d len_bit: %d \n', image_resolution, epoch_number, LR, trainset_length,len_bit);  
fprintf('Device parameters -- dwmin: %f wmax: %.2f Nstates: %d dw_min_std: %.2f \n', device.dw_min, device.wmax, device.Nstates, device.dw_min_std);  

% neuron & error function
switch neuron_type
    case 'sigmoid'
        neuron=@(x)sigmoid(x);
        neuron_d=@(x)sigmoid_d(x);
    case 'relu'
        neuron=@(x)relu(x);
        neuron_d=@(x)relu_d(x);
    case 'tanh'
        neuron=@(x)tanh(x);
        neuron_d=@(x)tanh_d(x);
end

switch error_function
    case 'mean_squared_error'
        c_function=@(x1, x2)mean_squared_error(x1, x2);
    case 'cross_entropy_error'
        c_function=@(x1, x2)cross_entropy_error(x1, x2);
end
%% start training
start_algorithm = tic;
for epoch=1:epoch_number
    start_epoch = tic;
    
    for iteration=1:size(trainset_inputs, 2) % start iteration
        % ===== split inputs into batches =====
        input_batch=trainset_inputs(:, iteration);
        label_batch=trainset_labels(:, iteration);

                % forward-propagation
                out1 = input_batch;
                in2 = (w12 - w12ref) * out1; out2 = neuron(in2); 
                in3 = (w23 - w23ref) * out2; out3 = neuron(in3); 
                in4 = (w34 - w34ref) * out3; out4 = softmax_neuron(in4);
                % backward-propagation
                d4 = label_batch - out4;
                d3 = neuron_d(in3) .* ((w34 - w34ref)' * d4);
                d2 = neuron_d(in2) .* ((w23 - w23ref)' * d3);
                % calculate delta w & update
                    % if epoch < 5
                    %     [w34]  = cal_pulse_new2(w34,device,array.w34,len_bit,d4,out3);
                    % end
                    % if epoch < 10
                    %     [w23]  = cal_pulse_new2(w23,device,array.w23,len_bit,d3,out2);
                    % end 
                [w34]  = stochastic_pulse_module(w34,device,array.w34,len_bit,d4,out3);
                [w23]  = stochastic_pulse_module(w23,device,array.w23,len_bit,d3,out2);
                [w12]  = stochastic_pulse_module(w12,device,array.w12,len_bit,d2,out1);
                
    end

% train accuracy     
out1 = trainset_inputs;
        in2 = (w12 - w12ref) * out1; out2 = neuron(in2); 
        in3 = (w23 - w23ref) * out2; out3 = neuron(in3); 
        in4 = (w34 - w34ref) * out3; out4 = softmax_neuron(in4);
[~, index_output]=max(out4);
[~, index_label]=max(trainset_labels);
Train_acc(epoch)=sum(index_output==index_label)/length(index_output)*100;
Train_E(epoch)=mean(c_function(out4, trainset_labels));
% test accuracy     
out1 = testset_inputs;
        in2 = (w12 - w12ref) * out1; out2 = neuron(in2); 
        in3 = (w23 - w23ref) * out2; out3 = neuron(in3); 
        in4 = (w34 - w34ref) * out3; out4 = softmax_neuron(in4);
[~, index_output]=max(out4);
[~, index_label]=max(testset_labels);
Test_acc(epoch)=sum(index_output==index_label)/length(index_output)*100;
Test_E(epoch)=mean(c_function(out4, testset_labels));

    if epoch>1
        fprintf(repmat('\b', 1, nbytes));   % delete previous command line
    end    
    nbytes = cal_nbytes(Train_acc(epoch), Test_acc(epoch), len_bit, epoch, epoch_number, start_epoch, start_algorithm);

    % LR decay
    if mod(epoch, LR_decay_period) == 0 && len_bit~=1
        len_bit = round(len_bit/2);
    end
end

fprintf('Training done \n');
%% plot accuracies
figure
plot(Train_acc,'-o'); hold on
plot(Test_acc,'-o'); hold off
grid on
% ylim([0 100])

%% functions for the simulations
function y=relu(x)
y=max(0, x);
end
function y=relu_d(x)
y=x>0;
end
function y=sigmoid(x)
y=1./(1+exp(-x));
end
function y=sigmoid_d(x)
y=sigmoid(x).*(1-sigmoid(x));
end
function y=softmax_neuron(x)
x=x-max(x); % substract max value to escape NaN due to too large exp(x)
x_sum=sum(exp(x));
y=exp(x)./x_sum;
end
function error=mean_squared_error(y, t)
error=0.5*sum((y-t).^2);
end
function error=cross_entropy_error(y,t)
d = 1e-8;
error= -sum(t .* log(y+d));
end
function [array] = W_init(NumNeurons,device)
    for i = 2:length(NumNeurons)
        array.(sprintf("w%d%d",i-1,i)).size = [NumNeurons(i), NumNeurons(i-1)];
        array.(sprintf("w%d%d",i-1,i)).bmax = device.wmax + device.w_max_dtod*randn(array.(sprintf("w%d%d",i-1,i)).size); % == array size 
        array.(sprintf("w%d%d",i-1,i)).bmin = device.wmin + device.w_min_dtod*randn(array.(sprintf("w%d%d",i-1,i)).size); 
        array.(sprintf("w%d%d",i-1,i)).bmax(array.(sprintf("w%d%d",i-1,i)).bmax<0) = 0.1; % hard bound condition
        array.(sprintf("w%d%d",i-1,i)).bmin(array.(sprintf("w%d%d",i-1,i)).bmin>0) = -0.1;
        r = exp(device.dw_min_dtod.*randn(array.(sprintf("w%d%d",i-1,i)).size)); % can not be negative
        p = device.up_down + device.up_down_dtod*randn(array.(sprintf("w%d%d",i-1,i)).size);
        array.(sprintf("w%d%d",i-1,i)).slope_p = device.dw_min*(r + p)./array.(sprintf("w%d%d",i-1,i)).bmax;
        array.(sprintf("w%d%d",i-1,i)).slope_n = device.dw_min*(r - p)./array.(sprintf("w%d%d",i-1,i)).bmin;
    end
end

function [w]  = stochastic_pulse_module(w,device,array,len_bit,d,x) % generate stochastic pulse streams and calculate pulse coincidences
A = [length(d), length(x)];
    % x=+ & d=+ case: x*d > 0, so potentiation case
    for i =1:len_bit
        pulse_x = (rand(A(2), 1) < abs(x)); % len(x) 
        pulse_x(x<0) = 0; % make (-) signals zero
        pulse_d = (rand(A(1), 1) < abs(d)); % length(delta)  
        pulse_d(d<0) = 0; % make (-) signals zero
    
        tmp = pulse_d./pulse_x'; % estimating coincidence
        stream = 2*(tmp==1); % coincidence case where x == 1 & d == 1
        w = w_update_new(w, stream, array, device); 
    end
    % x=+ & d=- case: x*d < 0, so depression case
    for i = 1: len_bit
        pulse_x = (rand(A(2), 1) < abs(x)); % len(x) 
        pulse_x(x<0) = 0; % make (-) signals zero
        pulse_d = (rand(A(1), 1) < abs(d)); % length(delta)  
        pulse_d(d>0) = 0; % make (+) signals zero
    
        tmp = pulse_d./pulse_x';
        stream = -2*(tmp==1); % coincidence case where x == 1 & d == 1
        w = w_update_new(w, stream, array, device);
    end
end

function [w] = w_update_new(w, stream, array, device)
    idx=find(stream == 2); % Coincidences for potentiation
        dw = array.slope_p(idx).*(array.bmax(idx)-w(idx))+device.dw_min*device.dw_min_std*randn(length(idx),1);
        w(idx) = w(idx) + dw;
    idx=find(stream == -2); % Coincidences for depression
        dw = -array.slope_n(idx).*(array.bmin(idx)-w(idx))+device.dw_min*device.dw_min_std*randn(length(idx),1);
        w(idx) = w(idx) + dw;
end

function nbytes = cal_nbytes(accuracy_train, accuracy_test, len_bit, epoch, epoch_number, start_epoch, start_algorithm)
    t_this_epoch = toc(start_epoch);
    t_total_min = floor(toc(start_algorithm)/60);
    t_total_sec = mod(floor(toc(start_algorithm)),60);
    t_time_left_min = floor(toc(start_epoch)*(epoch_number-epoch)/60);
    t_time_left_sec = mod(floor(toc(start_epoch)*(epoch_number-epoch)), 60);
    
    fprintf('Training[%d]: %.2f[%%]\tTest: %.2f[%%]\tstream: %d\n',epoch, accuracy_train, accuracy_test, len_bit);
    nbytes = fprintf('Epoch:\t\t%d/%d\nTime:\t\t%.1fs/%um%us (Time_left: %um%us)\n', ...
        epoch, epoch_number, t_this_epoch, t_total_min, ...
        t_total_sec, t_time_left_min, t_time_left_sec);

end
