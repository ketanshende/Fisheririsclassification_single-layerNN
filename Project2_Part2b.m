%% Training

clear all; close all; clc;

load 'fisheriris'
input_layers = 4;
hidden_layers = 5;
output_layers = 3;

training_data=[];
desired_train_output=[];
w_ih=rand(input_layers,hidden_layers);
w_bhid=rand(hidden_layers,1);
w_ho=rand(hidden_layers,output_layers);

learning_rate = 0.01;
% coding of 3 different classes
class1 = [1 0 0]';
class2 = [0 1 0]';
class3 = [0 0 1]';

for i=1:17   % 1:17 1:25 1:33 1:42
     template=[meas(i,:);meas(50+i,:);meas(100+i,:)]';
     training_data=[training_data template];
     desired_train_output = [desired_train_output,class1,class2,class3];
end

iteration=20000;
error=zeros(output_layers,iteration);

for iter=1:iteration
        
        %estimated output
        op_w=training_data'*w_ih;
        
        op_sig=1./(1+exp(-(op_w)));
        
        p=op_sig*w_ho;
        
        out2=1./(1+exp(-(p))); 
        e=-(desired_train_output'-out2);
        delta=(out2.*(1-out2)).*e;
        delta_hid=op_sig'.*(1-op_sig)'.*(w_ho*delta');
        
        %hidden layer weights updation
        
        w_ho=w_ho-learning_rate.*op_sig'*delta;
                
        %input layer weight updations
        
        w_ih=w_ih-learning_rate*(training_data*delta_hid');  

        op_w=training_data'*w_ih;
        
        op_sig=1./(1+exp(-(op_w)));
        
        p=op_sig*w_ho;
        
        out2=1./(1+exp(-(p)));

        e=-(desired_train_output'-out2);
        
        % convergence verification
        if(norm(e)<2)
            break;
        end
    iter;
    
    
end

% Find misclassifications for training
out1 = [];
for i=1:size(training_data,2)
    op_w=training_data(:,i)'*w_ih;
    op_sig=1./(1+exp(-(op_w+w_bhid')));
    out1(:,i)=(1./(1+exp(-(op_sig*w_ho))))';                       
end

out1(out1==max(out1))=1;      % Make greatest propability equal to 1
out1(out1~=1)=0;              % Make lowest probabilities equal to 0
linear_index = find(out1(:,:)~=desired_train_output(:,:));   % Find mismatches (linear indices)
s = size(out1);

% Find index of mismatch between output and desired test output

[I1,J1] = ind2sub(s,linear_index);
% disp(out1);
disp('Misclassifications (Training): ');
disp(length(J1)/2) % since one column mismatch implies two of the elements in the column are
                  % mismatching, divide by two to get rid of repeated
                  % column mismatches 


%% Testing
sse=sum((error(:,1:iter).^2),1);
testing_data = [meas(18:50,:);meas(68:100,:);meas(118:150,:)]'; % Case 1, 99 test samples
% testing_data = [meas(26:50,:);meas(76:100,:);meas(126:150,:)]'; % Case 2, 75 test samples
% testing_data = [meas(34:50,:);meas(84:100,:);meas(134:150,:)]'; % Case 3, 51 test examples
% testing_data = [meas(43:50,:);meas(93:100,:);meas(143:150,:)]'; % Case 4, 24 test samples
size_test = size(testing_data);
desired_test_out = [repmat(class1,[1,max(size_test)/3]),...
    repmat(class2,[1,max(size_test)/3]),repmat(class3,[1,max(size_test)/3])];
out2 = [];
for i=1:size(testing_data,2)
    op_w=testing_data(:,i)'*w_ih;
    op_sig=1./(1+exp(-(op_w+w_bhid')));
    out2(:,i)=(1./(1+exp(-(op_sig*w_ho))))'; 
end
% disp("Convergence Output :")
% disp(out2)

% Find misclassifications

out2(out2==max(out2))=1;   % Make greatest propability equal to 1
out2(out2~=1)=0;          % Make lowest probabilities equal to 0
linear_index = find(out2(:,:)~=desired_test_out(:,:));   % Find mismatches (linear indices) 
s = size(out2);

% Find index of mismatch between output and desired test output

[I2,J2] = ind2sub(s,linear_index);    % find mismatches (matrix indices)
% disp(out2)
disp('Misclassifications (Testing): ');
disp(length(J2)/2) % since one column mismatch implies two of the elements in the column are
                  % mismatching, divide by two to get rid of repeated
                  % column mismatches 
disp('Training Iterations for Convergence: ');
disp(iter)
