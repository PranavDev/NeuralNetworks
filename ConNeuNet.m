% ENTS669D Project #2 Convolutional Neural Network

clc;
close all;

% Loading the training and testing datasets
train_dataset = MNIST_Data_Read('/Users/pranavdeo/Desktop/Datasets/train-images.idx3-ubyte');
test_dataset = MNIST_Data_Read('/Users/pranavdeo/Desktop/Datasets/t10k-images.idx3-ubyte');

% Loading the training and testing labels
Labels_train = MNIST_Labels_Read('/Users/pranavdeo/Desktop/Datasets/train-labels.idx1-ubyte');
Labels_test = MNIST_Labels_Read('/Users/pranavdeo/Desktop/Datasets/t10k-labels.idx1-ubyte');


train_dataset = double(reshape(train_dataset',28,28,[]))/255;
train_dataset = permute(train_dataset, [2 1 3]);
Labels_train = double(Labels_train');

test_dataset = double(reshape(test_dataset',28,28,[]))/255;
test_dataset = permute(test_dataset, [2 1 3]);
Labels_test = double(Labels_test');

disp("######################### CNN Module #########################");
disp(" ");

cnn.layers = {
    % Input Layer
    struct('type','i')
    
    % Convolutional Layer C1
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5)
    
    % SubSampling Layer S1 (avg pooling)
    struct('type', 's', 'scale', 2)
    
    % Convolutional Layer C2
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5)
    
    % SubSampling Layer S2 (avg pooling)
    struct('type', 's', 'scale', 2)
    };

cnn = cnnsetup(cnn, train_dataset, Labels_train);
disp("**CNN SETUP DONE");

learning_rate = 0.5;
batch_size = 150;
n_epochs = 1;

opts.alpha = learning_rate;
opts.batchsize = batch_size;
opts.numepochs = n_epochs;

cnn = cnntrain(cnn, train_dataset, Labels_train, opts);
disp("**DONE TRAINING");

[error_rate, cnn, bad] = cnntest(cnn, test_dataset, Labels_test);
disp("**DONE TESTING");

disp("Accuracy : "+(100-(error_rate*100))+"%");



%############################# FUNCTIONS ##################################

% Function to read the MNIST files and convert them
function MNIST_Images = MNIST_Data_Read(fname)

fileptr = fopen(fname, 'rb');
assert(fileptr ~= -1, ['Unable to open', fname, '']);

magic = fread(fileptr, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Corrupt magic no ', fname, '']);

num_Imgs = fread(fileptr, 1, 'int32', 0, 'ieee-be');
num_Rows = fread(fileptr, 1, 'int32', 0, 'ieee-be');
num_Cols = fread(fileptr, 1, 'int32', 0, 'ieee-be');

MNIST_Images = fread(fileptr, inf, 'unsigned char');
MNIST_Images = reshape(MNIST_Images, num_Cols, num_Rows, num_Imgs);
MNIST_Images = permute(MNIST_Images,[2 1 3]);

fclose(fileptr);

MNIST_Images = reshape(MNIST_Images, size(MNIST_Images, 1) * size(MNIST_Images, 2), size(MNIST_Images, 3));
MNIST_Images = double(MNIST_Images) / 255;

end



% Function to read the MNIST labels and convert them
function Labels = MNIST_Labels_Read(fname)

fileptr = fopen(fname, 'rb');
assert(fileptr ~= -1, ['Could not open ', fname, '']);

magic = fread(fileptr, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Corrupt magic number in ', fname, '']);

number_of_Labels = fread(fileptr, 1, 'int32', 0, 'ieee-be');
Labels = fread(fileptr, inf, 'unsigned char');
assert(size(Labels, 1) == number_of_Labels, 'Mismatch in label count');

fclose(fileptr);

end

%#########################################################################