% ENTS669D Project #2 SVM
clc;
close all;

%get train and test image files
train_images = MNIST_Data_Read('/Users/pranavdeo/Desktop/Datasets/train-images.idx3-ubyte');
test_images = MNIST_Data_Read('/Users/pranavdeo/Desktop/Datasets/t10k-images.idx3-ubyte');
train_images = train_images';
test_images = test_images';

%get train and test label files
labels_train = MNIST_Labels_Read('/Users/pranavdeo/Desktop/Datasets/train-labels.idx1-ubyte');
labels_test = MNIST_Labels_Read('/Users/pranavdeo/Desktop/Datasets/t10k-labels.idx1-ubyte');

disp("######################### SVM Module #########################");
disp(" ");
ch = input("## Enter:  1. PCA-SVM   2. LDA-SVM   3.Linear-SVM  : ");


if ch == 1
    
    st = "PCA";
    flag = 0;
    
    % Applying PCA before SVM
    [coeff1, score1] = pca(train_images,'Algorithm','svd');
    [coeff2, score2] = pca(test_images,'Algorithm','svd');
    
    X_train = score1 * coeff1';
    X_test = score2 * coeff2';
    
    % Model buliding using SVM
    pca_svm_model = fitcecoc(X_train,labels_train);
    disp("> TRAINING COMPLETE");
    disp(" ");
    
    % Getting the Prediction Labels for the testing dataset using model
    final_labels = predict(pca_svm_model,X_test);
    disp("> TESTING COMPLETE");
    disp(" ");

    
elseif ch == 2
    
    st = "LDA";
    flag = 0;
    
    train_m1 = fitcdiscr(train_images,labels_train, 'DiscrimType','pseudoquadratic');
    test_m2 = fitcdiscr(test_images,labels_test, 'DiscrimType','pseudoquadratic');
    X_train = train_m1.X;i
    l_train = train_m1.Y;
    X_test = test_m2.X;
    l_test = test_m2.Y;
    
    lda_svm_model = fitcecoc(X_train,l_train);
    disp("> TRAINING COMPLETE");
    disp(" ");
    
    % Getting the Prediction Labels for the testing dataset using model
    final_labels = predict(lda_svm_model,X_test);
    disp("> TESTING COMPLETE");
    disp(" ");
    
    
elseif ch == 3
    
    st = "Linear-SVM";
    flag = 0;
    
    linear_svm_model = fitcecoc(train_images,labels_train);
    disp("> TRAINING COMPLETE");
    disp(" ");
    
    % Getting the Prediction Labels for the testing dataset using model
    final_labels = predict(linear_svm_model,test_images);
    disp("> TESTING COMPLETE");
    disp(" ");
    
else
    disp("## EXIT ##");
    flag = 1;
end


if flag == 0
    
    % Checking for accuracy of actual and predicted 
    match = 0;
    for i=1:10000
        if final_labels(i) == labels_test(i)
            match = match + 1;
        end
    end

    accuracy = (match / 10000) * 100;
    disp("*** Accuracy of "+st+" is: "+accuracy+"% ***");
    
else
    disp("*** TERMINATED ***");
end

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