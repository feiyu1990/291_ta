

%   PreprocessDataSet()
%   Training and test set assembly for training TM ("The Model", Dailey and Cottrell (1999))
%   Author: Panqu Wang
%   This is only a toy version. Do not distribute without permission.
%   12 training images, 4 testing images per individual.


% Finding location of data set.
%loc=['./SampleDataSet'];

clear all;close all; clc;

s = 5; %scale
o = 8; %orientation
im_size = 64; %image size
size_gb=48; %gabor size
k = zeros(1, s); phi = zeros(1, o);
std_dev=pi;
num_pca=8;
dwsp=8;%downsampled rate


root = strcat(pwd, '/../../programming/PA3/NimStim/Crop-White Background/');
files_temp = [dir( fullfile(root,'*.bmp')) ; dir( fullfile(root,'*.BMP'))];   
files = {};
for i=1:length(files_temp)
    if (files_temp(i).name(1) ~= '.')
        files = [files;files_temp(i).name];   
    end
end
label = zeros(length(files),1);

for i=1:length(files)
    label(i) = str2num(files{i}(1:2));
end


[label,I] = sort(label);
files = files(I);

curr = 0;
prev = 0;
for i=1:length(label)
    if(prev ~= label(i))
        curr = curr+1;
        prev = label(i);
        label(i) = curr;
    else
        label(i) = curr;
    end
end
count = [1];
for i =2:length(label)
    if label(i) ~= label(i-1)
        count = [count,1];
    else
        count(end) = count(end)+1;
    end
end

train_indice = ones(length(files),1);
start = 1;
for i=1:length(count)
    indice = randperm(count(i));
    temp = int16(count(i)*1/4);
    for j =1:temp
        train_indice(indice(j)+start-1) = 0;
    end
    start = start + count(i);
end
label_test = label(train_indice==0);
label_train = label(train_indice==1);

total_train = sum(train_indice);
total_test = length(train_indice) - total_train;
f_filtered_normalized_dwsp_vector_allsub_train = zeros(8*8*s*o, total_train);
f_filtered_normalized_dwsp_vector_allsub_test = zeros(8*8*s*o, total_test);
train_i = 1;
test_i = 1;


%% Gabor filter

for i=1:5
    k(i)=(2*pi/im_size)*2^i;
end
%orientation
for i=1:8
    phi(i)=(pi/8)*(i-1);
end
carrier = zeros(size_gb, size_gb);
envelop = zeros(size_gb, size_gb);
gabor = zeros(size_gb, size_gb, o, s);

for scale=1:size(k,2)
    for orientation=1:size(phi,2)
        for ii=-size_gb+1:size_gb
            for j=-size_gb+1:size_gb
                carrier(ii+size_gb,j+size_gb)=exp(1i*(k(scale)*cos(phi(orientation))*ii+k(scale)*sin(phi(orientation))*j));
                envelop(ii+size_gb,j+size_gb)=exp(-(k(scale)^2*(ii^2+j^2))/(2*std_dev*std_dev));
                gabor(ii+size_gb,j+size_gb,orientation,scale)=carrier(ii+size_gb,j+size_gb)*envelop(ii+size_gb,j+size_gb);
            end
        end
    end
end

%%


for i=1:length(files)
    display(['Image ' num2str(i)]);
    file_path = strcat(root, files{i});

    %% for each image, do gabor_filtering
    f=imread(file_path);
    if size(size(f),2)==2
        f=imresize(im2double(f),[im_size im_size]);
    else
        f=rgb2gray(im2double(f));
        f=imresize(f,[im_size im_size]);
    end
    %constructing gabor filter (16*16) and filtering the input image
    for scale=1:size(k,2)
        %scale
        for orientation=1:size(phi,2)
            %subplot(2,4,orientation); imshow(gabor(:,:,orientation),[]);
            %f_filtered{scale}(:,:,orientation)=imfilter(f,gabor(:,:,orientation),'replicate','conv');       
            f_filtered_dwsp{scale}(:,:,orientation)=imresize(imfilter(f,gabor(:,:,orientation,scale),'replicate','conv'),[dwsp,dwsp]);       
            %imshow(f_filtered{scale}(:,:,orientation),[])
        end
        %now we have 8 orientations for each scale, downsample and do normalization
        for orientation=1:size(phi,2)
            f_filtered_normalized_dwsp{scale}(:,:,orientation)=abs(f_filtered_dwsp{scale}(:,:,orientation))./sum(abs(f_filtered_dwsp{scale}),3);               
        end
    end
    %normalize them for each scale
    for scale=1:size(k,2);
        f_filtered_normalized_dwsp_vector((scale-1)*dwsp*dwsp*size(phi,2)+1:(scale)*dwsp*dwsp*size(phi,2))=f_filtered_normalized_dwsp{scale}(:);             
    end
    %testset and trainingset assembly
    if train_indice(i) == 0
        f_filtered_normalized_dwsp_vector_allsub_test(:,test_i)=f_filtered_normalized_dwsp_vector(:);
        test_i = test_i + 1;
        %f_filtered_normalized_dwsp_vector_allsub_test(:,end+1)=f_filtered_normalized_dwsp_vector;
    else
        f_filtered_normalized_dwsp_vector_allsub_train(:,train_i)=f_filtered_normalized_dwsp_vector(:);
        %f_filtered_normalized_dwsp_vector_allsub_train(:,end+1)=f_filtered_normalized_dwsp_vector;
        train_i = train_i + 1;
    end
    clear f_filtered_normalized_dwsp_vector;
end
data.train = f_filtered_normalized_dwsp_vector_allsub_train;
data.test = f_filtered_normalized_dwsp_vector_allsub_test;
data.train_label = label_train;
data.test_label = label_test;
save(['GaborData_mine.mat'],'data')
    %PCA on different scale
for scales=1:size(k,2) 
    scale_all=f_filtered_normalized_dwsp_vector_allsub_train((scales-1)*dwsp*dwsp*size(phi,2)+1:scales*dwsp*dwsp*size(phi,2),:);
    scale_all_test=f_filtered_normalized_dwsp_vector_allsub_test((scales-1)*dwsp*dwsp*size(phi,2)+1:scales*dwsp*dwsp*size(phi,2),:);
    mean_images=mean(scale_all,2);    
        
    %turk and pentland trick, for each scale
    mean_subst=scale_all-repmat(mean_images,1,total_train);
    mean_subst_test=scale_all_test-repmat(mean_images,1,total_test);
    cov_scale=(mean_subst'*mean_subst)*(1/total_train); %(estimate of covariance)
    [vector_temp, value]=eig(cov_scale);
    vector_biggest=vector_temp(:,end-num_pca+1:end);
        
    %original principal components
    vector_ori=mean_subst*vector_biggest;

    %projection onto the basis vector vector_ori(dimension 512-dimension 8)
    %normal
    f_PCA_scale_normal=zscore(vector_ori'*mean_subst);
    f_PCA_scale_test_normal=zscore(vector_ori'*mean_subst_test);
        
%         f_PCA_scale_normal=(vector_ori(:,:,scales)'*mean_subst(:,:,scales));
%         f_PCA_scale_test_normal=(vector_ori(:,:,scales)'*mean_subst_test(:,:,scales));        
        
    f_PCA_temp_normal((scales-1)*num_pca+1:scales*num_pca,:)=f_PCA_scale_normal;
    f_PCA_test_temp_normal((scales-1)*num_pca+1:scales*num_pca,:)=f_PCA_scale_test_normal;
end


data.train = f_PCA_temp_normal;
data.test = f_PCA_test_temp_normal;
data.train_label = label_train;
data.test_label = label_test;


f_PCA_normal_DATASET_train = cell(1,max(label_train));
start = 1;
for i=1:max(label_train)
    c = sum(label_train == i);
    f_PCA_normal_DATASET_train{i}= f_PCA_temp_normal(:,start:start+c-1);
    start = start + c;
end

f_PCA_normal_DATASET_test = cell(1,max(label_test));
start = 1;
for i=1:max(label_test)
    c = sum(label_test == i);
    f_PCA_normal_DATASET_test{i}= f_PCA_test_temp_normal(:,start:start+c-1);
    start = start + c;
end

preprocessedData=[];
preprocessedData(1).trainingSet=f_PCA_normal_DATASET_train;
preprocessedData(1).testSet=f_PCA_normal_DATASET_test;
save(['PreprocessedData_orig.mat'],'preprocessedData')
save(['PreprocessedData_mine.mat'],'data')


train_image = files(train_indice == 1);
test_image = files(train_indice == 0);

information.train= train_image;
information.test = test_image;
save('Image_info.mat','information');

display 'finished.'
