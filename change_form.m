tmp = load('/home/hpc/ssd/lyj/liu_data/action_training_nobg.mat');
data = tmp.posAction;
dataset = [];
dataset.image = [];
dataset.size = [];
dataset.joints = [];
count = 0;
dict = [13,12,8,7,6,2,1,0,9,10,11,3,4,5];
file_prefix = '/home/hpc/ssd/lyj/liu_data/';
height = 256;
width = 256;

for num_list = 1:length(data)
    for num_file = 1:length(data{1,num_list})
        count = count + 1;
        dataset(count).image = [file_prefix,data{1,num_list}(num_file).fileName];
        tmp_img = imread(dataset(count).image);
        tmp_img = imresize(tmp_img,[256,256]);
        imwrite(tmp_img,dataset(count).image)
        h = data{1,num_list}(num_file).height;
        w = data{1,num_list}(num_file).width;
        x_scale = 256/w;
        y_scale = 256/h;
        dataset(count).size = [3,256,256];
        cur_point = data{1,num_list}(num_file).point;
        tmp = zeros(14,3);
        for num_joint = 1:14
            tmp(num_joint,:) = [dict(num_joint),cur_point(num_joint,1)*x_scale,cur_point(num_joint,2)*y_scale];
        end
        dataset(count).joints{1} = tmp;
    end
end
save('/home/hpc/ssd/lyj/liu_data/dataset.mat','dataset')
