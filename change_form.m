tmp = load('action_training_nobg.mat'); 
data = tmp.posAction;
dataset = [];
dataset.image = [];
dataset.size = [];
dataset.joints = [];
count = 0; 
dict = [13,12,8,7,6,2,1,0,9,10,11,3,4,5];
file_prefix = '/home/hpc/ssd/lyj/liu_data/';

for num_list = 1:length(data)
    for num_file = 1:length(data{1,num_list})
        count = count + 1;        
        dataset(count).image = [file_prefix,data{1,num_list}(num_file).fileName]; 
        dataset(count).size = [3,data{1,num_list}(num_file).height,data{1,num_list}(num_file).width];
        cur_point = data{1,num_list}(num_file).point;
        tmp = zeros(14,3);
        for num_joint = 1:14 
            tmp(num_joint,:) = [dict(num_joint),cur_point(num_joint,1),cur_point(num_joint,2)];
        end 
        dataset(count).joints{1} = tmp;
    end
end
save('dataset.mat','dataset')

