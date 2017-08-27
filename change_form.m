% tmp = load('/home/hpc/Downloads/lspet_dataset/joints.mat');
% data = tmp.joints;
% dataset = [];
% dataset.image = [];
% dataset.size = [];
% dataset.joints = [];
% count = 0;
% dict = [0,1,2,3,4,5,6,7,8,9,10,11,12,13];
% file_prefix = '/home/hpc/Downloads/lspet_dataset/images/';
% 
% % get 9000 training data
% for i = 1:9000
%     dataset(i).image = [file_prefix,'im',num2str(i+1000,'%05d'),'.jpg'] ;
%     img_size = size(imread(dataset(i).image));
%     dataset(i).size = [3,img_size(1:2)];
%     tmp = [];
%     data_tmp = data(:,:,i+1000);
%     for j= 1:14
%     % erase the occuluded joint
%         if  data_tmp(j,3) == 0
%             continue;
%         end
%         tmp = cat(1,tmp,[dict(j),data_tmp(j,1:2)]);
%     end
%     dataset(i).joints{1} = tmp;
% end
% 
% tmp_ex = load('./lsp_2000_gt.mat');
% data_ex = tmp_ex.lsp_gt;
% file_prefix_ex = '/home/hpc/Downloads/lsp_dataset/img_resized/';
% 
% % add extra 2000 training data
% for i = 1:2000
%     dataset(i+9000).image = [file_prefix_ex,'im',num2str(i,'%04d'),'.jpg'] ;
%     img_size = size(imread(dataset(i+9000).image));
%     dataset(9000+i).size = [3,img_size(1:2)];
%     tmp = [];
%     data_tmp = data_ex(:,:,i);
%     for j= 1:14
%     % erase the occuluded joint
%         if  data_tmp(j,3) == 0
%             continue;
%         end
%         tmp = cat(1,tmp,[dict(j),data_tmp(j,1:2)]);
%     end
%     dataset(i+9000).joints{1} = tmp;
% end
% save('lsp_train.mat','dataset');
% 
% % get 1000 testing data
% lsp_gt = zeros(14,3,1000);
% lsp_gt(:,:,:) = data(:,:,1:1000);
% save('lsp_test_1000.mat','lsp_gt');

tmp = load('/home/hpc/ssd/lyj/liu_data/action_training_nobg.mat');
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
        
%         dataset(count).size = [3,data{1,num_list}(num_file).height,data{1,num_list}(num_file).width];
        h = data{1,num_list}(num_file).height; 
        w = data{1,num_list}(num_file).width; 
        x_scale = 256 / w; 
        y_scale = 256 / h;
        dataset(count).size = [3,256,256];
        
        cur_point = data{1,num_list}(num_file).point;
        tmp = zeros(14,3);
        for num_joint = 1:14
%             tmp(num_joint,:) = [dict(num_joint),cur_point(num_joint,1),cur_point(num_joint,2)];
            tmp(num_joint,:) = [dict(num_joint),cur_point(num_joint,1) * x_scale,cur_point(num_joint,2)*y_scale];
        end
        dataset(count).joints{1} = tmp;
    end
end
save('cus_train.mat','dataset')


% tmp = load('/home/hpc/ssd/lyj/liu_data/action_testing_nobg.mat');
% dict = [14,13,9,8,7,3,2,1,10,11,12,4,5,6];
% data = tmp.posAction;
% count = 0;
% for num_list = 1:length(data)
%     for num_file = 1:length(data{1,num_list})
%         cur_point = data{1,num_list}(num_file).point;
%         tmp = zeros(14,2);
%         
%         h = data{1,num_list}(num_file).height; 
%         w = data{1,num_list}(num_file).width; 
%         x_scale = 256 / w; 
%         y_scale = 256 / h;
% 
%         for num_joint = 1:14
% %             tmp(dict(num_joint),1:2) = cur_point(num_joint,1:2);
%             tmp(dict(num_joint),1) = cur_point(num_joint,1) * x_scale;
%             tmp(dict(num_joint),2) = cur_point(num_joint,2) * y_scale;
%         end
%         if count == 0
%             gt = tmp;
%         else
%             gt = cat(3,gt,tmp);
%         end
%         count = count + 1;
%     end
% end
% % save('test_cus_original.mat','gt');
% save('test_cus_256.mat','gt');