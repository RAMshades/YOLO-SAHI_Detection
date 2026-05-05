%% Parameters to set
% Resolution of the x and y values
px = 0.04;py = 0.04; % values in meters

% folder of cropped images
cropped_image_dir = 'croppedimages';

% features data
feature_data_name = "features_hash.mat";

% size limits
area_limits = [0.2,1.8]; % values in meters

% bound for grouped data for kmeans
step_size = 0.2;
area_groups = 0.4:step_size:1.8;
number_clusters_explore = 20;

% Save cluster data name
save_name = 'Clusters.mat';

%% Hash values from images
features = [];

load(feature_data_name)
hashtog = [features;hvalues];

%% Directory of the images
d = dir(fullfile(cropped_image_dir,'**/*.png'));

plist = {};
for file = 1:length(d)
    plist{file} = string(d(file).folder)+"/"+string(d(file).name);
end

%% Heights and widths of the images
imgheight = [];imgwidth = [];
for file = 1:length(plist)
    info = imfinfo(plist{file});
    imgheight(file) = info.Height;
    imgwidth(file) = info.Width;
    imgarea(file) = imgheight(file)*imgwidth(file);
end

%% Remove data that is outside our desired areas or include an additional check to remove known bad data
removeclass_area = areas>area_limits(1) & areas<area_limits(2);
features = features(removeclass_area,:);
plist = plist(removeclass_area);
imgheight = imgheight(removeclass_area);
imgwidth = imgwidth(removeclass_area);
imgarea = imgarea(removeclass_area);

%% look at histogram and make splits
figure
histogram(imgarea,200)
xlabel('Area (m^2)')
ylabel('Number of Observations')
set(gca,'FontSize',14)
axis tight
grid on

%% group data by data size
iter = 1;
gs = [];
for i = area_groups
    gs(iter,:) = imgarea>=i-step_size & imgarea<i;
    iter=iter+1;
end

%% Kmeans clustering
idk_area = {};
C ={};
tempsumd = [];
avesil = [];
for j = 2:number_clusters_explore
    for i = 1:length(area_groups)
        tempdata = double(features(gs(i,:),:));
        [idk_area{j,i},C{j,i},sumd{j,i}] = kmeans(tempdata,j,'Distance','hamming','Replicates',5);
        tempsumd(j,i) = mean(sumd{j,i});
        s = silhouette(tempdata, idk_area{j,i}, 'Hamming');
        avesil(j,i) = mean(s);
    end
end

%% save information
save(save_name,'avesil','tempsumd',"idk_area","C",'sumd')
