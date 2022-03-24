clc;clear
addpath('./SIFTflow')
addpath('./SIFTflow/mexDenseSIFT')
addpath('./SIFTflow/mexDiscreteFlow')

ori = 'grid/';
recti = 'scan/'; 
File = dir(fullfile(ori,'*.png')); 
FileNames = {File.name}';
nn = 130; 
ms = zeros(nn,1);ld = zeros(nn,1);
for i = 1:nn
    name = FileNames{i};
    name1 = strsplit(name,'_');
    z = [ori,name]
    A = imread([ori,name]);ref = imread([recti,name1{1},'.png']);
    b = [880,680];
    A = imresize(A,b);ref = imresize(ref,b);

    mss = zeros(3,1);lds = zeros(3,1);
    for s = 1:3
        [mss(s), lds(s)] = evalUnwarp(A(:,:,s), ref(:,:,s));
    end
    ms(i) = mean(mss(:));
    ld(i) = mean(lds(:));
    [i ,ms(i),ld(i)]
end
[mean(ms(:)),mean(ld(:))]

 