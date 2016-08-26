load digitStruct.mat

s = struct([]);
lable_im_ = [];

for i = 1:13010
    if i == 174
        continue
    elseif i == 4381
        continue
    end
    
    a = [];
    b = [];
    c = [];
    d = [];
    im = imread([digitStruct(i).name]);
    im = rgb2gray(im);
    for j = 1:length(digitStruct(i).bbox)
       [height, width] = size(im);
       aa = max(digitStruct(i).bbox(j).top+1,1);
       a = [a, aa]; 
       A = min(a);
       bb = min(digitStruct(i).bbox(j).top+digitStruct(i).bbox(j).height, height);
       b = [b, bb];
       B = max(b);
       cc = max(digitStruct(i).bbox(j).left+1,1);
       c = [c, cc];
       C = min(c);
       dd = min(digitStruct(i).bbox(j).left+digitStruct(i).bbox(j).width, width);        
       d = [d, dd];
       D = max(d);
    end
    
    
    IM = im(A:B, C:D, :);
    S = imresize(IM, [54,54]);
    I = double(reshape(S, [1, 54*54]));
    digitStruct(i).name    
    
label_ = [digitStruct(i).bbox.label];
label_(label_ == 10) = 0;
label_1 = [str2num(sprintf('%d',label_))];
lable_im = cat(2, label_1, I);
lable_im_ = cat(1,lable_im_, lable_im);
    
end

csvwrite('test_fun.csv',lable_im_)



