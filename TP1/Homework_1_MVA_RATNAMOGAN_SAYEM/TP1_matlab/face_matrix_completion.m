clear all

%percentage of missing entries
ratios =0.20:0.20:0.80;

%We collect all the 10 images on one matrix
for i=1:10
     img_temp = imresize(loadimage(1,i),0.5); 
     img(:,i) = img_temp(:);
end

%defining parameters of lrmc
D = size(img,1); 
N = size(img,2); 

%one resulting matrix for each percentage/ratio
A = zeros(D,N,4);
img_ = zeros(D,N,4);
     
     
for j=1:4
    ratio = ratios(j);
    %random selection for missing entries
    W = binornd(1,1-ratio , D, N);
    %number of observed entries
    M = size(img,2)*size(img,1)*(1-ratio);
    %parameters
    beta = min(2,D*N/M);
    tau = 5*10^4;
    %deleting some entries
    img_(:,:,j) = img.*W ;
    
    %applying LRMC
    A(:,:,j) = lrmc(img_(:,:,j), W, tau, beta);
    
    %mean square error for each percentage
    mse(j) = sum (sum ((img-A(:,:,j)).^2))/(N*D);
    
end


O = reshape(img_(:,1,2),[96 84]);
R=reshape(A(:,1,2),[96 84]);
imshowpair(O, R , 'montage')

           
