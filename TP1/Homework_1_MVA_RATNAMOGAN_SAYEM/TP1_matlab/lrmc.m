function  A = lrmc(X,W,tau,beta)
%%% LRMC algrithm
Z=0; 
[U,S,V] = svd(W.*Z); 
m = min(size(S,1),size(S,2));
for i =1:m
    S(i,i) = shrink(S(i,i),tau); 
end 
A = U*S*V' ; 
Z_ = Z; 
Z= Z + beta*(W.*X - W.*A) ;
t=1;
while( norm(Z-Z_) > 1 && t<10000) %stopping conditions 
    [U,S,V] = svd(W.*Z); 
    for i =1:m 
        S(i,i) = shrink(S(i,i),tau); 
    end 
    A = U*S*V' ;  
    Z_ = Z; 
    Z= Z + beta*(W.*X - W.*A) ;
    t=t+1;
    norm(Z-Z_)
end

