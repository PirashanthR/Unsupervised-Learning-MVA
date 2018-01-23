clear all
mdb = readtable('db.csv') ; 

r = 0.20;

%Horror = mdb(mdb.genreId==1,:);
%Romace = db(db.genreId==2,:);

db=mdb;
X=rate_mat(db);
[row,col,v] = find(X);

% Number of real ratings
l=length(row); 

% ratio of deleted data that will be used to validate
r = 0.2 ; 

% random selection of training set
pos_zeros_r  = unidrnd(max(row),floor(r*l),1) ; 
X_train = X;
for i=1:floor(r*l)
    id = pos_zeros_r(i);
    X_train(row(id),col(id)) = 0;
end

W = (X_train~=0) ;
tau = 10000;
beta = 0.2;

A = lrmc(X_train,W,tau,beta) ;
R = arrondir_mat(A);

%validation : 
for i=1:floor(r*l)
    d(i) = abs(R(row(i),col(i)) - v(i));
end
