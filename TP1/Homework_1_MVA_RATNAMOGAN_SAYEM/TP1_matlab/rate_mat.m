function [M] = rate_mat(db)

    numb_mov = length(unique(db.movieId)); 
    numb_user = length(unique(db.userId)) ;

    M= zeros(numb_mov+1,numb_user+1); 
    M(1,2:(numb_user+1))=unique(db.userId);
    M(2:(numb_mov+1)) = unique(db.movieId); 

    for i=1:length(db.rating)
      movieId = find(M(:,1) == db.movieId(i) ) ;
      userId = find( M(1,:) == db.userId(i) ) ; 
      M(movieId, userId ) = db.rating(i) ; 
    end
    M(1,:)=[];
    M(:,1)=[];
end