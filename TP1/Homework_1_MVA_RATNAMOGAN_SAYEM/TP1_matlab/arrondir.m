function [y] = arrondir(x)
%Arrondit les valeurs de la matrice entre 0 et 5 :
if(x<0.5)
    y=0.5;
else
    if (x>5)
        y=5;
    else 
        if(x-floor(x)>=0.5)
            y=floor(x)+0.5;
        else 
            y=floor(x);
        end
    end
end
end

