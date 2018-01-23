function [s] = shrink(y,tau)
if ( y >= tau)
    s = y-tau; 
else 
    if ( y <= -tau)
        s = y+tau ; 
    else 
        s = 0 ; 
    end 
end
end

