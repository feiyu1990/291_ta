function h=tanh_mine(a)
  sigmoid=1./(1+exp(-2.*a));
  h = 2*sigmoid - 1;
end