if(func=="mixture"){
  sig_mix = (.4 / sqrt(d))^2                                   #careful this is a variance
  s = .5  /  sqrt(d)                                        
  mu_true = rep(0,d);
  parameters <- list(mean = s * rep(1,d), sd = diag(sig_mix,d))
  if (d == 2) {a = NULL} else {a = rep(0,d-2)}
  mu_start = c(1,-1, a) / sqrt(d)
}

if(func=="NIPS"){
  sig_mix = (.4 / sqrt(d))^2
  mu_true = rep(5,d) / sqrt(d)
  parameters = list(mean = mu_true, sd = diag(sig_mix,d)) 
  mu_start = rep(0,d)
}

if(func=="banana"){
  mu_true = rep(0,d)
  parameters = list(mean = mu_true) 
  mu_start = rep(0,d)
  #vg = (5)^2 * (df-2) / df; 
  #vg2 = vg;
  #sig2 = rep(sqrt(vg2),d);
}