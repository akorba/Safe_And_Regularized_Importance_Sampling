f2=function(z){
  #bphi=0
  xx=z
  xx[,1]=z[,1]/sigphi
  xx[,2]=z[,2]-bphi*(z[,1]**2-sigphi**2)
  return(.5*CompK(xx)/sigphi + .25*CompK(sweep(z,2,c1,FUN="-")/2)/2^d + .25*CompK(sweep(z,2,c2,FUN="-")/2)/2^d
  )
}

#f3 is the Gaussian of the NIPS paper
f3 = function(x) {
  yy = sweep(x,2,mu_true,FUN="-")
  yy = sweep(yy,2,diag(sigma_true),'/')
  return(CompK(yy) / prod(diag(sigma_true)))
  }




#evaluation of normal density
CompK=function(z){
  y = rowSums(z**2)/2
  y = exp(-y)/(2*pi)**(dim(z)[2]/2)
  return(y)
}

##for PAWL 
logdensity_mixture <- function(x, parameters) log(  .5*dmvnorm(x, -parameters$mean, parameters$sd, log = FALSE) +
                                                   .5*dmvnorm(x, parameters$mean, parameters$sd, log = FALSE) )

mixture <- function(x, parameters)   p * dmvnorm(x, -parameters$mean, parameters$sd, log = FALSE) +
                                          (1-p) * dmvnorm(x, parameters$mean, parameters$sd, log = FALSE) 


logdensity_cs = function(x, parameters) dmvnorm(x, parameters$mean, parameters$sd, log = TRUE)
cs = function( x, parameters)  dmvnorm(x, parameters$mean, parameters$sd, log = FALSE)

rugby = function( x, parameters)  dmvnorm(x, parameters$mean, parameters$sd, log = FALSE)



####################################################################
################specify which f we take
                      


if(func == "cold_start"){f = cs; logf = logdensity_cs
sig_mix = (.4 / sqrt(d))^2
mu_true = rep(5,d) / sqrt(d)
parameters = list(mean = mu_true, sd = diag(sig_mix,d)) 
mu_start = rep(0,d)
}
 


if(func == "mixture"){f = mixture;logf = logdensity_mixture
p=.5
sig_mix = (.4 / sqrt(d))^2                                   #careful this is a variance
s = 1 / (2 * sqrt(d))                                        
mu_true = rep(0,d);
parameters <- list(mean = s * rep(1,d), sd = diag(sig_mix,d))
if (d == 2) {a = NULL} else {a = rep(0,d-2)}
mu_start = c(1,-1, a) / sqrt(d)

} #vg2 = vg;sig2 = rep(sqrt(vg2),d);}

if(func == "mixture2"){f = mixture;logf = logdensity_mixture
p = .5
sig_mix = (.4 / sqrt(d))^2                                   #careful this is a variance
s = 1 / (sqrt(d))                                        
mu_true = rep(0,d);
parameters <- list(mean = s * rep(1,d), sd = diag(sig_mix,d))
if (d == 2) {a = NULL} else {a = rep(0,d-2)}
mu_start = c(1,-1, a) / sqrt(d)

} #vg2 = vg;sig2 = rep(sqrt(vg2),d);}

if(func == "rugby"){f = mixture;
p = 1/4
sig_mix = (.4 / sqrt(d))^2                                   #careful this is a variance
s = 1 / (2 * sqrt(d))                                        

SIG =  diag(sig_mix,d)
SIG[1,1] = SIG[1,1] * 10
parameters <- list(mean = s * rep(1,d), sd = SIG)
mu_true = - parameters$mean * p + parameters$mean * (1-p) 
if (d == 2) {a = NULL} else {a = rep(0,d-2)}
mu_start = c(1,-1, a) / sqrt(d)

} #vg2 = vg;sig2 = rep(sqrt(vg2),d);}


####################################################################
###########to use parallel package, the call is made using a function
NPAIS_paral = function(i){
  source("NPAIS.R")
  return(res)
}