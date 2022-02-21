if(i > 1){
  iter = iter + 1
  ml = rmultinom(1, Nt_np[i], w / JUf)
  data_save[,"weights2"] = ml
  data_reduced = rbind(data_save[data_save[,"weights2"]>0,])
  n_eff =  dim(data_reduced)[1]
  xn_small = rbind(data_reduced[,(1:d)])
  wn_small = data_reduced[,"weights2"]
    
}

xnew = cbind()

# One iteration of Explore-Update
fhat = matrix(0,nt[i],1)

if (lambdat==1){
  xnew = rmt(nt[i] , mean = Jmean_current, S=diag(S_start,d), df=df)
  gx = dmt(xnew,  mean = Jmean_current, S=diag(S_start,d), df=df)
  kyun = gx
} else {
  Ng = rbinom(1,nt[i], lambdat) # Number of g variables
  n1 = nt[i] - Ng
  wn_norm = wn_small / sum(wn_small)
  if (n1>0){
    ml = rmultinom(1,n1,wn_norm)
    ml = rep(1:length(ml),ml)
    xnew = rbind(xn_small[ml,])
    xnew = matrix(rnorm(d*n1),nrow=n1) * ht + xnew[,1:d]
    } 
  if (Ng>0){
    xnew2 = rmt(Ng , mean = Jmean_current, S=diag(S_start,d), df=df)
    xnew = rbind(xnew,xnew2)
  }
  for (j in (1:nt[i])){
    yy = sweep(rbind(xn_small[,(1:d)]),2,xnew[j,],'-')
    yy = yy / ht
    fhat[j] = sum(wn_norm * CompK(yy) / prod(ht) )  
  }
  gx = dmt(xnew,  mean=Jmean_current, S=diag(S_start,d), df = df)
  kyun = (1 - lambdat) *  (fhat)  + lambdat * gx
}

fx = f(xnew,parameters)
wn = as.vector(fx / (kyun ) )

if(ADA_ETA == TRUE){
alph = .5
wnorm = wn / sum(wn)
wnorm = wnorm[wnorm>0]
n_w = length(wnorm)
etavKL =  min( 1/ (2 * sqrt (d) ) - sum( wnorm * log(wnorm)  ) / ( log( n_w ) )  , 1)
etav = min( 1/ (2 * sqrt (d) )  + 1 - log( sum (wnorm ^ alph * (1/n_w) ^ (1-alph)) ) / ( alph -1 )  / log(n_w) ,1) 
} else {etav = eta}

wn = wn^{etav}                     

dJUf = sum(wn)
JUf = JUf + dJUf
dJUmean = colSums(xnew * wn)
JUmean = JUmean + dJUmean
Jmean_norm = JUmean / JUf

Jmean_current = dJUmean / dJUf
err_norm = sum((Jmean_norm - mu_true)^2)

#checkpoints
w = c(w,wn)
datan = cbind(x = xnew, step = i, weights = wn,weights2 = 0)
data_save = rbind(data_save, datan)


