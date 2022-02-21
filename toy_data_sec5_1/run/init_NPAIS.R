#Key global variables 
# hts : Sequence of length number of blocks containing vectors
#       Scale parameter for non parametric estimation. 
# lambdats : Sequence of length number of blocks probabilities
#            Proportion of data drawn according to g
# x : Array of size (Ntot,d)
#     The n simulated samples. Each sample on a row.
# w : Sequence of length Ntot.
#     Unnormalized weight of each sample, f/qhat in the paper. 
# kyus : Array of length Nt
#        Values of qi(xi) 
# Jf, JUf, JUfalpha, JUmean : 
#    Integral estimates (improved, raw), and same non normalized

# creation of variables
hts = rbind(rep(0,d))          #bandwidth
lambdats = c(1)               #lambda
alpha = c()            #weigths in wNPAIS
alpha2 = c()            #weigths in wNPAIS

w = c()                #to store the weights
w_small = c()
count = 0              #count the number of w>q
data_save = cbind()    #to store some elements of previous stages
res = cbind()              #to store the error

xnew =cbind()

n_eff = 0
JUf = 0                #strore sum of f(xi)
dJUf_small = 0        #strore sum of f(xi)1(w_i>q)
JUfalpha = 0           #strore sum of (f(xi) * weights_i)
JUmean = 0             #strore sum of (xi * w(xi))
Jmean_norm = 0
JU_mean_alpha = 0      #strore sum of (xi * w(xi) * weights_i)
JUsquare = 0           #strore sum of (xi^2 * w(xi))
JUfalpha2 = 0
JU_mean_alpha2 =0
Jmean_norm_cor2 = 0
Jmean_current = 0
Jmean_current_up = 0
Jm = 0

