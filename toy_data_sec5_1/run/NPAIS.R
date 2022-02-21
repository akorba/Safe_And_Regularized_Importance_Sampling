source("init_NPAIS.R")

iter = 0

for (i in (1:Ttot)){
  lambdat = lambdamem[i]
  ht = rep(hmem[i],d)
  
  source("expl_update_NPAIS_fast.R")
    rest=c(NPAIS_norm = err_norm , n_eff = n_eff, bdw = hmem[i], lbd = lambdat,etav)
  res=rbind(res,rest)

  
}

