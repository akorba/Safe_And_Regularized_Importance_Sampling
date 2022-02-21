# Parameters that you may play with
# func : the (unnormalized) densities used in the paper. Can take the value "cold_start", "mixture", "rugby".
# Ntot : Total number of simulated samples
# nsim : Number of requests to each methods for a given configurations
# nbcor: number of cores in your computer
# d : Dimension of the integration domain

rm(list = ls())

library(mnormt)
library(mvtnorm)
library(parallel)

nbcor = 1
nsim = 50      
Ntot = 4e+5

for(func in c("cold_start","mixture","rugby")){
  for(d in c(4,8,16)){
    setwd("run/")
    source("functions.R")
    source("param_npais.R")
  
    ADA_ETA = TRUE
    err_NPAIS_ADA = mclapply(1:nsim, NPAIS_paral, mc.cores = nbcor)

    ADA_ETA = FALSE
    eta = 1/4
    err_NPAIS1 = mclapply(1:nsim, NPAIS_paral, mc.cores = nbcor)
  
    eta = 2/4
    err_NPAIS2 = mclapply(1:nsim, NPAIS_paral, mc.cores = nbcor)

    eta = 3/4
    err_NPAIS3 = mclapply(1:nsim, NPAIS_paral, mc.cores = nbcor)
  
    eta = 4/4
    err_NPAIS4 = mclapply(1:nsim, NPAIS_paral, mc.cores = nbcor)
  
    setwd("../data")
    file_name = paste(func,"d",d,sep="_")
    save.image(file = paste(file_name,".RData",sep=""))
    setwd("../")
    }
  }
