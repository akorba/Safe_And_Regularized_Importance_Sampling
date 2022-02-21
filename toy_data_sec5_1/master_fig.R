rm(list = ls())
library("RColorBrewer")
chemgraph = "fig/"
chem_data = "data/"


for(func in c("cold_start","mixture","rugby")){
  d = 4
#for(d in c(4,8,16)){

  file_name = paste(func,"d",d,sep="_")
  load(file = paste(chem_data,file_name,".RData",sep=""))
 
  means = cbind()
  for(i in sel){
    means = cbind(means,colMeans(all_you_need[[i]]))
  }

  data = log(means,10)

  source("run/graph_final.R")



}


