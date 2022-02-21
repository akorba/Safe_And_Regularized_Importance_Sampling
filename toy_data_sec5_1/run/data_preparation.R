##################NPAIS
ax_size = 1.9
lab_size = 2.6
lwdpoints = 1.7
lwdlines = 3.5
point_size = 2.3
colMH = "grey50"
colOracle = "grey1" 
  
mat_NPAIS_4 = cbind()
mat_NPAIS_3 = cbind()
mat_NPAIS_2 = cbind()
mat_NPAIS_1 = cbind()
mat_NPAIS_ADA = cbind ()
mat_eta_ada = cbind()

for(i in 1:nsim){
  
  temp = err_NPAIS1[[i]][,'NPAIS_norm']
  mat_NPAIS_1 = rbind(mat_NPAIS_1,temp)
  
  temp = err_NPAIS2[[i]][,'NPAIS_norm']
  mat_NPAIS_2 = rbind(mat_NPAIS_2,temp)
  
  temp = err_NPAIS3[[i]][,'NPAIS_norm']
  mat_NPAIS_3 = rbind(mat_NPAIS_3,temp)
  
  temp = err_NPAIS4[[i]][,'NPAIS_norm']
  mat_NPAIS_4 = rbind(mat_NPAIS_4,temp)
  
  temp = err_NPAIS_ADA[[i]][,'NPAIS_norm']
  mat_NPAIS_ADA = rbind(mat_NPAIS_ADA,temp)
  
  temp = err_NPAIS_ADA[[i]][,5]
  mat_eta_ada = rbind(mat_eta_ada,temp)
  }

all_you_need = list(mat_NPAIS_1,mat_NPAIS_2,mat_NPAIS_3,mat_NPAIS_4,mat_NPAIS_ADA,mat_eta_ada)
names(all_you_need) = names_leg
