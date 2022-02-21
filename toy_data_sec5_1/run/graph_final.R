

pcha = (1:5)
ltya = rep(1,5)
sel = c(1,2,3,4,5)

rainbowcols = c(rev(brewer.pal(n=9, name = "Blues")[c(5,6,7,8,9)]))
rainbowcols2 = c(rep("lightblue",3),rep("darkorange",1),rep("lightgreen",1))
first_leg = c(1:5)
second_leg = (4:5)
nbr = length(sel)
pcha = pcha[sel]
ltya = ltya[sel]

graph_name = paste(paste(func,"d_mean",d,sep="_"),".pdf",sep="")
t_leg = 0
leg_size = 3

if(func == "mixture" & d ==12){t_leg = 1}
if(func == "rugby" & d ==4 ){t_leg = 1}

place_leg = "bottomleft"
if(func == "NIPS") place_leg = "topright"
names_leg = names_leg[sel]


ax_size = 1.9
lab_size = 2.5
lwdpoints = 3
lwdlines = 4
point_size = 2.6
colMH = "grey50"
colOracle = "grey1" 


y1 = min(data) -  t_leg
y2 = max(data) 
atime = Nt

pdf(file = paste(chemgraph,graph_name,sep=''), onefile=FALSE, width = 10, height = 14) 

par(mai = c(1,1,.1,0.6), cex.axis = ax_size, bg = 'grey95')

plot(x=atime,y=rep((y1-10), length(atime)), xlim = c(atime[1], atime[(Ttot)]), ylim=c(y1,y2),
       col= "white",xlab = "sample size", ylab = "log of MSE", cex.lab = lab_size) 

rect(par("usr")[1], par("usr")[3], par("usr")[2], par("usr")[4], col = 
             "grey90")


for(j in (1:nbr)){
  lines(x=atime, y = data[,j],lwd = lwdlines,col = rainbowcols[j],lty = ltya[j])
  points(x=atime, y=data[,j],lwd = lwdpoints,cex = point_size, col = rainbowcols[j] ,pch = pcha[j])
}


legend(place_leg,names_leg,
       bty = "n", col = rainbowcols, pch = pcha,lty = ltya, 
       cex = leg_size,lwd = lwdpoints)


graphics.off()



graph_name = paste(paste(func,"etas",d,sep="_"),".pdf",sep="")
pdf(file = paste(chemgraph,graph_name,sep=''), onefile=FALSE, width = 10, height = 14) 

ax_size = 1.2
par(mai = c(1,1,.1,0.6), cex.axis = ax_size, bg = 'grey95')

# plot(x=atime,y=rep((y1-10), length(atime)), xlim = c(atime[1], atime[(Ttot)]), ylim=c(0,1),
#      col= "white",xlab = "sample size", ylab = "log of MSE", cex.lab = lab_size) 
sel = 10
BPdata = all_you_need[[6]]
BPdata = BPdata [,(1:sel)]
boxplot(BPdata,ylab = "values of eta",col = "blue",xlab = "sample size", cex.lab = lab_size,names = atime[1:sel])

rect(par("usr")[1], par("usr")[3], par("usr")[2], par("usr")[4], col = 
       "grey90")
boxplot(BPdata,ylab = "values of eta",col = "blue",xlab = "sample size", cex.lab = lab_size,names = atime[1:sel])

graphics.off()
