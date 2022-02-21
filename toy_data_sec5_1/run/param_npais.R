#number of stages and number of samples in each stages
Ttot = 20
Nburnin = round(Ntot / 10)
m = round((Ntot - Nburnin) / Ttot)
nt = c(Nburnin,  rep(m,Ttot))
Ttot = Ttot + 1
Nt = cumsum(nt)
Nt2 = c(0,Nt)

#Resampling rate: only Nt^delta particles should be used to build the kernel estimate 
delta = 1/2
Nt_np = c( 0  , 10 * round( Nt[1:(Ttot-1)]^{delta}  )) 


#lambda and h definitions
h1 = .3 / sqrt (d)                          #this determines the std. The variance is (h1)^2                   
lambda1 = .3                                           
hmem = c( h1 , h1 * (1 + Nt_np[1:(Ttot-1)] / Nt[1]  ) ** (-1/(4 + d)))
lambdamem = c( 1 ,  lambda1 * (Nt[2:(Ttot)] / Nt[1] )** (-2 / (4 + d)) )

#degree of freedom of g for NPAIS and of the student proposal in AIS
df = 3
S_start = (5 / d) * (df-2) / df                          ##careful, the variance is then 5/d


