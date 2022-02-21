# README #
This README would normally document whatever steps are necessary to get similar figures to the one of Section 5.1 in the paper



### How do I get set up? ###

Before starting be sure that the working directory is the one of the files ''master_run.R'' and ''master_fig.R''



### How to run the code  ###

Run the file ''master_run.R''

> source(master_run.R) 

This might take some time depending on the number of cores (nbcor = 1 is the default value) and the number of  requests to each methods (nsim = 50 is the default value)
Those are defined at the beginning of the file ''master_run.R''

After running, check that the folder ''data'' now contains several files ''.RData'' 
Those files contain the output of all the algorithms



### How to get the figures ###
Check that your working directory is still the one of the files ''master_run.R'' and ''master_fig.R''

Run the file ''master_fig.R''

> source(master_fig.R) 

After running, the folder fig should be provided with all the graphs of the paper

