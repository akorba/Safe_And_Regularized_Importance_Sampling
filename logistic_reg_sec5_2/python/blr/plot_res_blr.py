import matplotlib.pyplot as plt
import numpy as np
import glob
import re
import os

def read_res(filename):
    f1 = open(filename, 'r')
    fileline = f1.read()
    fileline = re.split(' |\n', fileline)
    return np.array([float(fileline[j]) for j in range(len(fileline)-1)])


def wrapper_function(directory, T, strName2):
    f_renyi = glob.glob(os.path.join(directory,strName2))
    nb_repeat = len(f_renyi)
    summation = np.zeros(T)

    for i in range(nb_repeat):
        file_read = read_res(f_renyi[i])
        summation = summation + np.array(file_read)
        print(f_renyi[i])
        #print(summation)

    summation = summation / nb_repeat

    return summation

#### Parameters for RIS ####
covertype = False

if covertype:
    dim_latent=56
    freq_eval = 50
    str_plot = '(Covertype dataset)'
else:
    dim_latent=22
    freq_eval = 10
    str_plot = '(Waveform dataset)'

N_tot = 51
nb_output = int(N_tot/freq_eval) +1
eta_0_init = 1 / (2 * np.sqrt(dim_latent))
eta_0_list = [eta_0_init, 1.]
alpha_list = [0.05, 0.08, 0.1, 0.15, 0.2, 0.3]
batch_init = 2000
batch_size = 200
str_params = 'Ntot_' + str(N_tot) + '_batch_init_' + str(batch_init) + '_batch_size_' + str(batch_size)

#### Plot style ####
plt.style.use('bmh')
plt.rcParams['figure.facecolor'] = '1'
plt.rcParams['axes.facecolor'] = 'whitesmoke'
plt.rcParams["legend.loc"] = 'lower right'#'upper left'

#### Plot the results #####
plt.figure()
plt.title('Dimension ' + str(dim_latent) + ' ' + str_plot)
plt.xlabel('sample size')
plt.ylabel(r'Accuracy')
plt.legend(loc='lower right')
directory_to_results = './python/blr/results/'


numlines = len(alpha_list)
range_val = np.linspace(0.1,0.7, numlines)

sample_size_for_plot = []
nb_samples = batch_init
for j in range(nb_output):
    sample_size_for_plot.append(nb_samples)
    nb_samples += freq_eval*batch_size

for eta_0 in eta_0_list:
    directory = directory_to_results +'dim' + str(dim_latent) + "/eta" + str(eta_0) + '/'

    if eta_0 == eta_0_init:
        i=0
        for alpha in alpha_list:
            directory = directory_to_results + "dim" + str(dim_latent) + "/eta" + str(eta_0) + '/alpha' + str(alpha) + '/'

            out = wrapper_function(directory, nb_output, 'f_ris_eval' + str_params + '*.txt')
            plt.plot(sample_size_for_plot, out, label=r'$\alpha$ = ' + str(alpha), c=plt.cm.plasma(range_val[i]), marker='.')
            i+=1

    else:
        out = wrapper_function(directory, nb_output, 'f_ris_eval' + str_params + '*.txt')

        if eta_0 == 1.:
            plt.plot(sample_size_for_plot, out, label=r'SAIS', marker='+')
        else:
            plt.plot(sample_size_for_plot, out, label=r'$\eta_k$ = ' + str(eta_0), marker='+')

plt.legend()
plt.show()

eta_0 = eta_0_init

plt.figure()
plt.title(r"Evolution of $(\eta_{k, \alpha})_{k \geq 1}$ " + str_plot)
index = np.array(range(N_tot))[::freq_eval] - 1
i=0
for alpha in alpha_list:

    directory = directory_to_results + "dim" + str(dim_latent) + "/eta" + str(eta_0) + '/alpha' + str(alpha) + '/'
    out_eta = wrapper_function(directory, N_tot, 'f_ris_eta' + str_params + '*.txt')
    plt.plot(sample_size_for_plot, np.array(out_eta)[::freq_eval], label= r'$\alpha$ = '+ str(alpha), c=plt.cm.plasma(range_val[i]), marker='.')
    i +=1
    
    
plt.legend(loc='upper right')
plt.xlabel('sample size')
plt.ylabel(r'$\eta_{k,\alpha}$')
plt.show()
