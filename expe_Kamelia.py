#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
import numpy.matlib as nm
from scipy.stats import gamma, multivariate_normal, entropy
from scipy.io import loadmat
from functools import partial
from joblib import Parallel, delayed
from pathlib import Path
import numpy as np
from scipy.stats import multivariate_normal

### Load Waveform Dataset ###
matdata=scipy.io.loadmat('waveform.mat');
X_train = matdata['X_train']
X_test = matdata['X_test']
y_train = np.int_(np.squeeze(matdata['y_train']))
y_test = np.int_(np.squeeze(matdata['y_test']))

y_train[y_train == 0] = -1
y_test[y_test == 0] = -1

X_train = np.delete(X_train, 0, axis=1)
X_test = np.delete(X_test, 0, axis=1)

def sample_from_k_theta_1d(mean, sd, nb_samples):
    return np.random.normal(mean, sd, nb_samples)


class SRAIS:

    def __init__(self, i, model_lnprob, func_eval, D, N_tot, h_k, eta_k, lambda_k, q0_mean, q0_sd, batch_size,
                 batch_init, bool_bootstrap, batch_bootstrap, freq_eval, RAR):
        '''
        Initialise the parameters of the Safe and Regularized Adaptive Importance Sampling (SRAIS) algorithm
        :param i: used for paralellisation (int)
        :param lnprob: function which computes the log prob of a given model (func)
        :param func_eval: function that evaluates the performances of the algorithm (func)
        :param D: dimension of the latent space (int)
        :param N_tot: total number of iterations (int)
        :param h_k: function returning the bandwidths (func)
        :param eta_k: function returning the mixture weights policy (func)
        :param lambda_k: function returning the regularisation parameter according to the policy (func)
        :param q0_mean: initial sampler mean (vector of dimension d)
        :param q0_sd: initial sampler standard deviation (float > 0)
        :param batch_size: batch_size at time k \geq 1 (int)
        :param batch_init: initial batch size (int)
        :param bool_bootstrap: (boolean) if True: perform bootstrap in the computation of q_k,
                               if False: compute q_k exactly
        :param batch_bootstrap: batch_size for the bootstrap computation of q_k (int)
        :param freq_eval: frequency for calling the evaluation function that is potentially costly (int)
        :param RAR: boolean enabling Renyi’s Adaptive Regularization (boolean)
        '''
        # Model
        self.lnprob = model_lnprob

        # Dimension
        self.D = D

        # Parameters
        self.N_tot = N_tot
        self.h_k0 = h_k
        self.h_k = h_k
        self.lambda_k = lambda_k
        self.eta_k = eta_k
        self.q0_mean = q0_mean
        self.q0_sd = q0_sd

        self.batch_size = batch_size
        self.batch_init = batch_init

        if self.D == 1:
            self._generate_from_multivariate = sample_from_k_theta_1d
        else:
            self._generate_from_multivariate = self._sample_from_k_theta_nd

        self.func_eval = func_eval
        self.bool_bootstrap = bool_bootstrap
        self.batch_bootstrap = batch_bootstrap
        self.freq_eval = freq_eval
        self.RAR=RAR

    def _sample_from_k_theta_nd(self, mean, sd, nb_samples):
        return np.random.multivariate_normal(mean=mean, cov=sd * np.identity(self.D), size=nb_samples)

    def _sample_from_muk(self, weights, means, sd, nb_samples_y, nb_centers):
        repartition = np.random.multinomial(nb_samples_y, weights)

        samples = []
        for i in range(nb_centers):
            nb = repartition[i]
            u = self._generate_from_multivariate(means[i], sd[i], nb)
            samples.extend(u)
        return np.array(samples)

    def _compute_muk(self, y, weights, means, sigmas, nb_centers):
        pdf_y = np.zeros(len(y))

        weights_eval = weights.copy()
        means_eval = means.copy()
        sigmas_eval = sigmas.copy()
        nb_centers_eval = nb_centers

        if self.bool_bootstrap:
            repartition = np.random.multinomial(self.batch_bootstrap, weights)
            samples = []
            sigmas_samples = []
            for i in range(nb_centers):
                nb = repartition[i]
                samples.extend([means[i]]*nb)
                sigmas_samples.extend([sigmas[i]])

            means_eval = samples
            sigmas_eval = sigmas_samples
            weights_eval = [1/self.batch_bootstrap]*self.batch_bootstrap
            nb_centers_eval = self.batch_bootstrap

        for i in range(nb_centers_eval):
            pdf_y += weights_eval[i] * multivariate_normal.pdf(y, mean=means_eval[i], cov=sigmas_eval[i] * np.identity(self.D))

        return pdf_y

    def generate_thetas(self, full_list_w, full_list_X, full_list_h, nb_centers, batch_size):

        new_thetas = self._sample_from_muk(full_list_w, full_list_X, full_list_h, batch_size, nb_centers)

        return np.reshape(new_thetas, (-1, self.D))

    def _full_algorithm(self):

        # sample according to q0
        k = 0
        X_1 = self._generate_from_multivariate(self.q0_mean, self.q0_sd, self.batch_init)

        # compute RIS weight
        q0_lnprob = multivariate_normal.logpdf(X_1, mean=self.q0_mean, cov=self.q0_sd * np.identity(self.D))

        if self.batch_init == 1:
            g_k = q0_lnprob - self.lnprob(X_1, 1)
            IS_w = [np.exp(-g_k)]
        else:
            g_k = q0_lnprob - self.lnprob(X_1, self.batch_init)
            IS_w = np.exp(-g_k)


        eta_val = self.eta_k(k, IS_w)
        eta_val_lst = []
        if self.RAR:
            eta_val_lst.append(eta_val)
        w_1 = np.exp(- eta_val * g_k)

        # add to list
        list_X = X_1
        if self.batch_init == 1:
            list_w = [w_1]
        else:
            list_w = w_1

        nb_X_tot = self.batch_init + 1
        full_list_X = np.vstack((list_X, self.q0_mean))
        val_lbd = self.lambda_k(k)

        full_list_w = (1 - val_lbd) / np.sum(list_w) * np.array(list_w)
        full_list_w = np.append(full_list_w, val_lbd)

        full_list_h = np.array(self.h_k(k, self.batch_init, nb_X_tot))
        full_list_h = np.append(full_list_h, self.q0_sd)

        evaluation_lst = []
        llh_lst = []

        evaluation, llh = self.func_eval(full_list_w, full_list_X, full_list_h, nb_X_tot)
        evaluation_lst.append(evaluation)
        llh_lst.append(llh)
        #print("------eval-------", evaluation)
        #print("------llh-------", llh)

        k = 1
        while k < self.N_tot:
            
            if np.mod(k,10) == 0:
                print('iter ',k)
            #print(k)

            # sampling according to qk
            X_k_plus_one = self.generate_thetas(full_list_w, full_list_X, full_list_h, nb_X_tot, self.batch_size)

            # compute RIS weight
            if self.batch_init == 1:
                g_k = np.log(self._compute_muk(X_k_plus_one, full_list_w, full_list_X, full_list_h, nb_X_tot))
                g_k += - self.lnprob(X_k_plus_one, 1)
            else:
                g_k = np.log(self._compute_muk(X_k_plus_one, full_list_w, full_list_X, full_list_h, nb_X_tot)) \
                      - self.lnprob(X_k_plus_one, self.batch_size)

            if self.batch_init == 1:
                IS_w = [np.exp(-g_k)]
            else:
                IS_w = np.exp(-g_k)

            eta_val = self.eta_k(k, IS_w)
            if self.RAR:
                eta_val_lst.append(eta_val)

            w_k_plus_one = np.exp(- eta_val * g_k)
            list_X = np.vstack((list_X, X_k_plus_one))

            if self.batch_init == 1:
                list_w.extend(w_k_plus_one)
            else:
                list_w = list(list_w) + list(w_k_plus_one)

            full_list_X = np.vstack((list_X, self.q0_mean))
            val_lbd = self.lambda_k(k)
            full_list_w = (1 - val_lbd) / np.sum(list_w) * np.array(list_w)
            full_list_w = np.append(full_list_w, val_lbd)

            nb_X_tot += self.batch_size
            full_list_h = np.array(self.h_k(k, self.batch_init, nb_X_tot))
            full_list_h = np.append(full_list_h, self.q0_sd)

            if k % self.freq_eval == 0:
                evaluation, llh = self.func_eval(full_list_w, full_list_X, full_list_h, nb_X_tot)
                #print("------eval-------", evaluation)
                #print("------llh-------", llh)
                evaluation_lst.append(evaluation)
                llh_lst.append(llh)

            k += 1



        return list_X, list_w, evaluation_lst, eta_val_lst, llh_lst


def sample_from_k_theta_1d(mean, sd, nb_samples):
    return np.random.normal(mean, sd, nb_samples)

'''
    Example of Bayesian Logistic Regression (the same setting as Gershman et al. 2012):
    The observed data D = {X, y} consist of N binary class labels, 
    y_t \in {-1,+1}, and d covariates for each datapoint, X_t \in R^d.
    The hidden variables \theta = {w, \alpha} consist of d regression coefficients w_k \in R,
    and a precision parameter \alpha \in R_+. We assume the following model:
        p(\alpha) = Gamma(\alpha; a, b)
        p(w_k | a) = N(w_k; 0, \alpha^-1)
        p(y_t = 1| x_t, w) = 1 / (1+exp(-w^T x_t))
'''


class BayesianLR:
    def __init__(self, X, Y, batchsize=100, a0=1, b0=0.01):
        self.X, self.Y = X, Y
        self.batchsize = min(batchsize, X.shape[0])
        self.a0, self.b0 = a0, b0

        self.N = X.shape[0]
        self.permutation = np.random.permutation(self.N)
        self.iter = 0

        self._generate_from_multivariate = self._sample_from_k_theta_nd

    def _sample_from_k_theta_nd(self, mean, sd, nb_samples, dim_latent):
        return np.random.multivariate_normal(mean=mean, cov=sd * np.identity(dim_latent), size=nb_samples)

    def lnprob(self, theta, batch_loop):
        my_array_unif = range(self.X.shape[0])
        if self.batchsize > 0:
            batch = [i % self.N for i in range(self.iter * self.batchsize, (self.iter + 1) * self.batchsize)]
            ridx = self.permutation[batch]
            self.iter += 1
        else:
            ridx = np.random.permutation(self.X.shape[0])

        Xs = self.X[ridx, :]
        Ys = self.Y[ridx]

        list_out = []
        for j in range(batch_loop):
            theta_j = theta[j]
            w = theta_j[:-1]  # logistic weights
            alpha = np.exp(theta_j[-1])  # the last column is logalpha
            d = len(w)
            coff = np.matmul(Xs, w.T)

            log_prior_alpha = gamma.logpdf(alpha, a=self.a0, scale=1/self.b0)
            log_prior_w = multivariate_normal.logpdf(w, mean=[0]*d, cov=1/alpha * np.identity(d))
            log_lik = np.log(1./(1. + np.exp(- Ys * coff)))

            list_out.append(log_prior_alpha + log_prior_w + np.sum(log_lik))

        return list_out

    def _compute_muk(self, y, weights, means, sigmas, nb_centers):
        pdf_y = np.zeros(len(y))

        weights_eval = weights.copy()
        means_eval = means.copy()
        sigmas_eval = sigmas.copy()
        nb_centers_eval = nb_centers

        if bool_bootstrap:
            repartition = np.random.multinomial(batch_bootstrap, weights)

            samples = []
            sigmas_samples = []
            for i in range(nb_centers):
                nb = repartition[i]
                samples.extend([means[i]]*nb)
                sigmas_samples.extend([sigmas[i]])

            means_eval = samples
            sigmas_eval = sigmas_samples
            weights_eval = [1/batch_bootstrap]*batch_bootstrap
            nb_centers_eval = batch_bootstrap

        for i in range(nb_centers_eval):
            pdf_y += weights_eval[i] * multivariate_normal.pdf(y, mean=means_eval[i], cov=sigmas_eval[i] * np.identity(dim_latent))

        return pdf_y

    def evaluation(self, weights, theta, h_t, J_t, X_test, y_test, nb_samples):
        n_test = len(y_test)
        prob = np.zeros(n_test)
        dim_latent = len(theta[0])

        repartition = np.random.multinomial(nb_samples, weights)
        full_y = []

        for j in range(J_t):
            nb = repartition[j]
            for _ in range(nb):
                y = self._generate_from_multivariate(theta[j], h_t[j], 1, dim_latent)[0]
                full_y.append(y)

                y = y[:-1]

                coff = np.multiply(y_test, np.sum(-1 * np.multiply(nm.repmat(y, n_test, 1), X_test), axis=1))
                prob += np.divide(np.ones(n_test), (1 + np.exp(coff)))

        muk_y = self._compute_muk(full_y, weights, theta, h_t, J_t)
        lnprob_val = self.lnprob(full_y, nb_samples)
        llh = np.sum(np.exp(lnprob_val) / muk_y)
        llh = np.log(llh / nb_samples)

        prob = 1/nb_samples * prob
        predictive = np.mean(prob > 0.5)
        return predictive, llh


def save_file(j, directory, strname, arraytosave):
    filename = directory + strname + str(j) + '.txt'
    np.savetxt(filename, arraytosave)


def joblib_function(j):
    func_eval = partial(model.evaluation, X_test=X_test, y_test=y_test, nb_samples=nb_eval)

    try:

        ris_algo = SRAIS(0, model.lnprob, func_eval, dim_latent, N_tot, h_k, eta_k, lambda_k, q0_mean, q0_sd,
                             batch_size, batch_init, bool_bootstrap, batch_bootstrap, freq_eval, RAR)
        thetas_ris, weights_ris, evaluation_lst_ris, eta_val_lst, llh_lst = ris_algo._full_algorithm()
        save_file(j, directory, 'f_ris_eval' + str_params, evaluation_lst_ris)
        save_file(j, directory, 'f_ris_llh' + str_params, llh_lst)

        if len(eta_val_lst) > 1:
            save_file(j, directory, 'f_ris_eta' + str_params, eta_val_lst)

    except:
        print('except')
        pass

    return 0

### Setting the functions to define (lambda_k)_{k \geq 1}, (h_k)_{k \geq 1} and (eta_k)_{k \geq 1} ###
def lambda_k(k):
    return 0.9/np.sqrt(k+1)


def h_k(k, batch_init, batch_size_tot):
    out = [np.power(k+1, -1/(4 + dim_latent))]* batch_init
    for j in range(2, k+2):
        out.extend([np.power(k+1, -1/(4 + dim_latent))]* batch_size_tot)
    return out


def eta_k_cte(k, weights, eta_0):
    return eta_0


def eta_k_RAR(k, weights, eta_0, alpha):
    length = len(weights)
    weights_normalised = weights.copy()/np.sum(weights)
    unif_w = [1/length]*length

    if alpha == 1.:
        entropy_val = entropy(weights_normalised, unif_w)/np.log(length)
    elif alpha == 0.:
        entropy_val = entropy(unif_w, weights_normalised)/np.log(length)
    else:
        entropy_val = np.log(np.sum(np.power(weights_normalised, alpha) * np.power(unif_w, 1-alpha))) / ((alpha - 1.)*np.log(length))

    out = min(1- entropy_val, 1.)
    if out < 0:
        out = eta_0

    return out

N = X_train.shape[0]
d = X_train.shape[1]
dim_latent = d +1
freq_eval = 10  # frequency for calling the evaluation function that is potentially costly

### Hyper-parameters of the BLR ####
a0, b0 = 1, 0.01  # hyper-parameters
model = BayesianLR(X_train, y_train, 100, a0, b0)  # batchsize = 100

### Parameters for RIS ###
N_tot = 300 # total number of iterations
RAR_list = [True, False]  # boolean for RAR
batch_init = 2000  # initial batch size for burn-in period
batch_size = 200  # batch size

bool_bootstrap = True  # boolean to allow for bootstrap
batch_bootstrap = 2000  # batch for the bootstrap

nb_eval = 500  # nb of samples used in the evaluation function

nb_cores_used = -1  # nb of cores, set > 1 for paralellisation
nb_repeat_exp = 100 # nb of trials

q0_sd = 5. # initial sampler sd
q0_mean = np.array([0.0] * dim_latent) # initial sampler mean

#### Launching the numerical experiments ####

str_params = 'Ntot_' + str(N_tot) + '_batch_init_' + str(batch_init) + '_batch_size_' + str(batch_size)
i_list = range(nb_repeat_exp)
for RAR in RAR_list:
    if not RAR:
        eta_0_list = [1.] # constant policy (SAIS)

    else:
        eta_0_list = [1 / (2 * np.sqrt(dim_latent))]
        alpha_list = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]

    for eta_0 in eta_0_list:
        if not RAR:
            directory = "./results/dim" + str(dim_latent) + "/eta" + str(eta_0) + '/'
            Path(directory).mkdir(parents=True, exist_ok=True)
            eta_k = partial(eta_k_cte, eta_0=eta_0)
            Parallel(nb_cores_used)(delayed(joblib_function)(i) for i in i_list)
            for i in i_list:
                if np.mod(i,10) == 0:
                  print('nb of repeat exp', i)
                joblib_function(i)

        else:
            for alpha in alpha_list:
                print('current alpha ', alpha)
                directory = "./results/dim" + str(dim_latent) + "/eta" + str(eta_0) + '/alpha' + str(alpha) + '/'
                Path(directory).mkdir(parents=True, exist_ok=True)
                eta_k = partial(eta_k_RAR, eta_0=eta_0, alpha=alpha)
                Parallel(nb_cores_used)(delayed(joblib_function)(i) for i in i_list)
                for i in i_list:
                    joblib_function(i)