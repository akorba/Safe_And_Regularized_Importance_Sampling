import numpy as np
from scipy.stats import multivariate_normal


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
        print("------eval-------", evaluation)
        print("------llh-------", llh)

        k = 1
        while k < self.N_tot:
            print(k)

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
                print("------eval-------", evaluation)
                print("------llh-------", llh)
                evaluation_lst.append(evaluation)
                llh_lst.append(llh)

            k += 1



        return list_X, list_w, evaluation_lst, eta_val_lst, llh_lst
