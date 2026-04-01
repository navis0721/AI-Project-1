import numpy as np
from scipy.special import softmax
import torch

# parameters
N = 20000   # number of patients
T = 1 # number of timestamps
k = 25  # number of covariates
k_t = 20 # number of temporal covariates
k_hidden = 5      # number of hidden confounders
alpha_A = 2.0 # hidden_treat_strength
alpha_Y = 2.0  # hidden_outcome_strength
noise_std = 2.0 # outcome noise
treatment_bias = 0.8

np.random.seed(66)
torch.manual_seed(66)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def gen_next(X, A ,t):

    lam = np.random.normal(0, 0.2, size=(N, 1))

    weights = np.arange(1, t+1)

    weights = softmax(weights)

    x_t = np.average(X[:, :, :t], axis=-1, weights=weights) +lam* np.average(A[:, :, :t], axis=-1, weights=weights)
    return x_t


def gen_factual_data():

    # initial state
    mu = np.zeros(shape=(k))
    cov = np.random.uniform(low=-1, high=1, size=(k,k))
    x_cov = 0.1*(np.dot(cov,cov.T))

    x = np.random.multivariate_normal(mean=mu, cov=x_cov, size=(N,))
    x_t = x[:, :k_t]
    x_sta = x[:, k_t:]
    
    # hidden confounders
    U = np.random.normal(loc=0.0, scale=1.0, size=(N, k_hidden))
    # coefficients for hidden confounders
    beta_u_treat = np.random.normal(0, 1, size=(k_hidden, 1))
    beta_u_outcome = np.random.normal(0, 1, size=(k_hidden, 1))

    m = np.random.normal(0, 0.1, (N,))
    s = np.random.multivariate_normal(mean=np.zeros(shape=(k)), cov=0.1 * np.identity(k), size=(1,)).T

    # treatment probability
    prob_obs = np.matmul(x, s).squeeze(-1) + m
    prob_hidden = alpha_A * np.dot(U, beta_u_treat).squeeze(-1)
    prob = sigmoid(prob_obs + prob_hidden + treatment_bias)
    a = np.random.binomial(n=1, p=prob, size=(N,)).reshape((N,1))

    X = np.zeros(shape=(N, k_t, T))
    A = np.zeros(shape=(N, 1, T))
    Y = np.zeros(shape=(N, 2, T))   # factual and counterfactual
    
    X[:,:,0] = x_t
    A[:,:,0] = a

    # simulate y
    cov = np.random.uniform(low=-1, high=1, size=(k, k))
    w_cov = 0.1 * (np.dot(cov, cov.T))
    mu1 = np.zeros(shape=(k))
    mu0 = np.ones(shape=(k))
    w1 = np.random.multivariate_normal(mean=mu1, cov=w_cov, size=(1,)).T
    w0 = np.random.multivariate_normal(mean=mu0, cov=w_cov, size=(1,)).T

    tmp = np.concatenate((X[:, :, 0], x_sta), axis=1)
    Y_hidden = alpha_Y * (np.dot(U, beta_u_outcome))
    Y1 = np.dot(tmp, w1) + Y_hidden
    Y0 = np.dot(tmp, w0) + Y_hidden
    eps = np.random.normal(0, noise_std, size=(N, 1) )   # noise
                           
    y_f = a * Y1 + (1 - a) * Y0 + eps
    y_cf = a * Y0 + (1 - a) * Y1 + eps

    Y[:, 0, 0] = y_f.squeeze()
    Y[:, 1, 0] = y_cf.squeeze()

    for t in range(1, T):

        # generate X
        x_t = gen_next(X, A, t)
        X[:,:,t] = x_t

        # generate A
        prob_obs = np.matmul(x, s).squeeze(-1) + m
        prob_hidden = alpha_A * np.dot(U, beta_u_treat).squeeze(-1)
        prob = sigmoid(prob_obs + prob_hidden + treatment_bias)

        a = np.random.binomial(n=1, p=prob, size=(N,)).reshape((N, 1))
        A[:, :, t] = a

        # generate Y
        tmp = np.concatenate((x_t,x_sta),axis=1)
        Y_hidden = alpha_Y * np.dot(U, beta_u_outcome)
        Y1 = np.dot(tmp, w1) + Y_hidden
        Y0 = np.dot(tmp, w0) + Y_hidden
        
        y_f = a * Y1 + (1 - a) * Y0 + eps
        y_cf = a * Y0 + (1 - a) * Y1 + eps

        Y[:, 0, t] = y_f.squeeze()
        Y[:, 1, t] = y_cf.squeeze()

    # normalize
    print("Y mean: ", Y.mean()) # -0.08771662864893053
    print("Y std: ", Y.std())   # 8.946346049255661
    Y = (Y - np.mean(Y)) / np.std(Y)

    x_mean = np.mean(X, axis=(0, 2))
    x_std = np.std(X, axis=(0, 2))
    for i in range(k_t):
        X[:,i,:] = (X[:,i,:]-x_mean[i])/x_std[i]

    x_static_mean = np.mean(x_sta, axis=0)
    x_static_std = np.std(x_sta, axis=0)
    for i in range(k-k_t):
        x_sta[:,i] = (x_sta[:,i]-x_static_mean[i])/x_static_std[i]

    print(X.shape, A.shape, Y.shape, x_sta.shape)  #(5000, 20, 50) (5000, 1, 50) (5000, 2, 50) (5000, 5)
    X = X.transpose(0, 2, 1)   # [N, T, k_t]
    A = A.transpose(0, 2, 1)   # [N, T, 1]
    Y = Y.transpose(0, 2, 1)   # [N, T, 2]
    
    results = {'X': X, 'A_factual': A, 'Y': Y, 'X_sta': x_sta}

    return results


if __name__ == '__main__':

    data = gen_factual_data()

    X, A, Y, X_sta = data['X'], data['A_factual'], data['Y'], data['X_sta']
    
    extract_N = 5000

    statics = np.array(X_sta[:extract_N])
    feature = np.array(X[:extract_N])
    y = np.array(Y[:extract_N])
    treatment = np.array(A[:extract_N])
    print("after extraction: ", statics.shape, feature.shape, y.shape, treatment.shape)
    
    data = {
        'statics': torch.FloatTensor(statics),
        'feature': torch.FloatTensor(feature),
        'y': torch.FloatTensor(y),
        'treatment': torch.FloatTensor(treatment),
    }

    torch.save(data, 'datasets/syn_dataset_hidden2_noise2_5k.pt')
    print("Synthetic data generated and saved.")



