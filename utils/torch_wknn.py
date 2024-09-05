# -*- coding: utf-8 -*-

"""

P. Bugata, P. Drotár, Weighted nearest neighbors feature selection,
Knowledge-Based Systems (2018), doi:https://doi.org/10.1016/j.knosys.2018.10.004

"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import OneHotEncoder    # needed for multi-class classification
from sklearn.base import TransformerMixin
import time

from torch import optim


class WkNNFeatureSelector(TransformerMixin):

    def __init__(self, max_features, n_iters=1000, n_iters_in_loop=100,
                 metric='euclidean', p=None, kernel='rbf',
                 error_type='mse', delta=1.0,
                 lambda0=0.001, lambda1=0.001, lambda2=0.001, alpha=100,
                 optimizer='SGD', learning_rate=0.1,
                 normalize_gradient=True, data_type='float32', scaling=True,
                 apply_weights=False, n_iters_weights=300, verbose=False
                 ):
        # this class implements TransformerMixin interface
        TransformerMixin.__init__(self)

        # how many features to select
        self.max_features_ = max_features
        # number of epochs
        self.n_iters_ = n_iters
        # number of iterations to display
        self.n_iters_in_loop_ = n_iters_in_loop
        # distance metric
        self.metric_ = metric
        # p norm used in minkowski distance
        self.p_ = p
        # kernel - distance evaluation function
        self.kernel_ = kernel
        # error function
        self.error_type_ = error_type
        # delta - used only for Huber Loss function
        self.delta_ = delta
        # regularization parameter for pseudo L0 regularization
        self.lambda0_ = lambda0
        # regularization parameter for L1 regularization
        self.lambda1_ = lambda1
        # regularization parameter for L2 regularization
        self.lambda2_ = lambda2
        # alpha - used only for pseudo L0 regularization
        self.alpha_ = alpha
        # optimizer type
        self.optimizer_ = optimizer
        # learning rate
        self.learning_rate_ = learning_rate
        # gradient normalization flag
        self.normalize_gradient_ = normalize_gradient
        # data type for precision and numerical stability
        self.data_type_ = data_type
        # standardization of input data
        self.scl_ = None
        if scaling:
            self.scl_ = StandardScaler()

        # selected features
        self.selected_features_ = None
        # selected features after fine-tuning weights
        self.final_selected_features_ = None

        # feature weights
        self.weights_ = None
        # feature weights after fine-tuning weights
        self.final_weights_ = None

        # error
        self.error_ = None
        # error after fine-tuning weights
        self.final_error_ = None

        # checking parameters
        # unsupported
        if metric not in ['euclidean', 'cityblock', 'minkowski']:
            raise ValueError('Unsupported metric')
        if kernel not in ['rbf', 'exp']:
            raise ValueError('Unsupported kernel')
        if error_type not in ['mse', 'mae', 'huber', 'ce']:
            raise ValueError('Unsupported error type')
        if optimizer not in ['SGD', 'Adam', 'Nadam', 'Adagrad', 'Adadelta', 'RMSProp', 'Momentum']:
            raise ValueError('Unsupported optimizer')
        if data_type not in ['float32', 'float64']:
            raise ValueError('Unsupported data type')
        # unallowed
        if error_type == 'huber' and delta is None:
            raise ValueError('Parameter delta for Huber function is missing.')
        if alpha is None:
            raise ValueError('Parameter alpha for L0 regularization is missing.')
        if optimizer != 'SGD' and self.normalize_gradient_:
            raise ValueError('Gradient normalization is alloved only for SGD.')

        # Constant for numerical stability
        if self.data_type_ == torch.float32:
            self.epsilon_ = torch.tensor(1e-14, dtype=torch.float32)
        elif self.data_type_ == torch.float64:
            self.epsilon_ = torch.tensor(1e-300, dtype=torch.float64)

        # classification task flag for multi-class classification
        self.classification = None

        # flag to decide whether apply feature weights when transforming data
        self.apply_weights = apply_weights

        # number of epochs to fine-tuning weights
        self.n_iters_weights_ = n_iters_weights

        # debug prints
        self.verbose = verbose

    # pairwise sqeuclidean distance
    def sqeuclidean_dist(self, A, B=None):
        # Squared Euclidean distance with optional pairwise distance computation
        if B is not None:
            norm_a = torch.sum((A ** 2) * self.weights_, dim=1)
            norm_b = torch.sum((B ** 2) * self.weights_, dim=1)
            norm_a = norm_a.view(-1, 1)
            norm_b = norm_b.view(1, -1)
            scalar_product = torch.mm(A * self.weights_, B.t())
        else:
            norm_a = torch.sum((A ** 2) * self.weights_, dim=1)
            norm_b = norm_a
            norm_a = norm_a.view(-1, 1)
            norm_b = norm_b.view(1, -1)
            scalar_product = torch.mm(A * self.weights_, A.t())

        D = norm_a - 2 * scalar_product + norm_b

        # To avoid computational problems when computing derivatives, replace small values with epsilon
        D = torch.where(D > self.epsilon_, D, torch.ones_like(D) * self.epsilon_)

        return D



    # minkowski distance using expand_dim (slower then map_fn)
    def minkowski_dist_map(self, A, B=None):
        # Minkowski distance map
        if B is None:
            B = A

        def dist_(x):
            # Compute the element-wise absolute difference
            result = torch.abs(x - B)

            # Apply the p-norm if p is not 1
            if self.p_ != 1:
                result = torch.pow(result, self.p_)

            # Multiply by weights and sum along the second dimension
            result = torch.sum(result * self.weights_, dim=1)

            if self.p_ != 1:
                # To avoid computational problems, use small value instead of 0
                result = torch.where(result > self.epsilon_, result, torch.ones_like(result) * self.epsilon_)

                # Take the p-th root
                result = torch.pow(result, 1 / self.p_)

            return result.t()

        # Use torch.vmap or a loop since PyTorch lacks an equivalent to tf.map_fn
        D = torch.stack([dist_(a) for a in A])

        return D

    # using of nested map fn - slow!
    def manhattan_dist(self, A, B=None):
        # Manhattan (L1) distance function
        if B is None:
            B = A

        def dist_(x):
            # Function to compute the weighted Manhattan distance between a row of A and B
            def dist_row_(y):
                result = torch.abs(x - y)
                result = torch.sum(result * self.weights_, dim=1)
                return result

            # Apply dist_row_ to each row in B
            return torch.stack([dist_row_(b) for b in B])

        # Apply dist_ to each row in A
        D = torch.stack([dist_(a) for a in A])

        return D

    # computing similarity matrix using corresponding metric and kernel
    def similarity_matrix(self, A, B=None):
        similarity_mat = None

        if (self.metric_ == 'euclidean' and self.kernel_ == 'rbf'):
            dists = self.sqeuclidean_dist(A, B)
            similarity_mat = torch.exp(-dists)

        if (self.metric_ == 'euclidean' and self.kernel_ == 'exp'):
            dists = self.sqeuclidean_dist(A, B)
            dists_sqrt = torch.sqrt(dists)
            similarity_mat = torch.exp(-dists_sqrt)

        if self.metric_ == 'cityblock':
            self.p_ = 1
            dists = self.minkowski_dist_map(A, B)
            if self.kernel_ == 'exp':
                similarity_mat = torch.exp(-dists)
            elif self.kernel_ == 'rbf':
                similarity_mat = torch.exp(-torch.square(dists))

        if self.metric_ == 'minkowski':
            if self.p_ is None:
                raise ValueError('Parameter p is missing')
            dists = self.minkowski_dist_map(A, B)
            if self.kernel_ == 'exp':
                similarity_mat = torch.exp(-dists)
            elif self.kernel_ == 'rbf':
                similarity_mat = torch.exp(-torch.square(dists))

        if similarity_mat is None:
            raise ValueError('Unsupported metric/kernel combination.')

        return similarity_mat

    # compute prediction vector
    # prediction for i-th dataset point is weighted average of target values of other points
    def predict(self, similarity_matrix, y):
        # Create a zero matrix of the same shape as the similarity matrix
        zero_mat = torch.zeros_like(similarity_matrix)

        # Create a diagonal matrix with ones along the diagonal
        diag_mat = torch.eye(y.shape[0], dtype=y.dtype, device=y.device)

        # Create a mask where the diagonal is greater than zero
        diag_mask = diag_mat > zero_mat

        # Modify the similarity matrix by replacing the diagonal with zeros
        mod_sim_matrix = torch.where(diag_mask, zero_mat, similarity_matrix)

        # Sum the modified similarity matrix along the rows
        weight_sums = torch.sum(mod_sim_matrix, dim=1, keepdim=True)

        # Ensure that the weight sums are greater than epsilon to avoid division errors
        assert torch.all(weight_sums >= self.epsilon_), f"Minimum weight sum: {torch.min(weight_sums)}"

        # Compute the predictions by multiplying the modified similarity matrix with the labels (y)
        # and dividing by the sum of weights
        predictions = torch.matmul(mod_sim_matrix, y) / weight_sums

        return predictions

    def compute_error(self, y, y_pred):
        if self.error_type_ == 'mse':
            # Mean Squared Error
            losses = torch.square(y - y_pred)
        elif self.error_type_ == 'mae':
            # Mean Absolute Error
            losses = torch.abs(y - y_pred)
        elif self.error_type_ == 'huber':
            # Huber loss
            errors = torch.abs(y - y_pred)
            hlf_1 = 0.5 * torch.square(y - y_pred)
            hlf_2 = self.delta_ * errors - 0.5 * self.delta_ ** 2
            losses = torch.where(errors <= self.delta_, hlf_1, hlf_2)
        elif self.error_type_ == 'ce':
            # Cross Entropy loss
            err_clipped = torch.clamp(1 - torch.abs(y - y_pred), min=self.epsilon_)
            if self.classification:
                losses = -y * torch.log(err_clipped)
            else:
                losses = -torch.log(err_clipped)

        # Sum over classes if y is one-hot encoded (classification)
        if self.classification:
            losses = torch.sum(losses, dim=1)

        # Return the mean loss
        return torch.mean(losses)

    # pseudo L0 regularization
    # sum of sigmoid functions indicating non-zero variable weight
    # Pseudo L0 Regularization
    def pseudo_l0_regularization(self):
        reg = self.weights_
        reg = torch.exp(reg * (-self.alpha_))
        reg = reg + torch.ones_like(reg)
        reg = torch.ones_like(reg) / reg
        reg = reg + torch.ones_like(reg) * -0.5
        reg = torch.sum(reg) * 2
        return reg

    # L1 Regularization
    def l1_regularization(self):
        # Penalize the sum of absolute values of weights
        return torch.sum(self.weights_)

    # L2 Regularization
    def l2_regularization(self):
        # Penalize the sum of squares of weights
        return torch.sum(torch.square(self.weights_))

    # Normalize gradient to max decrease by 1 * learning_rate
    def normalize_grad(self, grad, val):
        # Gradient L1 norm
        gradient_norm = torch.sum(torch.abs(grad))

        # Correction for negative weights
        new_val = val - grad * self.learning_rate_
        neg_sum = torch.sum(torch.minimum(new_val, torch.tensor(0.0, device=new_val.device, dtype=new_val.dtype)))
        gradient_norm += neg_sum / self.learning_rate_

        # Ensure max increase
        gradient_norm = torch.maximum(gradient_norm, torch.tensor(0.01, device=grad.device, dtype=grad.dtype))

        # Normalize gradient
        normalized_grad = grad / gradient_norm
        return normalized_grad

    # gradient clipping
    def modified_minimize(self, optimizer, cost_function):
        # Compute gradients with respect to the weights
        optimizer.zero_grad()  # Clear any existing gradients
        cost = cost_function()  # Compute the loss
        cost.backward()  # Compute gradients

        # Retrieve gradients for weights
        gvs = [(param.grad, param) for param in [self.weights_] if param.grad is not None]

        # Normalize or clip gradients
        if self.normalize_gradient_:
            capped_gvs = [(self.normalize_grad(grad, param), param) for grad, param in gvs]
        else:
            capped_gvs = [(torch.nn.utils.clip_grad_norm_(param, 1.0), param) for grad, param in gvs]

        # Apply the gradients to update the weights
        optimizer.step()

        # Clip negative weights by setting them to 0 after the update
        with torch.no_grad():
            self.weights_.data = torch.clamp(self.weights_.data, min=0.0)

        return self.weights_

    # computing cost function
    def compute_cost(self, X, y):

        # applying distance evaluation function
        similarity_matrix_op = self.similarity_matrix(X, None)

        # predict target values
        y_pred = self.predict(similarity_matrix_op, y)

        # compute error
        mean_error = self.compute_error(y, y_pred)

        return mean_error

    # optimizer loop in computation graph
    def optimizer_loop(self, optimizer, X, y, n_iter):
        # Variables to track error and regularization terms
        var_err = torch.tensor(0.0, dtype=y.dtype, device=y.device)
        var_reg0 = torch.tensor(0.0, dtype=y.dtype, device=y.device)
        var_reg1 = torch.tensor(0.0, dtype=y.dtype, device=y.device)
        var_reg2 = torch.tensor(0.0, dtype=y.dtype, device=y.device)

        # Loop over the number of iterations
        for i in range(n_iter):
            # Compute cost (mean error)
            mean_error = self.compute_cost(X, y)
            var_err = mean_error

            # Compute regularization terms
            if self.lambda0_ != 0 and self.lambda1_ != 0 and self.lambda2_ != 0:
                reg0 = self.pseudo_l0_regularization()
                reg1 = self.l1_regularization()
                reg2 = self.l2_regularization()
                var_reg0 = reg0
                var_reg1 = reg1
                var_reg2 = reg2
                cost = mean_error + reg0 * self.lambda0_ + reg1 * self.lambda1_ + reg2 * self.lambda2_
            elif self.lambda0_ != 0 and self.lambda1_ != 0:
                reg0 = self.pseudo_l0_regularization()
                reg1 = self.l1_regularization()
                var_reg0 = reg0
                var_reg1 = reg1
                var_reg2 = 0.0
                cost = mean_error + reg0 * self.lambda0_ + reg1 * self.lambda1_
            elif self.lambda0_ != 0 and self.lambda2_ != 0:
                reg0 = self.pseudo_l0_regularization()
                reg2 = self.l2_regularization()
                var_reg0 = reg0
                var_reg1 = 0.0
                var_reg2 = reg2
                cost = mean_error + reg0 * self.lambda0_ + reg2 * self.lambda2_
            elif self.lambda1_ != 0 and self.lambda2_ != 0:
                reg1 = self.l1_regularization()
                reg2 = self.l2_regularization()
                var_reg0 = 0.0
                var_reg1 = reg1
                var_reg2 = reg2
                cost = mean_error + reg1 * self.lambda1_ + reg2 * self.lambda2_
            elif self.lambda0_ != 0:
                reg0 = self.pseudo_l0_regularization()
                var_reg0 = reg0
                var_reg1 = 0.0
                var_reg2 = 0.0
                cost = mean_error + reg0 * self.lambda0_
            elif self.lambda1_ != 0:
                reg1 = self.l1_regularization()
                var_reg0 = 0.0
                var_reg1 = reg1
                var_reg2 = 0.0
                cost = mean_error + reg1 * self.lambda1_
            elif self.lambda2_ != 0:
                reg2 = self.l2_regularization()
                var_reg0 = 0.0
                var_reg1 = 0.0
                var_reg2 = reg2
                cost = mean_error + reg2 * self.lambda2_
            else:
                cost = mean_error
                var_reg0 = 0.0
                var_reg1 = 0.0
                var_reg2 = 0.0

            # Minimize the cost using the modified optimizer step
            self.modified_minimize(optimizer, lambda: cost)

            # Manually clamp the weights to avoid negative values
            with torch.no_grad():
                self.weights_.data = torch.clamp(self.weights_.data, min=0.0)

        # Return the last error and regularization values
        return var_err, var_reg0, var_reg1, var_reg2

    # creating optimizer
    def create_optimizer(self):
        if self.optimizer_ == 'SGD':
            return optim.SGD([self.weights_], lr=self.learning_rate_)
        elif self.optimizer_ == 'Adam':
            return optim.Adam([self.weights_], lr=self.learning_rate_)
        elif self.optimizer_ == 'Nadam':
            return optim.NAdam([self.weights_], lr=self.learning_rate_)
        elif self.optimizer_ == 'Adagrad':
            return optim.Adagrad([self.weights_], lr=self.learning_rate_)
        elif self.optimizer_ == 'Adadelta':
            return optim.Adadelta([self.weights_], lr=self.learning_rate_)
        elif self.optimizer_ == 'RMSProp':
            return optim.RMSprop([self.weights_], lr=self.learning_rate_)
        elif self.optimizer_ == 'Momentum':
            return optim.SGD([self.weights_], lr=self.learning_rate_, momentum=0.9, nesterov=True)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_}")

    # building model - select variables and determining their weights
    def fit(self, X, y, init_weights=None):
        # Data standardization
        if self.scl_ is not None:
            X_scl = self.scl_.fit_transform(X)
        else:
            X_scl = X

        # Conversion of y to numpy
        y = np.array(y)

        # If y is 1D, reshape to column
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Classification flag is automatically set according to the number of y columns
        self.classification = y.shape[1] > 1

        # Convert X and y to PyTorch tensors
        X_var = torch.tensor(X_scl, dtype=torch.float32, requires_grad=False)
        y_var = torch.tensor(y, dtype=torch.float32, requires_grad=False)

        m = X_var.shape[1]
        if init_weights is None:
            self.weights_ = torch.ones(1, m, dtype=torch.float32, requires_grad=True) / m
        else:
            self.weights_ = torch.tensor(init_weights, dtype=torch.float32, requires_grad=True)

        # Create the optimizer
        optimizer = self.create_optimizer()

        # Optimizer loop
        for e in range(self.n_iters_):
            optimizer.zero_grad()

            # Compute the cost
            mean_error = self.compute_cost(X_var, y_var)

            # Add regularization
            if self.lambda0_ != 0:
                reg0 = self.pseudo_l0_regularization()
            else:
                reg0 = torch.tensor(0.0)

            if self.lambda1_ != 0:
                reg1 = self.l1_regularization()
            else:
                reg1 = torch.tensor(0.0)

            if self.lambda2_ != 0:
                reg2 = self.l2_regularization()
            else:
                reg2 = torch.tensor(0.0)

            cost = mean_error + reg0 * self.lambda0_ + reg1 * self.lambda1_ + reg2 * self.lambda2_

            # Backward pass and optimization step
            cost.backward()
            optimizer.step()

            # Clamp weights to be non-negative
            with torch.no_grad():
                self.weights_.data = torch.clamp(self.weights_.data, min=0.0)

            # Verbose output
            if self.verbose and e % 100 == 0:
                print(
                    f'Epoch {e}, Error: {cost.item()}, L0 reg: {reg0.item()}, L1 reg: {reg1.item()}, L2 reg: {reg2.item()}')

            # Early stopping if cost is NaN
            if torch.isnan(cost):
                break

        # Final weights
        self.weights_ = self.weights_.detach().numpy().flatten()

        # Select non-zero features
        self.selected_features_ = np.argsort(self.weights_)[::-1]
        nonzero_count = np.sum(self.weights_ > 0)

        if self.verbose:
            print('Non-zero weights: ', nonzero_count)
            print('Big weights: ', np.sum(self.weights_ > 0.001))
            print('Weight sum: ', np.sum(self.weights_))

        self.selected_features_ = self.selected_features_[:min(self.max_features_, nonzero_count)]

        # Fine-tune weights
        if not self.apply_weights:
            return self

        weights = self.weights_
        selected_features = self.selected_features_
        n_iters = self.n_iters_
        scl = self.scl_
        error = self.error_

        # Fine-tuning loop
        self.n_iters_ = self.n_iters_weights_
        self.scl_ = None
        self.apply_weights = False
        X_transformed = X_scl[:, selected_features]

        if self.verbose:
            print('Selected features: ', selected_features.tolist())
            print('Selected weights: ', weights[selected_features].tolist())
            print('Selected weights sum: ', weights[selected_features].sum())

        self.fit(X_transformed, y, weights[selected_features])
        self.final_selected_features_ = selected_features[self.selected_features_]
        self.final_weights_ = np.zeros(m, dtype=weights.dtype)
        self.final_weights_[self.final_selected_features_] = self.weights_[self.selected_features_]
        self.final_error_ = self.error_

        if self.verbose:
            print('Final selected features: ', self.final_selected_features_.tolist())
            print('Final selected weights: ', self.final_weights_[self.final_selected_features_].tolist())

        # Restore original state
        self.weights_ = weights
        self.selected_features_ = selected_features
        self.scl_ = scl
        self.n_iters_ = n_iters
        self.apply_weights = True
        self.error_ = error

        return self

    # transforming data by variable selection and/or applying weights
    def transform(self, X):
        # data standardization
        if self.scl_ is not None:
            X_transformed = self.scl_.transform(X)
        else:
            X_transformed = X

        # applying variable weights to transformed data
        if self.apply_weights:
            X_transformed = X_transformed[:, self.final_selected_features_]
            # correct application of weights according to metric
            weights_to_use = self.final_weights_[self.final_selected_features_]
            if self.metric_.count('minkowski') > 0 and self.p_ > 1:
                weights_to_use = weights_to_use ** (1 / self.p_)
            elif self.metric_.count('euclidean') > 0:
                weights_to_use = np.sqrt(weights_to_use)
            X_transformed = np.multiply(X_transformed, weights_to_use)
        else:
            X_transformed = X_transformed[:, self.selected_features_]

        return X_transformed


if __name__ == '__main__':
    # because of reproducible results
    #tf.set_random_seed(1)

    # path to data
    path = 'D:/Users/pBugata/data/madelon'
    dataset_name = 'madelonHD'
    delim = None

    # loading data
    X = np.loadtxt(path + '/' + dataset_name + '_X.txt', delimiter=delim)
    y = np.loadtxt(path + '/' + dataset_name + '_y.txt', delimiter=delim)

    # options for initial weights
    m = len(X[0])
    v_m = np.ones([1, m]) * 1 / m
    v_zero = np.zeros([1, m])

    # necessary to uncomment for multi-class classification
    # y = OneHotEncoder(sparse=False).fit_transform(y)

    # set parameters of transformer

    # max_features - number of features to select
    # n_iters - number of iterations (epochs)
    # n_iters_in_loop - number of iterations to display progress
    # metric - definition of distance (euclidean, cityblock, minkowski)
    # p - parameter for Minkowski distance
    # kernel - distance evaluation function (rbf, exp)
    # error_type - error function (mse, mae, huber, ce)
    # delta - parameter of Huber loss function
    # lambda0, lambda1, lambda2 - regularization parameter for pseudo L0, L1, and L2 regularization
    # alpha - parameter for pseudo L0 regularization (steepness of sigmoid function)
    # optimizer - optimizer type (SGD, Adam, Nadam, Adagrad, Adadelta, RMSProp, Momentum)
    # learning_rate - parameter for gradient descent
    # normalize_gradient - flag for L1 normalization of gradient
    # data_type - data type for controlling precision
    # scaling - flag for standardization of input data
    # apply_weights - flag for using weights for transformation of entire dataset
    # n_iters_weights - number of iterations for fine-tuning weights
    # verbose - boolean flag indicating whether print messages

    transformer = WkNNFeatureSelector(
        max_features=30, n_iters=10000, n_iters_in_loop=1000,
        metric='euclidean', p=2, kernel='exp',
        error_type='ce', delta=0.1,
        lambda0=0.00, lambda1=0.00, lambda2=0.00, alpha=100,
        optimizer='SGD', learning_rate=0.1,
        normalize_gradient=True, data_type='float64', scaling=True,
        apply_weights=False, n_iters_weights=1, verbose=False)

    t_start = time.time()

    # X, y - input data
    # init_weights - set initial weights
    transformer.fit(X, y, init_weights=v_m)

    col_indices = transformer.selected_features_
    feature_weights = transformer.weights_[col_indices]
    print('Variables: ', col_indices.tolist())
    print('Variable weights: ', feature_weights.tolist())

    t_end = time.time()
    print("Duration:", t_end - t_start)

    # creating and saving transformed data for future use (not necessary)
    X_transformed = transformer.transform(X)
    np.savetxt(X=X_transformed, fname=path + '/' + dataset_name + '_transformed_X.txt', delimiter=' ')
