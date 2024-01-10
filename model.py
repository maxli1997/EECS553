import sys
import getopt
import torch
import numpy as np
from sklearn.model_selection import KFold
import time


class Sparse:
    def __init__(self, n_rows, n_columns, per_column):
        rng = np.random.default_rng()
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.rows = []
        self.cols = []
        for col in range(n_columns):
            for i in range(per_column):
                self.cols.append(col)
            rows = np.random.choice(n_rows, size=per_column, replace=False)
            for val in rows:
                self.rows.append(val)
        self.vals = torch.tensor(rng.choice([1., -1.], n_columns*per_column), requires_grad=True, device=device)

    def get_matrix(self):
        S = torch.zeros((self.n_rows, self.n_columns), device=device)
        for i, idx in enumerate(zip(self.rows, self.cols)):
            S[idx] += self.vals[i]
        return S


# power method to calculate the leading singular vector/value
def PM(A, T=100, eps=1e-6):
    N = A.shape[1]
    v1 = torch.ones(N, device=device)/np.sqrt(N)
    B = A.T @ A
    for itr in range(T):
        v_prev = v1
        Bv1 = torch.matmul(B, v1)
        v1 = Bv1/torch.linalg.norm(Bv1)
        if torch.linalg.norm(v1-v_prev)/torch.linalg.norm(v_prev) < eps:
            break
    Av1 = torch.matmul(A, v1)
    e1 = torch.linalg.norm(Av1)
    u1 = Av1/e1
    return u1, v1, e1


# calculate svd using power method
def diff_SVD(A, m):
    U = []
    Vt = []
    E = []
    for j in range(m):
        u, v, e = PM(A)
        U.append(u)
        Vt.append(v)
        E.append(e)
        A = A - e*torch.outer(u, v)
    # seems we need to transpose to get the correct U
    U = torch.stack(U).T
    Vt = (torch.stack(Vt))
    E = torch.diag(torch.stack(E))
    return U, E, Vt


# main algorithm
def clarkson_woodruff(S, A, k):
    SA = torch.matmul(S, A)
    m = S.shape[0]
    U, E, Vt = diff_SVD(SA, m)
    V = Vt.T
    U2, E2, Vt2 = diff_SVD(torch.matmul(A, V),k)
    return U2@E2@Vt2@Vt


# minimizing emperical risk on given batch of training samples
def train_step(S, k, x_batch, batch_size, lr=1e-3, ld=10):
    optimizer = torch.optim.SGD([S.vals], lr=lr)
    optimizer.zero_grad(set_to_none=True)
    loss = 0
    mat = S.get_matrix()
    for j in range(batch_size):
        A = torch.tensor(x_batch[:, :, j],dtype=torch.float,device=device)
        A.requires_grad = True
        CW = clarkson_woodruff(mat, A, k)
        loss += torch.norm(A - CW, 'fro')
    # l2-regularization
    # loss += ld*torch.sum(torch.absolute(S.vals))
    loss.backward()
    optimizer.step()
    return S.vals


def test(S, k, x_test, ld):
    loss = 0
    for i in range(x_test.shape[0]):
        A = x_test[i]
        CW = clarkson_woodruff(S, A, k)
        loss += torch.norm(A - CW, 'fro') + ld*torch.norm(S, 'fro')
    return loss


def main(m=10, k=10, nnz=1, dataset="mnist"):
    outfile = "m{}-k{}-z{}-{}-sketch".format(m, k, nnz, dataset)
    st = time.process_time()

    print(torch.cuda.is_available())
    torch.cuda.empty_cache()
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)

    learning_rate = 1e-2
    batch_size = 50
    train_size = 400
    ld = 1
    #epoch = 200
    #For mnist using epoch = 10
    epoch = 10

    full_data = np.load("Evaluation/Dataset/data.npz")
    data = full_data[dataset][:, :, 0:train_size]

    n = data[:, :, 0].shape[0]
    N = data.shape[2]
    S = Sparse(m, n, nnz)
    # S = torch.randint(high=2, size=(m, n)).float()*2 - 1
    # S = S.to(device)
    # U,E,Vt = diff_SVD(S,S.shape[0])
    # S = U@E@Vt
    # S.requires_grad=True

    for i in range(epoch):
        perm = np.random.permutation(N)
        start, stop = 0, batch_size
        print(f'Starting epoch {i+1} / {epoch}')
        for j in range(int(N/batch_size)):
            print(f'Starting batch {j+1} / {int(N/batch_size)}')
            idx = perm[start:stop]
            start += batch_size
            stop += batch_size
            if stop > N:
                stop = N
            training_data = data[:,:,idx]
            S_prev = S.get_matrix()
            S.vals = train_step(S, k, training_data, batch_size, learning_rate, ld)
            if (torch.norm(S.get_matrix()-S_prev,'fro')<1e-6):
                sketch = S.get_matrix()
                print(sketch)
                torch.save(sketch, '{}.pt'.format(outfile))
                et = time.process_time()
                print('CPU Execution time:', et-st, 'seconds')
                return

        sketch = S.get_matrix()
        print(sketch)
        torch.save(sketch, '{}.pt'.format(outfile))
        et = time.process_time()
        print(device, 'Execution time:', et-st, 'seconds')


def hyperparameters(data, validation, m=10, k=10):

    # Optimization hyperparameters.
    batch_size = 128
    num_epochs = 100
    learning_rate = 1e-3
    ld = 1e-3

    # TODO how do we determine m and initialize S?
    n = data[0].shape[0]
    S = torch.rand([m, n], requires_grad=True)

    # 5-fold cross-validation, can change the number of folds
    kf = KFold(n_splits=5)
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        print(f"Fold {i}:")
        x_train = data[train_index]
        x_test = data[test_index]
        N = x_train.shape[0]

        for epoch in range(num_epochs):
            perm = np.random.permutation(N)
            start, stop = 0, batch_size
            print(f'Starting epoch {epoch} / {num_epochs}')
            for i in range(int(x_train.shape[0]/batch_size)):
                idx = perm[start:stop]
                x_batch = x_train[idx]
                start += batch_size
                stop += batch_size
                S = train_step(S, k, x_batch, batch_size, learning_rate, ld)

        test_loss = test(S, k, x_test)
        print("Emperical risk for fold "+str(i)+" is "+str(test_loss))

    # final evaluation after parameter is fixed
    '''
    validation_loss = test(S,k,validation)
    print("Emperical risk for validation is "+str(test_loss))
    '''


if __name__ == '__main__':
    m = 10
    k = 10
    nnz = 1
    dataset = "mnist"

    arg_list = sys.argv[1:]
    options = "m:k:z:d:"
    try:
        # Parsing argument
        arguments, values = getopt.getopt(arg_list, options)
        # checking each argument
        for curr_arg, curr_val in arguments:
            if curr_arg in ("-m"):
                m = int(curr_val)
            elif curr_arg in ("-k"):
                k = int(curr_val)
            elif curr_arg in ("-z"):
                nnz = int(curr_val)
            elif curr_arg in ("-d"):
                dataset = curr_val
    except getopt.error as err:
        print(str(err))

    main(m=m, k=k, nnz=nnz, dataset=dataset)
