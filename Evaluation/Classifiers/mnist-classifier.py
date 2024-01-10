import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

x = np.load("fashion_mnist_images.npy")
y = np.load("fashion_mnist_labels.npy")
d, n = x.shape

x_new = np.insert(x, 0, np.ones_like(y), axis=0)
training = x_new[:, 0:5000]


class Newton:

    def __init__(self, x, y, n):
        self.x = x
        self.y = y
        self.n = n

    def obj(self, theta):
        j = 0
        for i in range(0, self.n):
            e = np.exp(-self.y[0][i] * np.dot(theta, self.x[:, i]))
            j += np.log(1 + e)
        reg = np.dot(theta, theta)
        return j + reg

    def grad(self, theta):
        grad = np.zeros(d+1)
        for i in range(0, self.n):
            e = np.exp(-self.y[0][i] * np.dot(theta, self.x[:, i]))
            k = (-self.y[0][i] * e) / (1+e)
            grad += k*self.x[:, i]
        return grad + 2*theta

    def hess(self, theta):
        hess = np.zeros([d+1, d+1])
        for i in range(0, self.n):
            e = np.exp(-self.y[0][i] * np.dot(theta, self.x[:, i]))
            k = ((self.y[0][i]**2)*e) / ((1+e)**2)
            hess += k*np.outer(self.x[:, i], self.x[:, i])
        return hess + 2*np.eye(d+1)

    def calc(self):
        eps = 10**-6
        theta_curr = np.zeros(d+1)
        theta_prev = theta_curr
        num_iterations = 0
        while True:
            num_iterations += 1
            grad_j = self.grad(theta_prev)
            hess_j = self.hess(theta_prev)
            old_val = self.obj(theta_prev)
            theta_curr = theta_prev - np.matmul(inv(hess_j), grad_j)
            new_val = self.obj(theta_curr)
            if abs(new_val - old_val)/old_val <= eps:
                break
            theta_prev = theta_curr
        return theta_curr, num_iterations


def boundary_vals(x, theta):
    d, n = x.shape
    bound_vals = np.empty(n)
    for i in range(0, n):
        boundary = np.dot(theta, x[:, i])
        bound_vals[i] = boundary
    return bound_vals



newt = Newton(x_new, y, 5000)
theta, num_iterations = newt.calc()

test_x = x_new[:, 5000:]
test_y = y[:, 5000:]

bound_vals = boundary_vals(test_x, theta)

predictions = np.sign(bound_vals)

diff = test_y[0] - predictions

print("Correct % {}".format(np.sum(diff == 0) / 1000))
print("Number of iterations: {}".format(num_iterations))
print("Objective value: {}".format(newt.obj(theta)))

bad = np.zeros_like(bound_vals)

for i in range(0, 1000):
    if diff[i] == 0:
        bad[i] = np.Inf
    else:
        bad[i] = bound_vals[i]

indices = np.argsort(np.absolute(bad))[:20] + 5000

figure, axis = plt.subplots(4, 5)

count = 0
for i in range(0, 4):
    for j in range(0, 5):
        idx = indices[count]
        axis[i, j].imshow(np.reshape(x[:, idx], (int(np.sqrt(d)), int(np.sqrt(d)))), cmap="Greys")
        group = "Dress" if y[0][idx] == 1 else "Coat"
        axis[i, j].set_title(group, fontstyle='italic')
        count += 1
plt.show()
