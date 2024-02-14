import pennylane as qml
import numpy as np
import torch
from torch.autograd import Variable
from scipy.stats import unitary_group
np.random.seed(42)

# we generate a three-dimensional random vector by sampling
# each entry from a standard normal distribution
v = np.random.normal(0, 1, 3)
device = qml.device('default.qubit', wires=3)


# number of qubits in the circuit
nr_qubits = 3
# number of layers in the circuit
nr_layers = 2

# randomly initialize parameters from a normal distribution
params = np.random.normal(0, np.pi, (nr_qubits, nr_layers, 3))
params = Variable(torch.tensor(params), requires_grad=True)
#print(params)

# a layer of the circuit
def layer(params, j):
    for i in range(nr_qubits):
        qml.RX(params[i, j, 0], wires=i)
        qml.RY(params[i, j, 1], wires=i)
        qml.RZ(params[i, j, 2], wires=i)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[0, 2])
    qml.CNOT(wires=[1, 2])


@qml.qnode(device=device, interface="torch")
def circuit(param_for_gate):
    # repeatedly apply each layer in the circuit
    for j in range(nr_layers):
        layer(param_for_gate, j)
    return qml.expval(qml.PauliZ(wires=0))

def matrix_loss_func(a, b):
    return torch.linalg.norm(b - a)
# cost function
def cost_fn(params):
    expected_qvalues = unitary_group.rvs(8)
    predicted = qml.matrix(circuit(params))
    sub_loss = matrix_loss_func(predicted, expected_qvalues) * 1
    #for k in range(3):
    #    cost = torch.abs(circuit(params)-target)**2
    return sub_loss

# set up the optimizer
opt = torch.optim.Adam([params], lr=0.1)

# number of steps in the optimization routine
steps = 500

# the final stage of optimization isn't always the best, so we keep track of
# the best parameters along the way
best_cost = cost_fn(params)
best_params = np.zeros((nr_qubits, nr_layers, 3))

print("Cost after 0 steps is {:.4f}".format(cost_fn(params)))

# optimization begins
for n in range(steps):
    opt.zero_grad()
    loss = cost_fn(params)
    loss.backward()
    opt.step()

    # keeps track of best parameters
    if loss < best_cost:
        best_cost = loss
        best_params = params

    # Keep track of progress every 10 steps
    if n % 10 == 9 or n == steps - 1:
        print("Cost after {} steps is {:.4f}".format(n + 1, loss))

# print results
print("Target Bloch vector = ", params)
#print("Output Bloch vector = ", actually_unita)
