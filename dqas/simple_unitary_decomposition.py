import pennylane as qml
import numpy as np
import torch
from torch.autograd import Variable
from scipy.stats import unitary_group

device = qml.device('default.qubit', wires=3)
nr_qubits = 3
nr_layers = 2
params = np.random.normal(0, np.pi, (nr_qubits, nr_layers, 3))
params = Variable(torch.tensor(params), requires_grad=True)

@qml.qnode(device=device, interface="torch")  # creat circuit
def circuit(params):
    for j in range(nr_layers):
        for i in range(nr_qubits):
            qml.RX(params[i, j, 0], wires=i)
            qml.RY(params[i, j, 1], wires=i)
            qml.RZ(params[i, j, 2], wires=i)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 2])
            qml.CNOT(wires=[1, 2])
        #qml.CNOT(wires=[0, 1])

    return qml.expval(qml.PauliZ(wires=0))
    #return qml.probs(wires=0)
unitary_matrix = qml.matrix(circuit)(params)
print(unitary_matrix)
def loss_fn(params): # define lost function
    unitary_matrix = qml.matrix(circuit)(params)
    #print(unitary_matrix)
    #print(qml.draw(circuit(params)))
    expected_qvalues = torch.tensor(unitary_group.rvs(8))
    loss = torch.linalg.norm(unitary_matrix - expected_qvalues) * 1
    return loss

opt = torch.optim.Adam([params], lr=0.001)  # set up the optimizer
epoch = 10000 # number of epoch in the optimization routine

best_cost = loss_fn(params)
best_params = np.zeros((nr_qubits, nr_layers, 3))

for n in range(epoch): # optimization begins
    opt.zero_grad()
    loss = loss_fn(params)
    loss.backward()
    opt.step()

    if loss < best_cost:  # keeps track of best parameters
        best_cost = loss
        best_params = params

    if n % 10 == 9 or n == epoch - 1: # # Keep track of progress every 10 steps
        print("loss after {} steps is {:.3f}".format(n + 1, loss))
    if loss <= 0.1:
        output = "final loss after {} steps is {:.3f}".format(n + 1, loss)
# print results
print("Target = ", params)
