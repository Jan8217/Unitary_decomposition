import pennylane as qml
from pennylane import numpy as np
import torch
from scipy.stats import unitary_group

# 定义量子设备
dev = qml.device('default.qubit', wires=3)

# 目标酉矩阵
U = unitary_group.rvs(8)
nr_qubits = 3
nr_layers = 2
@qml.qnode(dev, interface='torch')
def target_state():
    qml.QubitUnitary(U, wires=[0, 1, 2])
    return qml.state()

@qml.qnode(device=dev, interface="torch")  # creat circuit
def circuit(params):
    for j in range(nr_layers):
        for i in range(nr_qubits):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.RZ(params[2], wires=2)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 2])
            qml.RX(params[3], wires=0)
            qml.RY(params[4], wires=1)
            qml.RZ(params[5], wires=2)
    return qml.state()

def fidelity(state1, state2):
    """计算两个量子态之间的保真度"""
    return torch.abs(torch.dot(state1.conj(), state2))**2

def loss(params):
    # 获取电路的输出状态和目标状态
    output_state = circuit(params)
    target = target_state()
    # 计算损失为1减去保真度
    return 1 - fidelity(output_state, target)

# 初始化参数
params = torch.tensor(np.random.random(6), requires_grad=True)

# 选择优化器
optimizer = torch.optim.Adam([params], lr=0.01)

# 优化循环
steps = 100000
for it in range(steps):
    optimizer.zero_grad()
    l = loss(params)
    l.backward()
    optimizer.step()
    if it % 10 == 0:
        print(f"Step {it}: Loss = {l.item()}")