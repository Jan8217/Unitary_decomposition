import pennylane as qml
from pennylane import numpy as np
import torch
from scipy.stats import unitary_group

# 生成一个8x8的随机酉矩阵
expected_qvalues = torch.tensor(unitary_group.rvs(8))

# 定义量子设备
dev = qml.device('default.qubit', wires=3)

# 参数化量子电路
@qml.qnode(device=dev, interface='torch')
def circuit(params):
    # 应用参数化量子门
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)
    return qml.state()

def loss(params):
    # 获取电路的酉操作
    qstate = qml.matrix(circuit(params))
    # 计算损失
    loss_A = torch.linalg.norm(qstate - expected_qvalues)*2

    return loss_A


# 初始化参数
params = torch.tensor(np.random.random(6), requires_grad=True)

# 选择优化器
optimizer = torch.optim.Adam([params], lr=0.01)

best_loss = loss(params)

# 优化循环
steps = 10000
for it in range(steps):
    optimizer.zero_grad()
    l = loss(params)
    l.backward()
    optimizer.step()

    if l < best_loss:  # keeps track of best parameters
        best_loss = l
        best_params = params

    if it % 10 == 9 or it == steps - 1:  # # Keep track of progress every 10 steps
        print("loss after {} steps is {:.3f}".format(it + 1, l))
    if l <= 0.1:
        output = "final loss after {} steps is {:.3f}".format(it + 1, l)
# print results
print("Target = ", params)

#if it % 10 == 0:
#        print(f"Step {it}: Loss = {l.item()}")