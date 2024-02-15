import numpy as np

# 定义单量子比特旋转门的矩阵
def rx(theta):
    return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], [-1j*np.sin(theta/2), np.cos(theta/2)]])

def ry(theta):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]])

# 定义CNOT门的矩阵
cnot = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])

# 参数
theta_rx = 0.352
theta_ry = 0.569

# 计算Rx和Ry门的矩阵，并扩展它们到三个量子比特
rx_matrix = np.kron(np.kron(rx(theta_rx), np.eye(2)), np.eye(2))  # Rx作用于第一个量子比特
ry_matrix = np.kron(np.eye(4), ry(theta_ry))  # Ry作用于第三个量子比特

# CNOT门已经是针对两个量子比特的，需要将其扩展到第一个和第二个量子比特
# 注意：CNOT门已经作用在第一个和第二个量子比特上，不需要额外扩展

# 计算整个电路的矩阵
circuit_matrix = ry_matrix.dot(cnot.dot(rx_matrix))

print(circuit_matrix)