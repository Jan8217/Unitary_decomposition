import numpy as np
from qiskit import QuantumCircuit

circ = QuantumCircuit(3)

# 在qubit0上添加一个H门，使量子位形成叠加态
circ.h(0)
# 添加一个CX(CNOT)门，qubit 0为控制位，qubit 1为目标位，使量子位形成Bell态
circ.cx(0,1)
# 添加一个CX(CNOT)门，qubit 0位控制位，qubit 2位目标位，使量子位形成GHZ态
circ.cx(0,2)

circ.draw('mpl')