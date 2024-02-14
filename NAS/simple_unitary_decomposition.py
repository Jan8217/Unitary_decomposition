import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=3, shots=1000)

