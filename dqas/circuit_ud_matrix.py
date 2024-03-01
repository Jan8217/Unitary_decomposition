from copy import deepcopy
import pennylane as qml
import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from itertools import zip_longest
from enum import Enum

SINGLEGATEFACTOR = 2 * np.pi
MULTIGATEFACTOR = 2 * np.pi


class Learning_State(Enum):
    LEARNING = 1
    FINISH = 0


class Circuit_manager():
    def __init__(self, sphc_struc, sphc_ranges, num_qubits
                 , num_placeholders, num_layers, ops
                 , noisy=False, learning_step=2, topK=2
                 , debug=False, learning_state=Learning_State.LEARNING.value):
        """
        current_layer_struc is changed by generator(tmp change) and updating(perpetual)
        learned_layer_struc contains top 2 gates for each leanring placeholder e.g. ["H,RZ", "RY", "E", "RZ", "E"]
        """
        self.learned_struc = []
        self.learned_ranges = []

        # sphc
        self._sphc_struc = sphc_struc
        self.sphc_ranges = sphc_ranges
        # learning state
        self.learning_state = learning_state  # must before init_struc
        # currents
        self.init_struc(num_placeholders=num_placeholders
                        , sphc_struc=sphc_struc
                        , sphc_ranges=sphc_ranges
                        , num_qubits=num_qubits
                        , layer_struc_parser=self._init_layer_struc_parser)
        # fixed numbers
        self._NUM_PLACEHOLDERS = len(self.learning_places)
        self._NUM_BLOCKS = len(self.current_layer_struc)
        self._NUM_QUBITS = num_qubits
        self._NUM_LAYERS = num_layers
        self._OPS = ops
        self.GATE_KEY = {}
        for k, v in ops.items():
            self.GATE_KEY[v[0]] = k
        self._NUM_OPS = len(ops)
        self._NUM_OPS_W = len(ops) + 1 if len(sphc_struc) else len(ops)  # op1, op2, op3 | op4 (from sphc_struc)
        self._TOPK = topK
        self._LEARNING_STEP = learning_step
        self._NOISY = noisy

        # currents
        self.indexs = [self.num_ops_w - 1 for _ in range(self.num_blocks)]
        self.current_learning_places = self.learning_places[:self._LEARNING_STEP]  # order is important
        self.remaining_learning_places = self.learning_places[self._LEARNING_STEP:]
        self._current_num_placeholders = len(self.current_learning_places)

        # learned struc
        self.learned_layer_struc = deepcopy(self.current_layer_struc)
        self.learned_layer_ranges = deepcopy(self.current_layer_ranges)
        self.debug = debug

        # var gates
        self.var_gates = ["U1", "U3", "CU3", "CU33", "CU3-single", "RX", "RY", "RZ", "CRZ", "ZZ", "rz-CNOT-rz",
                          "rz-CNOTT-rz"]
        # sampled struc
        self.current_sampled_struc = []

    def set_current_sampled_struc(self, value):
        # current_sampled_struc: [1 3]
        self.current_sampled_struc = value

    def get_current_sampled_struc(self):
        # current_sampled_struc: [1 3]
        return self.current_sampled_struc

    def add_learned_layer_struc(self, struc, ranges, places):
        self.learned_struc.append((struc.copy(), ranges.copy(), places.copy()))

    def init_struc(self, num_placeholders, sphc_struc, sphc_ranges, num_qubits, layer_struc_parser):
        """
        The learning circuit is described by
            1. current_layer_struc: ["E", "RZ", "E", "CNOT", "E"] end with "E" iff num_placeholders > len(sphc_struc)
            2. current_layer_ranges: [[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3]]
            3. learning_places [0, 2]
        """
        self.current_layer_struc, self.current_layer_ranges, self.learning_places = layer_struc_parser(
            num_placeholders=num_placeholders
            , sphc_struc=sphc_struc
            , sphc_ranges=sphc_ranges
            , num_qubits=num_qubits)

    def update_learning_places(self, prob):
        self.update_struc(prob)
        self.current_learning_places = self.remaining_learning_places[:self._LEARNING_STEP]
        self.remaining_learning_places = self.remaining_learning_places[self._LEARNING_STEP:]
        self._current_num_placeholders = len(self.current_learning_places)

        if self.current_num_placeholders > 0:
            return False
        else:
            self.learning_state = Learning_State.FINISH.value
            return True

    def update_struc(self, prob):

        for i, current_plh_i in enumerate(prob):
            _, best_gate_idxs = torch.topk(current_plh_i, k=self._TOPK)
            if self.debug:
                if i % len(prob) == 0:
                    print(f"best gate idx: {best_gate_idxs}")
            self.current_layer_struc[self.current_learning_places[i]] = self.ops[best_gate_idxs[0].item()][0]
            self.current_layer_ranges[self.current_learning_places[i]] = self.ops[best_gate_idxs[0].item()][1]
            self.learned_layer_struc[self.current_learning_places[i]] = ",".join(
                self.ops[j.item()][0] for j in best_gate_idxs)
            self.learned_layer_ranges[self.current_learning_places[i]] = [self.ops[j.item()][1] for j in best_gate_idxs]
            self.indexs[self.current_learning_places[i]] = best_gate_idxs[0].item()

    def _init_layer_struc_parser(self, num_placeholders, sphc_struc, sphc_ranges, num_qubits):
        if num_placeholders > 0:
            tmp_layer_struc = zip_longest(["E" for _ in range(num_placeholders)], sphc_struc, fillvalue="")
            layer_struc = [gate for t in tmp_layer_struc for gate in t if len(gate) > 0]  # [E, E, E]
            layer_ranges = [[*range(num_qubits)] for _ in range(len(layer_struc))]
            learning_places = [i for i, plh in enumerate(layer_struc) if plh == "E"]
        else:
            layer_struc = deepcopy(sphc_struc)
            layer_ranges = deepcopy(sphc_ranges)
            learning_places = []
            self.learning_state = Learning_State.FINISH.value
        # e.g. ["E", "RY", "E", "RZ", "E", "CNOT", "E"] if n_plds > n_sphc

        return layer_struc, layer_ranges, learning_places

    def get_layer_generator(self, gates, ranges=[]):
        # gns = [1,0,1]
        # print(gates)
        def make_layer_generator(gates, num_blocks, ops, num_ops_w, current_layer_struc, current_layer_ranges,
                                 current_learning_places, ranges, indexs):
            for i, gate in enumerate(gates):
                current_layer_struc[current_learning_places[i]] = ops[gate][0]
                current_layer_ranges[current_learning_places[i]] = ops[gate][1]
                if len(ranges):
                    assert (len(ranges) == len(gates))
                    assert (len(ranges[i]) > 0)
                    current_layer_ranges[current_learning_places[i]] = ranges[i]
                indexs[current_learning_places[i]] = gate

            return current_layer_struc, current_layer_ranges, indexs  # ["U3", "RY", "CU3", "RZ", "U3", "CNOT", "U3"]

        def make_learned_layer_generator(num_blocks, num_ops_w, learning_places, learned_layer_struc,
                                         learned_layer_ranges, gate_key, indexs):



            return learned_layer_struc, learned_layer_ranges, indexs

        if self.learning_state == Learning_State.LEARNING.value:
            return make_layer_generator(gates=gates
                                        , num_blocks=self.num_blocks
                                        , ops=self.ops
                                        , num_ops_w=self.num_ops_w
                                        , current_layer_struc=self.current_layer_struc
                                        , current_layer_ranges=self.current_layer_ranges
                                        , current_learning_places=self.current_learning_places
                                        , ranges=ranges
                                        , indexs=self.indexs
                                        )
        else:
            return make_learned_layer_generator(num_blocks=self.num_blocks
                                                , num_ops_w=self.num_ops_w
                                                , learning_places=self.learning_places
                                                , learned_layer_struc=self.current_layer_struc
                                                , learned_layer_ranges=self.current_layer_ranges
                                                , gate_key=self.GATE_KEY
                                                , indexs=self.indexs
                                                )

    def get_weights_shape(self):
        return (self.num_layers, self.num_qubits, self.num_blocks, self.num_ops_w)

    def check_gate(self, gate: str):
        return gate in self.var_gates
    def get_learned_layer_struc(self):
        return self.learned_layer_struc, self.learned_layer_ranges

    def get_current_layer_struc(self):
        return self.current_layer_struc, self.current_layer_ranges

    @property
    def num_ops(self):
        return self._NUM_OPS

    @property
    def num_ops_w(self):
        return self._NUM_OPS_W

    @property
    def num_blocks(self):  # caution ! check this for every new current layer struc
        return self._NUM_BLOCKS

    @property
    def num_qubits(self):
        return self._NUM_QUBITS

    @property
    def num_layers(self):
        return self._NUM_LAYERS

    @property
    def noisy(self):
        return self._NOISY

    @property
    def sphc_struc(self):
        return self._sphc_struc

    @property
    def current_num_placeholders(self):
        return self._current_num_placeholders

    @property
    def num_placeholders(self):
        return self._NUM_PLACEHOLDERS

    @property
    def ops(self):
        return self._OPS

    @property
    def learning_step(self):
        return self._LEARNING_STEP

def circuit(cm: Circuit_manager, data_reuploading=False, barrier=False):  # keep it outside DQN
    # dev = qml.device('qiskit.aer', wires=num_qubits)
    if cm.noisy:
        dev = qml.device("default.mixed", wires=cm.num_qubits)
    else:
        dev = qml.device("default.qubit", wires=cm.num_qubits)

    # U3 weights
    shapes = {
        "theta_weights": cm.get_weights_shape(),
        "phi_weights": cm.get_weights_shape(),
        "delta_weights": cm.get_weights_shape(),
    }
    def measure_block(num_qs):
        measurement = qml.PauliZ(wires=0)
        for i in range(1, num_qs):
            measurement @= qml.PauliZ(wires=i)
        return [qml.expval(measurement)]

    def encoding_block(inputs, num_qubits):
        for i in range(num_qubits):
            qml.Hadamard(wires=i)

    def layer(theta_weight, phi_weight, delta_weight, generators, generators_range, indexs):
        for p, op in enumerate(generators):
            op_range = generators_range[p]
            # -- u3 & cu3 gate --
            if op == "U3":
                for i in op_range:
                    qml.U3(theta=theta_weight[i, p, indexs[p]] * MULTIGATEFACTOR
                           , phi=phi_weight[i, p, indexs[p]] * MULTIGATEFACTOR
                           , delta=delta_weight[i, p, indexs[p]] * MULTIGATEFACTOR
                           , wires=i)
            elif op == "CU3":  # ring connection
                assert (len(op_range) > 1)
                for i in op_range:
                    def CU3(t_w, p_w, d_w, target):
                        qml.U3(theta=t_w[i, p, indexs[p]] * MULTIGATEFACTOR
                               , phi=p_w[i, p, indexs[p]] * MULTIGATEFACTOR
                               , delta=d_w[i, p, indexs[p]] * MULTIGATEFACTOR
                               , wires=target)

                    qml.ctrl(CU3, control=i)(theta_weight, phi_weight, delta_weight, (i + 1) % len(op_range))
            elif op == "CU3-single":
                for i in op_range:
                    def CU3(t_w, p_w, d_w, target):
                        qml.U3(theta=t_w[i, p, indexs[p]] * MULTIGATEFACTOR
                               , phi=p_w[i, p, indexs[p]] * MULTIGATEFACTOR
                               , delta=d_w[i, p, indexs[p]] * MULTIGATEFACTOR
                               , wires=target)

                    qml.ctrl(CU3, control=i)(theta_weight, phi_weight, delta_weight, (i + 1) % cm.num_qubits)
            elif op == "CU33":
                for i in op_range:
                    def CU3(t_w, p_w, d_w, target):
                        qml.U3(theta=t_w[i, p, indexs[p]] * MULTIGATEFACTOR
                               , phi=p_w[i, p, indexs[p]] * MULTIGATEFACTOR
                               , delta=d_w[i, p, indexs[p]] * MULTIGATEFACTOR
                               , wires=target)

                    qml.ctrl(CU3, control=i)(theta_weight, phi_weight, delta_weight, (i + 2) % cm.num_qubits)
            # -- simple one qubit gate --
            elif op == "U1":
                for i in op_range:
                    qml.U1(theta_weight[i, p, indexs[p]], wires=i) * SINGLEGATEFACTOR
            elif op == "RX":
                for i in op_range:
                    qml.RX(theta_weight[i, p, indexs[p]] * SINGLEGATEFACTOR, wires=i)
            elif op == "RY":
                for i in op_range:
                    qml.RY(theta_weight[i, p, indexs[p]] * SINGLEGATEFACTOR, wires=i)
            elif op == "RZ":
                for i in op_range:
                    qml.RZ(theta_weight[i, p, indexs[p]] * SINGLEGATEFACTOR, wires=i)
            elif op == "X":
                for i in op_range:
                    qml.X(wires=i)
            elif op == "T":
                for i in op_range:
                    qml.T(wires=i)
            elif op == "Ta":
                for i in op_range:
                    qml.adjoint(qml.T(wires=i))
            elif op == "H":
                for i in op_range:
                    qml.Hadamard(wires=i)
            elif op == "E":
                for i in op_range:
                    qml.Identity(wires=i)
            # -- two qubits gate --
            elif op == "CNOT":
                for i in op_range:
                    qml.CNOT(wires=[i, (i + 1) % cm.num_qubits])
            elif op == "CNOTT":
                for i in op_range:
                    qml.CNOT(wires=[i, (i + 2) % cm.num_qubits])
            elif op == "rz-CNOT-rz":
                i = op_range[0]
                qml.RZ(theta_weight[i + 1, p, indexs[p]] * SINGLEGATEFACTOR, wires=i + 1)
                qml.CNOT(wires=[i, (i + 1) % cm.num_qubits])
                qml.RZ(phi_weight[i + 1, p, indexs[p]] * SINGLEGATEFACTOR, wires=i + 1)
            elif op == "rz-CNOTT-rz":
                i = op_range[0]
                qml.RZ(theta_weight[i + 2, p, indexs[p]] * SINGLEGATEFACTOR, wires=i + 2)
                qml.CNOT(wires=[i, (i + 2) % cm.num_qubits])
                qml.RZ(phi_weight[i + 2, p, indexs[p]] * SINGLEGATEFACTOR, wires=i + 2)
            elif op == "rz-CNOT":
                i = op_range[0]
                qml.RZ(theta_weight[i + 1, p, indexs[p]] * SINGLEGATEFACTOR, wires=i + 1)
                qml.CNOT(wires=[i, (i + 1) % cm.num_qubits])
            elif op == "CNOT-rz":
                i = op_range[0]
                qml.CNOT(wires=[i, (i + 1) % cm.num_qubits])
                qml.RZ(phi_weight[i + 1, p, indexs[p]] * SINGLEGATEFACTOR, wires=i + 1)
            elif op == "rz-CNOTT-rz":
                i = op_range[0]
                qml.RZ(theta_weight[i + 2, p, indexs[p]] * SINGLEGATEFACTOR, wires=i + 2)
                qml.CNOT(wires=[i, (i + 2) % cm.num_qubits])
                qml.RZ(phi_weight[i + 2, p, indexs[p]] * SINGLEGATEFACTOR, wires=i + 2)
            elif op == "ZZ":
                assert (len(op_range) > 1)
                for i in op_range:
                    if i + 1 in op_range:
                        qml.IsingZZ(theta_weight[(i + 1) % len(op_range), p, indexs[p]],
                                    wires=[i, (i + 1) % len(op_range)])
            elif op == "QFT":
                assert (len(op_range) > 1)
                qml.QFT(wires=op_range)

    def make_circuit(inputs, theta_weights, phi_weights, delta_weights):
        gns, gns_ranges, indexs = cm.get_layer_generator(gates=cm.get_current_sampled_struc())

        for l in range(cm.num_layers):
            if data_reuploading or l == 0:
                encoding_block(inputs, cm.num_qubits)
            if barrier:
                qml.Barrier(wires=range(cm.num_qubits))
            layer(theta_weights[l], phi_weights[l], delta_weights[l], gns, gns_ranges, indexs)
            if barrier:
                qml.Barrier(wires=range(cm.num_qubits))
        if cm.noisy:
            for i in range(cm.num_qubits):
                qml.BitFlip(0.01, wires=i)
        return measure_block(cm.num_qubits)
    circuit = qml.QNode(make_circuit, dev, interface='torch')
    model = qml.qnn.TorchLayer(circuit, shapes)
    return model
class QDQN(nn.Module):
    def __init__(self
                 , cm: Circuit_manager
                 , data_reuploading=False
                 , barrier=False
                 , seed=1234):
        super(QDQN, self).__init__()
        torch.manual_seed(seed)
        self.num_qubits = cm.num_qubits
        self.cm = cm
        self.q_layers = circuit(cm=self.cm, data_reuploading=data_reuploading, barrier=barrier)

    def forward(self, inputs):

        return qml.matrix(self.q_layers.qnode)(inputs, self.q_layers.theta_weights, self.q_layers.phi_weights, self.q_layers.delta_weights)
        #return qml.state()

    def set_circuit_struc(self, gates):
        self.cm.set_current_sampled_struc(gates)
