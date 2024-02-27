import argparse
import timeit
from circuit_ud_matrix import QDQN, Circuit_manager
from trainer_matrix import DQAS4RL

parser = argparse.ArgumentParser()
parser.add_argument("--num_layers", default=1, type=int)
parser.add_argument("--num_placeholders", default=20, type=int)
parser.add_argument("--num_qubits", default=3, type=int)
args = parser.parse_args()

start = timeit.default_timer()
ops = {0:("U3", [0]), 1:("U3", [1]), 2:("U3", [2])
        , 3:("CU3-single", [0]), 4:("CU3-single", [1])
        , 5:("CU33", [0])
        , 6:("CNOT",[0]), 7:("CNOT",[1])
        , 8:("CNOTT", [0])
        , 9:("H", [2])
        , 10:("E", [0,1,2])}
sphc_struc = []
sphc_ranges = [[*range(args.num_qubits)] for _ in range(len(sphc_struc))]
cm = Circuit_manager(sphc_struc=sphc_struc
                    , sphc_ranges=sphc_ranges
                    , num_qubits=args.num_qubits
                    , num_placeholders=args.num_placeholders
                    , num_layers=args.num_layers
                    , ops=ops
                    , noisy=False
                    , learning_step=5)

# Define quantum network
qdqn = QDQN(cm=cm
        , data_reuploading=True
        , barrier=False
        , seed=1234)
dqas4rl = DQAS4RL(qdqn=qdqn,
                  lr=0.01,
                  lr_struc=0.01,
                  batch_size=32,
                  update_model=1,
                  update_targ_model=50,
                  memory_size=10000,
                  max_steps=200,
                  seed=1234,
                  cm=cm,
                  prob_max = 0,
                  loss_func='MSE',
                  opt='Adam',
                  opt_struc='Adam',
                  logging=True,
                  early_stop=195,
                  structure_batch=10,
                  struc_learning=cm.learning_state,
                  total_epochs=100,
                  struc_early_stop=0)
dqas4rl.learn()
stop = timeit.default_timer()
print(f'total time cost: {stop-start}')