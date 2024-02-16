import argparse
import timeit
from circuit_ud_matrix import QDQN, Circuit_manager
from trainer_matrix import DQAS4RL
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("--num_layers", default=2, type=int)
parser.add_argument("--num_placeholders", default=15, type=int)
parser.add_argument("--num_qubits", default=3, type=int)
args = parser.parse_args()

start = timeit.default_timer()
ops = {0:("RZ", [0]), 1:("RZ", [1]), 2:("RZ", [2])
        , 3:("CNOT", [0]), 4:("CNOT", [1])
        , 5:("CNOTT", [0])
        , 6:("H", [2])
        , 7:("E", [0,1,2])}

sphc_struc = []
sphc_ranges = [[*range(3)] for _ in range(len(sphc_struc))]

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
                  lr=0.001,
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
                  total_epochs=1000,
                  struc_early_stop=0)
dqas4rl.learn()
stop = timeit.default_timer()
print(f'total time cost: {stop-start}')
