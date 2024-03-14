import argparse
import timeit
from circuit_ud_matrix import QDQN, Circuit_manager
from trainer_matrix import DQAS4RL


dimensions = [0.0005,0.0001,0.00015]
n_repeats = 1
for i, in zip(dimensions):
    for k in range(n_repeats):
        parser = argparse.ArgumentParser()
        parser.add_argument("--num_layers", default=1, type=int)
        parser.add_argument("--num_placeholders", default=25, type=int)
        parser.add_argument("--num_qubits", default=3, type=int)
        args = parser.parse_args()

        start = timeit.default_timer()
        ops = {0:("U3", [0]), 1:("U3", [1]), 2:("U3", [2])
                , 3:("CU3-single", [0]), 4:("CU3-single", [1])
                , 5:("CU33", [0])
                , 6:("CNOT",[0]), 7:("CNOT",[1])
                , 8:("CNOTT", [0])
                , 9:("H", [2])
                , 10:("E", [0,1,2]), 11: ("RX", [0]), 12: ("RY", [0]), 13: ("RZ", [0])
                , 14: ("RX", [1]), 15: ("RY", [1]), 16: ("RZ", [1])
                , 17: ("RX", [2]), 18: ("RY", [2]), 19: ("RZ", [2])

        }
        #ops = {0: ("RX", [0,1]), 1: ("RY", [1,2]), 2: ("RZ", [0, 2])
        #    , 3: ("CNOT", [0,1]), 4: ("CNOTT", [0]), 5: ("H", [2]), 6: ("E", [0, 1, 2])}
        #ops = {0: ("RZ", [0]), 1: ("RZ", [1]), 2: ("RZ", [2])
        #    , 3: ("CNOT", [0]), 4: ("CNOT", [1])
        #    , 5: ("CNOTT", [0])
        #    , 6: ("H", [2])
        #    , 7: ("E", [0,1,2])}

        sphc_struc = []
        sphc_ranges = [[*range(args.num_qubits)] for _ in range(len(sphc_struc))]
        cm = Circuit_manager(sphc_struc=sphc_struc
                            , sphc_ranges=sphc_ranges
                            , num_qubits=args.num_qubits
                            , num_placeholders=args.num_placeholders
                            , num_layers=args.num_layers
                            , ops=ops
                            , noisy=False
                            , learning_step=10)

        # Define quantum network
        qdqn = QDQN(cm=cm
                , data_reuploading=True
                , barrier=False
                , seed=1234)
        dqas4rl = DQAS4RL(qdqn=qdqn,
                          lr=i,
                          lr_struc=i,
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
                          total_epochs=200000,
                          struc_early_stop=0)

        dqas4rl.learn()
        stop = timeit.default_timer()
        print(f'total time cost: {stop-start}')
