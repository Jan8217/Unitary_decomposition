import os
import torch
import numpy as np
from torch import nn
import torch.optim as topt
from circuit_ud_matrix import Circuit_manager
import datetime
import csv
from scipy.stats import unitary_group
import json
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

#
#TARGET = torch.Tensor(np.array([[1., 0., 0., 0., 0., 0., 0., 0.]
#                       , [0., 1., 0., 0., 0., 0., 0., 0.]
#                       , [0., 0., 1., 0., 0., 0., 0., 0.]
#                       , [0., 0., 0., 1., 0., 0., 0., 0.]
#                       , [0., 0., 0., 0., 1., 0., 0., 0.]
#                       , [0., 0., 0., 0., 0., 1., 0., 0.]
#                       , [0., 0., 0., 0., 0., 0., 0., 1.]
#                       , [0., 0., 0., 0., 0., 0., 1., 0.]]))
LOSS_FACTOR = 1
#TARGET = torch.eye(8)
TARGET = torch.tensor(unitary_group.rvs(8))

class DQAS4RL:
    def __init__(self,
                 qdqn,
                 lr,
                 lr_struc,
                 batch_size,
                 update_model,
                 update_targ_model,
                 memory_size,
                 max_steps,
                 seed,
                 cm: Circuit_manager,
                 prob_max=0.5,
                 loss_func='MSE',
                 opt='Adam',
                 opt_struc='Adam',
                 logging=False,
                 sub_batch_size=1,
                 early_stop=195,
                 structure_batch=10,
                 total_epochs=5000,
                 struc_learning=True,
                 struc_early_stop=0,
                 min_loss=0.1):

        self.qdqn = qdqn
        self.cm = cm
        self.lr = lr
        self.lr_struc = lr_struc
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.seed = seed
        self.update_model = update_model
        self.update_targ_model = update_targ_model
        self.loss_func_name = loss_func
        self.opt_name = opt
        self.opt_name_struc = opt_struc
        self.prob_max = prob_max
        self.avcost = 0
        self.early_stop = early_stop
        self.min_loss = min_loss
        self.struc_early_stop = struc_early_stop if struc_early_stop else int(total_epochs / (2 * (int(cm.num_placeholders / cm.learning_step))))
        self.struc_early_step = self.struc_early_stop
        self.struc_learning = struc_learning
        self.total_epochs = total_epochs
        self.device = 'cpu'
        self.logging = logging
        self.sub_batch_size = sub_batch_size
        self.structure_batch = structure_batch
        self.dtype = torch.float32
        self.config()

    def config(self):
        # device
        self.device = torch.device(self.device)

        # set network to device
        self.qdqn = self.qdqn.to(self.device)

        # initial struc parameters
        if self.struc_learning:
            v_struc_init = np.zeros([self.cm.current_num_placeholders, self.cm.num_ops], dtype=np.float32)
            self.tmp_ops_candidates = np.arange(self.cm.num_ops, dtype=np.float32)
            self.tmp_ops_candidates = list(np.tile(self.tmp_ops_candidates, (self.cm.current_num_placeholders, 1)))
            self.var_struc = torch.tensor(v_struc_init, requires_grad=True, dtype=self.dtype, device=self.device)

        # set loss functions
        self.loss_func = getattr(nn, self.loss_func_name + 'Loss')()
        torch_opt = getattr(topt, self.opt_name)
        if self.struc_learning:
            torch_opt_struc = getattr(topt, self.opt_name_struc)

        params = []
        params.append({'params': self.qdqn.q_layers.parameters()})

        # set optimizers
        self.opt = torch_opt(params, lr=self.lr)
        if self.struc_learning:
            self.opt_struc = torch_opt_struc([self.var_struc], lr=self.lr_struc, betas=(0.9, 0.99), weight_decay=0.001)

        if self.logging:
            # exp_name = datetime.now().strftime("DQN-%d_%m_%Y-%H_%M_%S")
            if not os.path.exists('./logs/'):
                os.makedirs('./logs/')
            out_path = "new_operation_pool_23"
            self.log_dir = f'./logs/{out_path}/'
            self.reprot_dir = f'./logs/{out_path}/reports/'
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            if not os.path.exists(self.reprot_dir):
                os.makedirs(self.reprot_dir)
            self.writer = SummaryWriter(log_dir=self.log_dir)

        self.reset()


    def matrix_loss_func(self, a, b):
        return torch.linalg.norm(b - a)
        #return 1 - qml.math.fidelity(b, a, check_state= True)

    def reset(self):
        self.global_step = 0
        self.epoch_count = 0

    def push_json(self, out, path):
        with open(path, 'w') as f:
            json.dump(out, f, indent=4)

    def train_structure(self):
        pass

    def make_inputs(self, states, struc):
        return torch.cat((states, struc), 1)

    def wrapper(self, input):
        while True:
            yield input

    def preset_byprob(self, prob):
        return np.array([np.random.choice(np.arange(prob.size()[1])
                        , p=np.array(prob[i].detach().cpu().clone().numpy())) for i in range(prob.size()[0])])

    def train_model(self, prob):

        self.qdqn.train()
        self.opt.zero_grad()

        # structure training
        loss_list = []
        deri_struc = []

        if self.prob_max and self.struc_learning:
            prob = torch.clamp(prob, min=(1 - self.prob_max) / self.cm.num_ops, max=self.prob_max)
            prob = torch.transpose(prob.t() / torch.sum(prob, dim=1), 0, 1)

        grad_params = {}
        if self.struc_learning:
            self.opt_struc.zero_grad()
            sum_sample_strucs = 0
            for idx in range(self.structure_batch):
                # -- sample struc --
                sample_struc = self.preset_byprob(prob)
                self.cm.set_current_sampled_struc(sample_struc)

                # -- predicted --
                states = torch.Tensor([0.] * self.cm.num_qubits)
                predicted = self.qdqn(states)
                #output_state = circuit(params)
                # -- target --
                expected_qvalues = TARGET
                #expected_qvalues = target_state()

                # -- loss --
                sub_loss = self.matrix_loss_func(predicted, expected_qvalues) * LOSS_FACTOR
                sub_sum = sum(
                    map(self.cm.check_gate, [self.cm.ops[gate][0] for gate in sample_struc] + self.cm.sphc_struc))

                sum_sample_strucs += sub_sum
                if sub_sum:
                    sub_loss.backward()
                    ## -- gradient --
                    for name, param in self.qdqn.named_parameters():

                        if param.grad is None:
                            gr = torch.from_numpy(np.zeros(self.cm.get_weights_shape(), dtype=np.float32)).type(self.dtype)
                        else:
                            gr = param.grad.detach().clone()
                        summary = grad_params.get(name, [])
                        summary.append(gr)
                        grad_params[name] = summary

                loss_list.append(sub_loss)
                with torch.no_grad():
                    grad_struc = (-prob).type(self.dtype).to(self.device).index_put(
                        indices=tuple(torch.LongTensor(list(zip(range(self.cm.current_num_placeholders), sample_struc))).t())
                        , values=torch.ones([self.cm.current_num_placeholders], dtype=self.dtype, device=self.device)
                        , accumulate=True).to(self.device)
                    deri_struc.append(((sub_loss - self.avcost) * grad_struc).to(self.device))

            with torch.no_grad():
                total_loss = torch.mean(torch.stack(loss_list)) * LOSS_FACTOR
                self.avcost = torch.mean(torch.stack(loss_list)) * LOSS_FACTOR

            if sum_sample_strucs:
                with torch.no_grad():
                    for name, param in self.qdqn.named_parameters():
                        if name in grad_params.keys():
                            param.grad = torch.mean(torch.stack(grad_params[name]), dim=0).type(self.dtype).to(
                                self.device)

                self.opt.step()

            with torch.no_grad():
                grad_batch_struc = torch.mean(torch.stack(deri_struc), dim=0).type(self.dtype).to(self.device)
                self.var_struc.grad = grad_batch_struc

            self.opt_struc.step()
        else:
            self.qdqn.train()
            self.opt.zero_grad()
            # -- predicted --
            states = torch.Tensor([.0] * self.cm.num_qubits)
            predicted = self.qdqn(states)

            # -- target --
            expected_qvalues = TARGET
            #expected_qvalues = target_state()
            total_loss = self.matrix_loss_func(predicted, expected_qvalues)
            total_loss.backward()
            self.opt.step()
        return total_loss.item()

    def update_prob(self):
        with torch.no_grad():
            prob = torch.zeros(self.var_struc.shape)
            for i in range(self.cm.current_num_placeholders):
                row_sum = torch.sum(torch.exp(self.var_struc[i][self.tmp_ops_candidates[i]]))
                for j, v in enumerate(self.var_struc[i]):
                    if j in self.tmp_ops_candidates[i]:
                        prob[i, j] = torch.exp(v) / row_sum
        return prob

    def epoch_train(self, epoch):
        epoch_loss = []

        prob = self.update_prob()

        # train in batch
        if epoch % self.update_model == 0:
            loss = self.train_model(prob)
            epoch_loss.append(loss)
            print('[%d] loss: %.3E' % (epoch + 1, loss))

        self.epoch_count += 1
        epoch_avg_loss = np.mean(epoch_loss)
        self.push_json([*self.cm.get_current_layer_struc()], self.log_dir + 'current_struc.json')

        if epoch > self.struc_early_stop and self.struc_learning:
            # learning place and state updated
            prob = self.update_prob()
            layer_learning_finished = self.cm.update_learning_places(prob=prob)
            if layer_learning_finished:
                self.struc_learning = self.cm.learning_state
                if not self.struc_learning:  # learning finish
                    return {
                        'steps': epoch,
                        'avg_loss': epoch_avg_loss,
                        'prob': prob.detach().cpu().clone().numpy()
                    }
                else:
                    pass

            torch_opt_struc = getattr(topt, self.opt_name_struc)
            v_struc_update_init = np.zeros([self.cm.current_num_placeholders, self.cm.num_ops], dtype=np.float32)

            self.var_struc = torch.tensor(v_struc_update_init, requires_grad=True, dtype=self.dtype, device=self.device)
            self.opt_struc = torch_opt_struc([self.var_struc], lr=self.lr_struc)
            self.tmp_ops_candidates = np.arange(self.cm.num_ops, dtype=np.float32)
            self.tmp_ops_candidates = list(np.tile(self.tmp_ops_candidates, (self.cm.current_num_placeholders, 1)))
            self.struc_early_stop += self.struc_early_stop

        if self.struc_learning:
            return {
                'steps': epoch,
                'avg_loss': epoch_avg_loss,
                'prob': prob.detach().cpu().clone().numpy(),
            }
        else:
            return {
                'steps': epoch,
                'avg_loss': epoch_avg_loss,
            }

    def plt_records(self, name, records):
        path = self.reprot_dir + name
        plt.scatter(list(range(len(records))), records, s=0.2)
        plt.title(name)
        plt.ylabel(name)
        plt.xlabel('epochs')
        plt.savefig(path)
        plt.close()

    def save_records(self, name, records):
        path = self.reprot_dir + name
        torch.save(torch.tensor(records), path + '.pt')
        self.plt_records(name, records)

    def learn(self, log_train_freq=-1, log_ckp_freq=-1,log_records_freq=-1):

        postfix_stats = {}
        records_steps = []
        records_avg_loss = []
        records_probs = []

        self.push_json(self.cm.ops, self.log_dir + 'operation_pool')

        def check_and_create_folder(path):
            if not os.path.exists(path):
                os.makedirs(path)

        folder_path = "C:/Users/yanzh/PycharmProjects/Unitary_decomposition/dqas/csv_files_for_matrix/dimension_qubit_3_second_version/"

        # 检查并创建new_operation_pool文件夹
        new_operation_pool_path = os.path.join(folder_path, "new_operation_pool_01")
        check_and_create_folder(new_operation_pool_path)

        # 更新self.log_dir为新的路径
        #self.log_dir = new_operation_pool_path
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")
        filename = f'epoch_loss_{timestamp}.csv'

        with open(os.path.join(self.log_dir, filename), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['steps', 'avg_loss'])

            for t in range(self.total_epochs):

                # train dqn
                train_report = self.epoch_train(t)

                postfix_stats['train/epoch_loss'] = train_report['avg_loss']
                postfix_stats['train/epochs'] = train_report['steps']
                postfix_stats['strucl'] = self.struc_learning

                records_steps.append(train_report['steps'])
                records_avg_loss.append(train_report['avg_loss'])
                if self.struc_learning:
                    records_probs.append(train_report['prob'].tolist())

                writer.writerow([t, train_report['avg_loss']])
################################################
                if self.logging and (t % log_train_freq == 0):
                    for key, item in train_report.items():
                        if key != "prob":
                            self.writer.add_scalar('train/' + key, item, t)

                if self.logging and (t % log_ckp_freq == 0):
                    torch.save(self.qdqn.state_dict()
                               , self.log_dir + 'epoch_{}.pt'.format(t))
                    if self.struc_learning:
                        torch.save(self.var_struc
                                   , self.log_dir + 'epoch_struc_{}.pt'.format(t))

                # records during training
                if t % log_records_freq == 0:
                    records = {'steps': records_steps ,
                               'avg_loss': records_avg_loss}

                    #print(f"save sturc1: {[*self.cm.get_learned_layer_struc()]}")
                    if self.struc_learning:
                        self.push_json(records_probs[-10:], self.log_dir + 'last_10_strucs_probs.json')
                        self.push_json([*self.cm.get_learned_layer_struc()],
                                       self.log_dir + 'learned_struc.json')
                    for key, value in records.items():
                        self.save_records(key, value)
                    states = torch.Tensor([0.] * self.cm.num_qubits)
                    self.push_json(
                        np.array2string(self.qdqn(states).detach().cpu().numpy(), precision=5,
                                        separator=',').replace(
                            'j,\n', '').replace('],\n', '\\n'), self.log_dir + 'current_matrix.json')

                if records_avg_loss[-1] <= self.min_loss:
                    output = f"problem solved at epoch {t}, with loss {records_avg_loss[-1]}"
                    if self.struc_learning:
                        output += "learning state {self.struc_learning}"
                    self.push_json(output, self.log_dir + 'problem_solved.json')
                    print(output)
                    break

            if self.logging and (log_ckp_freq > 0):
                torch.save(self.qdqn.state_dict()
                           , self.log_dir + 'episode_final.pt')
                if self.struc_learning:
                    torch.save(self.var_struc
                               , self.log_dir + 'episode_struc_final.pt')

            # final records
            records = {'steps': records_steps,
                       'avg_loss': records_avg_loss}

            self.push_json([*self.cm.get_learned_layer_struc()], self.log_dir + 'learned_struc.json')
            # print(f"save sturc2: {[*self.cm.get_learned_layer_struc()]}")
            if self.struc_learning:
                self.push_json(records_probs[-10:], self.log_dir + 'last_10_strucs_probs.json')
            for key, value in records.items():
                self.save_records(key, value)
