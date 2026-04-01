import copy
import time

import numpy as np
import torch

from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server


class FedAvgAcc(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.acc_tau = args.acc_tau
        self.client_acc_map = {}

        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"Accuracy temperature (acc_tau): {self.acc_tau}")
        print("Finished creating server and clients.")

        self.Budget = []

    def _evaluate_and_cache_client_acc(self, verbose=True, record=True):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        self.client_acc_map = {
            client_id: (correct / num_samples)
            for client_id, correct, num_samples in zip(stats[0], stats[2], stats[1])
        }

        if record:
            self.rs_test_acc.append(test_acc)
            self.rs_train_loss.append(train_loss)

        if verbose:
            print("Averaged Train Loss: {:.4f}".format(train_loss))
            print("Averaged Test Accuracy: {:.4f}".format(test_acc))
            print("Averaged Test AUC: {:.4f}".format(test_auc))
            print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
            print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def _compute_accuracy_weights(self):
        raw_weights = []
        for client_id, base_weight in zip(self.uploaded_ids, self.uploaded_weights):
            client_acc = self.client_acc_map.get(client_id, 0.0)
            raw_weights.append(base_weight * float(torch.exp(torch.tensor(self.acc_tau * client_acc)).item()))

        weight_sum = sum(raw_weights)
        if weight_sum <= 0:
            return self.uploaded_weights

        return [weight / weight_sum for weight in raw_weights]

    def aggregate_parameters(self):
        assert len(self.uploaded_models) > 0

        adapted_weights = self._compute_accuracy_weights()

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for weight, client_model in zip(adapted_weights, self.uploaded_models):
            self.add_parameters(weight, client_model)

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            verbose = self.should_print_round(i)
            record = verbose
            if verbose:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
            self._evaluate_and_cache_client_acc(verbose=verbose, record=record)

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self._evaluate_and_cache_client_acc(verbose=True, record=True)
