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
        self.acc_ema_rho = args.acc_ema_rho
        self.acc_gamma_max = args.acc_gamma_max
        self.client_acc_map = {}
        self.client_relative_acc_ema_map = {}
        self.current_round = 0

        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"Accuracy temperature (acc_tau): {self.acc_tau}")
        print(f"Relative-accuracy EMA rho: {self.acc_ema_rho}")
        print(f"Relative-accuracy gamma max: {self.acc_gamma_max}")
        print("Accuracy term: relative accuracy + EMA smoothing + progressive warm-up")
        print("Finished creating server and clients.")

        self.Budget = []

    def _evaluate_and_cache_client_acc(self, verbose=True, record=True):
        summary = self._collect_evaluation_summary()
        stats = summary["stats"]

        self.client_acc_map = raw_acc_map = {
            client_id: (correct / num_samples)
            for client_id, correct, num_samples in zip(stats[0], stats[2], stats[1])
        }
        if raw_acc_map:
            mean_acc = float(np.mean(list(raw_acc_map.values())))
            for client_id, raw_acc in raw_acc_map.items():
                relative_acc = raw_acc - mean_acc
                previous = self.client_relative_acc_ema_map.get(client_id, relative_acc)
                smoothed = self.acc_ema_rho * previous + (1.0 - self.acc_ema_rho) * relative_acc
                self.client_relative_acc_ema_map[client_id] = smoothed

        self._record_evaluation_summary(summary, record=record)

        if verbose:
            self._print_evaluation_summary(summary)

    def _accuracy_gamma(self):
        if self.global_rounds <= 0:
            return self.acc_gamma_max
        progress = min(max(self.current_round / float(self.global_rounds), 0.0), 1.0)
        return self.acc_gamma_max * progress

    def _compute_accuracy_weights(self):
        gamma_t = self._accuracy_gamma()
        raw_weights = []
        for client_id, base_weight in zip(self.uploaded_ids, self.uploaded_weights):
            client_acc = self.client_relative_acc_ema_map.get(client_id, 0.0)
            exponent = gamma_t * self.acc_tau * client_acc
            raw_weights.append(base_weight * float(torch.exp(torch.tensor(exponent)).item()))

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
            self.current_round = i
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
