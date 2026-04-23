import copy
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch

from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server


class DHCWFL(Server):
    """Domain-Grouped Historical Consensus Weighting Federated Learning."""

    def __init__(self, args, times):
        super().__init__(args, times)

        self.history_lambda = args.dhcw_history_lambda
        self.group_tau = args.dhcw_group_tau
        self.domain_eta = args.dhcw_domain_eta
        self.eps = 1e-12
        self.consistency_history = {}
        self.client_domains = self._load_client_domains()
        self.domain_to_client_ids = self._build_domain_to_client_ids()

        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"DHCW history lambda: {self.history_lambda}")
        print(f"DHCW group tau: {self.group_tau}")
        print(f"DHCW domain eta: {self.domain_eta}")
        print(f"Client domains: {self.domain_to_client_ids}")
        print("Aggregation: domain-grouped historical consensus weighting")
        print("Finished creating server and clients.")

        self.Budget = []

    def _load_client_domains(self):
        config_path = os.path.join("..", "dataset", self.dataset, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            domains = config.get("client_domains")
            if domains is not None and len(domains) == self.num_clients:
                return [str(domain) for domain in domains]

        if self.dataset.lower() == "jnu_cwru_mix" and self.num_clients % 2 == 0:
            half = self.num_clients // 2
            return ["jnu"] * half + ["cwru"] * (self.num_clients - half)

        return ["global"] * self.num_clients

    def _build_domain_to_client_ids(self):
        domain_to_ids = defaultdict(list)
        for cid, domain in enumerate(self.client_domains):
            domain_to_ids[domain].append(cid)
        return dict(domain_to_ids)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(
                range(self.num_join_clients, self.num_clients + 1), 1, replace=False
            )[0]
        else:
            self.current_num_join_clients = self.num_join_clients

        if len(self.domain_to_client_ids) <= 1:
            return super().select_clients()

        selected_ids = []
        domains = list(self.domain_to_client_ids.keys())
        remaining = self.current_num_join_clients
        remaining_domains = len(domains)

        for domain in domains:
            candidate_ids = self.domain_to_client_ids[domain]
            target = int(round(self.current_num_join_clients * len(candidate_ids) / self.num_clients))
            target = max(1, target)
            target = min(target, len(candidate_ids), remaining - (remaining_domains - 1))
            selected_ids.extend(random.sample(candidate_ids, target))
            remaining -= target
            remaining_domains -= 1

        if remaining > 0:
            unused_ids = [cid for cid in range(self.num_clients) if cid not in selected_ids]
            selected_ids.extend(random.sample(unused_ids, min(remaining, len(unused_ids))))

        return [self.clients[cid] for cid in selected_ids]

    def _model_delta_chunks(self, client_model):
        chunks = []
        for global_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            chunks.append((client_param.data - global_param.data).clone())
        return chunks

    def _flatten_chunks(self, chunks):
        return torch.cat([chunk.reshape(-1) for chunk in chunks])

    def _softmax_from_scores(self, scores, temperature):
        if len(scores) == 0:
            return []
        score_tensor = torch.tensor(scores, dtype=torch.float64)
        score_tensor = temperature * score_tensor
        score_tensor = score_tensor - torch.max(score_tensor)
        weights = torch.exp(score_tensor)
        weights = weights / torch.sum(weights)
        return [float(weight.item()) for weight in weights]

    def aggregate_parameters(self):
        assert len(self.uploaded_models) > 0

        uploaded_by_domain = defaultdict(list)
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            uploaded_by_domain[self.client_domains[cid]].append((cid, client_model))

        group_updates = []
        group_scores = []

        for domain, domain_items in uploaded_by_domain.items():
            delta_chunks_list = []
            unit_vectors = []
            client_ids = []

            for cid, client_model in domain_items:
                chunks = self._model_delta_chunks(client_model)
                delta_vec = self._flatten_chunks(chunks)
                unit_vec = delta_vec / (torch.norm(delta_vec) + self.eps)
                delta_chunks_list.append(chunks)
                unit_vectors.append(unit_vec)
                client_ids.append(cid)

            consensus = torch.zeros_like(unit_vectors[0])
            for unit_vec in unit_vectors:
                consensus += unit_vec
            consensus = consensus / (torch.norm(consensus) + self.eps)

            sim_scores = [float(torch.dot(unit_vec, consensus).item()) for unit_vec in unit_vectors]
            for cid, sim_score in zip(client_ids, sim_scores):
                if cid not in self.consistency_history:
                    self.consistency_history[cid] = sim_score
                else:
                    self.consistency_history[cid] = (
                        self.history_lambda * self.consistency_history[cid]
                        + (1.0 - self.history_lambda) * sim_score
                    )

            history_scores = [self.consistency_history[cid] for cid in client_ids]
            client_weights = self._softmax_from_scores(history_scores, self.group_tau)

            group_delta = []
            for template_chunk in delta_chunks_list[0]:
                group_delta.append(torch.zeros_like(template_chunk))

            for weight, chunks in zip(client_weights, delta_chunks_list):
                for group_chunk, client_chunk in zip(group_delta, chunks):
                    group_chunk.data += client_chunk.data.clone() * weight

            group_updates.append(group_delta)
            group_scores.append(sum(sim_scores) / len(sim_scores))

        domain_weights = self._softmax_from_scores(group_scores, self.domain_eta)

        global_delta = []
        for param in self.global_model.parameters():
            global_delta.append(torch.zeros_like(param.data))

        for domain_weight, group_delta in zip(domain_weights, group_updates):
            for global_chunk, group_chunk in zip(global_delta, group_delta):
                global_chunk.data += group_chunk.data.clone() * domain_weight

        new_global_model = copy.deepcopy(self.global_model)
        for param, delta in zip(new_global_model.parameters(), global_delta):
            param.data += delta.data.clone()
        self.global_model = new_global_model

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate(verbose=self.should_print_round(i))

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
            self.evaluate()
