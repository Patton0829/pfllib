import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import metrics
from utils.data_utils import read_client_data
from utils.dlg import DLG


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.few_shot = args.few_shot
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []
        self.rs_macro_f1 = []
        self.rs_per_class_acc = []
        self.latest_confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.latest_macro_f1 = 0.0
        self.latest_per_class_acc = np.zeros(self.num_classes, dtype=np.float64)

        self.times = times
        self.eval_gap = args.eval_gap
        self.print_gap = args.print_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.create_dataset('rs_macro_f1', data=np.asarray(self.rs_macro_f1, dtype=np.float64))
                hf.create_dataset('rs_per_class_acc', data=np.asarray(self.rs_per_class_acc, dtype=np.float64))
                hf.create_dataset('final_confusion_matrix', data=self.latest_confusion_matrix)
                hf.create_dataset('final_macro_f1', data=np.asarray([self.latest_macro_f1], dtype=np.float64))
                hf.create_dataset('final_per_class_acc', data=self.latest_per_class_acc)

            artifact_prefix = result_path + algo
            summary_csv_path = artifact_prefix + "_final_metrics.csv"
            with open(summary_csv_path, "w", encoding="utf-8") as f:
                f.write("metric,value\n")
                f.write(f"macro_f1,{self.latest_macro_f1:.6f}\n")
                for class_id, class_acc in enumerate(self.latest_per_class_acc):
                    f.write(f"per_class_accuracy_{class_id},{class_acc:.6f}\n")
                for row in range(self.num_classes):
                    for col in range(self.num_classes):
                        f.write(f"confusion_matrix_{row}_{col},{int(self.latest_confusion_matrix[row, col])}\n")
            self._save_confusion_matrix_plot(artifact_prefix + "_final_confusion_matrix.png")

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def _save_confusion_matrix_plot(self, file_path):
        row_totals = self.latest_confusion_matrix.sum(axis=1, keepdims=True)
        normalized_cm = np.divide(
            self.latest_confusion_matrix.astype(np.float64),
            row_totals,
            out=np.zeros_like(self.latest_confusion_matrix, dtype=np.float64),
            where=row_totals > 0,
        )

        fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
        im = ax.imshow(normalized_cm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title("Final Confusion Matrix (%)")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_xticks(np.arange(self.num_classes))
        ax.set_yticks(np.arange(self.num_classes))

        for row in range(self.num_classes):
            for col in range(self.num_classes):
                value = normalized_cm[row, col]
                text = f"{value * 100:.1f}%"
                text_color = "white" if value > 0.5 else "black"
                ax.text(col, row, text, ha="center", va="center", color=text_color, fontsize=9)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Percentage")
        fig.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        all_true = []
        all_pred = []
        for c in self.clients:
            try:
                result = c.test_metrics(return_detail=True)
            except TypeError:
                result = c.test_metrics()

            ct, ns, auc = result[:3]
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)
            if len(result) >= 5:
                all_true.append(np.asarray(result[3], dtype=np.int64))
                all_pred.append(np.asarray(result[4], dtype=np.int64))

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc, all_true, all_pred

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def should_print_round(self, round_idx):
        return self.print_gap > 0 and round_idx % self.print_gap == 0

    def _collect_evaluation_summary(self):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        all_true = np.concatenate(stats[4], axis=0) if len(stats) > 4 and len(stats[4]) > 0 else np.array([], dtype=np.int64)
        all_pred = np.concatenate(stats[5], axis=0) if len(stats) > 5 and len(stats[5]) > 0 else np.array([], dtype=np.int64)
        if all_true.size > 0 and all_pred.size > 0:
            labels = np.arange(self.num_classes)
            confusion_matrix = metrics.confusion_matrix(all_true, all_pred, labels=labels)
            class_totals = confusion_matrix.sum(axis=1)
            per_class_acc = np.divide(
                np.diag(confusion_matrix),
                class_totals,
                out=np.zeros(self.num_classes, dtype=np.float64),
                where=class_totals > 0,
            )
            macro_f1 = metrics.f1_score(all_true, all_pred, labels=labels, average='macro', zero_division=0)
        else:
            confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
            per_class_acc = np.zeros(self.num_classes, dtype=np.float64)
            macro_f1 = 0.0

        return {
            "stats": stats,
            "test_acc": test_acc,
            "test_auc": test_auc,
            "train_loss": train_loss,
            "accs": accs,
            "aucs": aucs,
            "macro_f1": float(macro_f1),
            "per_class_acc": per_class_acc.astype(np.float64),
            "confusion_matrix": confusion_matrix.astype(np.int64),
        }

    def _record_evaluation_summary(self, summary, record=True):
        self.latest_confusion_matrix = summary["confusion_matrix"]
        self.latest_macro_f1 = summary["macro_f1"]
        self.latest_per_class_acc = summary["per_class_acc"]

        if record:
            self.rs_test_acc.append(summary["test_acc"])
            self.rs_test_auc.append(summary["test_auc"])
            self.rs_train_loss.append(summary["train_loss"])
            self.rs_macro_f1.append(summary["macro_f1"])
            self.rs_per_class_acc.append(summary["per_class_acc"])

    def _print_evaluation_summary(self, summary):
        print("Averaged Train Loss: {:.4f}".format(summary["train_loss"]))
        print("Averaged Test Accuracy: {:.4f}".format(summary["test_acc"]))
        print("Averaged Test AUC: {:.4f}".format(summary["test_auc"]))
        print("Macro-F1: {:.4f}".format(summary["macro_f1"]))
        print("Per-class Accuracy: {}".format(np.array2string(summary["per_class_acc"], precision=4, separator=", ")))
        print("Std Test Accuracy: {:.4f}".format(np.std(summary["accs"])))
        print("Std Test AUC: {:.4f}".format(np.std(summary["aucs"])))

    def evaluate(self, acc=None, loss=None, verbose=True):
        summary = self._collect_evaluation_summary()
        self._record_evaluation_summary(summary, record=(acc is None and loss is None))

        if acc is not None:
            acc.append(summary["test_acc"])
        if loss is not None:
            loss.append(summary["train_loss"])

        if verbose:
            self._print_evaluation_summary(summary)

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
