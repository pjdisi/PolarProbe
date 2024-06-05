from load import Loader
from net import Model
from tqdm import tqdm
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch.nn as nn
import torch.optim as optim 
from torch.optim import lr_scheduler
from scipy.stats import spearmanr
from matplotlib import pyplot as plt
import wandb


# DATA: RAW, PREPROCESSING (Be able to recreate everything from raw)
# loss.py
# evaluation.py
# metrics.py
# plot.py
# stats.py (for brain projects)
# type hinting
# turn into pip installable, requirements.py
# pyproject.toml --> neuralset
# download data script, subprocesses check whether files exist
# black library -> formats scripts like commas and other non-standard things
# submitit -> avoid .sh files -> Submit Array (unifa more advanced)
# torch lightning -> standard structure
# dataclasses -> pydantic


class Trainer():
    def __init__(self, loader, model, layer, epoch_num=200, lr=0.0001, lam=0.5, beta=0.5):
        self.epoch_num = epoch_num
        self.lr = lr
        self.lam = lam
        self.beta = beta

        self.loader = loader
        self.model = model
        self.layer = layer
        
        self.device = self.model.device
        self.loss_l1 = nn.L1Loss()
        self.loss_mse = nn.MSELoss()

        # wandb.config.update({
        #     "learning_rate": self.lr,
        #     "epochs": self.epoch_num,
        #     "batch_size": self.loader.batch_size,
        #     "rank":self.model.probe.output_size,
        #     "model":self.model.model_name
        #     })
    
    def _calc_pdist(self,h_layer):
        L = len(h_layer)
        h_pdist = torch.zeros((L,L),device=self.device)
        for i in range(L):
            for j in range(i+1,L):
                square_euc = (h_layer[i] - h_layer[j]).square().sum()
                h_pdist[i,j] = square_euc
                h_pdist[j,i] = square_euc
        return h_pdist

    def _flatten_batch(self, batch_h, batch_pdist):
        accumulated_pdist = []
        accumulated_h_layer_pdist = []

        for h_probed_layer, pdist in zip(batch_h, batch_pdist):

            if len(h_probed_layer) != len(pdist):
                continue

            h_layer_pdist = self._calc_pdist(h_probed_layer)
            pdist = pdist.to(self.device)
            
            accumulated_pdist.append(pdist.flatten())
            accumulated_h_layer_pdist.append(h_layer_pdist.flatten())

        return accumulated_h_layer_pdist, accumulated_pdist

    def _extract_edges(self,batch_h,batch_G, only_head=False):
        edges = []
        rels = []
        for h_probed_layer, G in zip(batch_h, batch_G):
            N = G.number_of_nodes()

            if len(h_probed_layer) != N:
                continue

            for head, dep, attrs in G.edges(data=True):
                head, dep = head-1, dep-1
                rel = attrs.get('rel_type', None)
                if not only_head:
                    edge = h_probed_layer[head] - h_probed_layer[dep]
                else:
                    edge = h_probed_layer[head]

                edges.append(edge)
                rels.append(rel)

        edges = torch.stack(edges).to(self.device)
        return edges, rels

    def _extract_h(self,batch_txt):
        h_layer = []
        for txt in batch_txt:
            h_probed = self.model(txt)
            h_probed_layer = h_probed[self.layer]
            h_layer.append(h_probed_layer)
        return h_layer
    
    def _get_pred_h_edges(self, h_layer, edges, only_head=False):
        h_edges = []
        for u, v in edges:
            if not only_head:
                h_edge = h_layer[u-1] - h_layer[v-1]
            else:
                norm_u = torch.linalg.norm(h_layer[u-1]-self.model.probe.root_learned.weight)
                norm_v = torch.linalg.norm(h_layer[v-1]-self.model.probe.root_learned.weight)
                h_edge = h_layer[u-1] if norm_u < norm_v else h_layer[v-1]
            h_edges.append(h_edge)
        
        h_edges = torch.stack(h_edges).to(self.device)
        return h_edges

    def _find_batch_roots(self, batch_G):
        root_nodes = []
        for G in batch_G:
            root_node = [n for n, d in G.nodes(data=True) if 'root' in d and d['root']]
            root_nodes.append(root_node[0])
        return root_nodes
    
    def root_loss(self,probed_h_layer, batch_G):
        root_h = []
        root_target = []
        root_nodes = self._find_batch_roots(batch_G)

        for h_txt, root in zip(probed_h_layer, root_nodes):
            root_h.append(h_txt[root-1])
            root_target.append(self.model.probe.root_learned.weight)
        
        root_h = torch.stack(root_h).to(self.device)
        root_target = torch.stack(root_target).to(self.device)[:,0,:]

        root_loss = self.loss_mse(root_h, root_target)
        return root_loss
    
    def abs_sim_dist(self, x, y):
        sim = cosine(x,y) -1 
        dist = 1 - np.abs(sim)
        return dist

    def compute_mst(self, h_pdist):
        G = nx.Graph()
        num_vertices = len(h_pdist)
        for i in range(num_vertices):
            for j in range(i+1, num_vertices):
                G.add_edge(i+1, j+1, weight=h_pdist[i,j])

        mst = nx.minimum_spanning_tree(G)
        return mst
    
    def compute_labeled_mst(self, mst, rel_preds):
        for i, edge in enumerate(mst.edges()):
            mst.edges[edge]["rel_type"] = rel_preds[i]
        return mst
    
    def compute_directed_mst(self, mst, h_edges, centroids, rel_preds_nums, rel_preds):
        h_edges = h_edges.cpu().detach().numpy()
        centroids = centroids.cpu().detach().numpy()

        directed_mst = nx.DiGraph()

        for edge, h_edge, centroid_num, centroid_name in zip(mst.edges(),h_edges,rel_preds_nums,rel_preds):
            u, v = edge
            cos_sim = 1-cosine(h_edge,centroids[centroid_num])

            if cos_sim>0:
                directed_mst.add_edge(u, v, rel_type=centroid_name)
            else:
                directed_mst.add_edge(v, u, rel_type=centroid_name)
        return directed_mst

    def distance_loss(self, preds, golds):
        loss = torch.tensor(0.0,device=self.device)
        for pred, gold in zip(preds,golds):
            loss += self.loss_l1(pred,gold)
        loss = loss/len(preds)
        return loss

    def angular_loss(self, edges, rels):
        # print(dict(Counter(rels)))
        string_to_int_mapping = {string: index for index, string in enumerate(set(rels))}
        rels = torch.tensor([string_to_int_mapping[string] for string in rels]).to(self.device)

        sorted_indices = torch.argsort(rels)
        rels = rels[sorted_indices]
        edges = edges[sorted_indices]

        norm_edges = F.normalize(edges, p=2, dim=1)
        cos_sim_mat = torch.mm(norm_edges, norm_edges.t())

        cos_dist_mat = 1 - cos_sim_mat
        cos_dist_mat = torch.clamp(cos_dist_mat, min=0, max=2)

        gold_mat = torch.ones_like(cos_dist_mat).to(self.device)
        unique_rels = set(rels)
        for rel in unique_rels:
            indices_block = torch.where(rels == rel)[0]
            gold_mat[indices_block[:, None], indices_block] = 0

        # self.plot_ang(cos_dist_mat,gold_mat,rels)
                        
        loss = self.loss_mse(cos_dist_mat,gold_mat)
        return loss

    def plot_ang(self, cos_dist_mat, gold_mat, rels):
        lines = torch.diff(rels)!=0
        lines = torch.cat((torch.tensor([0],device=self.device), lines))

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(cos_dist_mat.detach())
        for i, line_value in enumerate(lines):
            if line_value:
                axes[0].axvline(x=i, color='r', linestyle='--',lw=0.1)
                axes[0].axhline(y=i, color='r', linestyle='--',lw=0.1)
        axes[0].set_title('Pairwise matrix')

        axes[1].imshow(gold_mat.detach())
        axes[1].set_title('Gold matrix')
        fig.savefig("angular.png")
        plt.close()

    def evaluate_dist(self, eval_set="dev"):
        dev_loader = Loader(eval_set,batch_size=1,skip_long=False)
        spr_dist = []
        
        for txt, pdist, G in dev_loader.generate():
            h_probed_layer = self._extract_h(txt)
            preds, golds = self._flatten_batch(h_probed_layer, pdist)
            for pred, gold in zip(preds,golds):
                correlation_coefficient, p_value = spearmanr(pred.detach().cpu().numpy(), gold.detach().cpu().numpy())
                spr_dist.append(correlation_coefficient)
        spr_dist = np.mean(spr_dist)
        return spr_dist

    def evaluate_ang(self, centroids, centroid_rels, mapping, only_head=False, eval_set="dev"):
        dev_loader = Loader(eval_set,batch_size=1,skip_long=False)
        all_edges = []
        all_rels = []
        
        for txt, pdist, G in dev_loader.generate():
            h_probed_layer = self._extract_h(txt)
            edges, rels = self._extract_edges(h_probed_layer, G, only_head=only_head)

            all_edges.append(edges.detach().cpu())
            all_rels.extend(rels)

        all_edges = torch.vstack(all_edges)
        knn_model = self.construct_knn(centroids, centroid_rels)

        acc_balanced = self.evaluate_knn(knn_model, all_edges, all_rels, mapping, balanced_accuracy_score)
        acc = self.evaluate_knn(knn_model, all_edges, all_rels, mapping, accuracy_score)

        return acc, acc_balanced
    
    def evaluate_las_uuas(self, centroids, centroid_rels, mapping, only_head=False, eval_set="dev"):

        dev_loader = Loader(eval_set,batch_size=1,skip_long=False)

        inverse_mapping = {value: key for key, value in mapping.items()}
        knn_model = self.construct_knn(centroids, centroid_rels, metric=self.abs_sim_dist)

        las_list = []
        uuas_list = []
        uas_list = []

        for txt, pdist, G in dev_loader.generate():
            h_probed_layer = self._extract_h(txt)[0]
            h_pdist = self._calc_pdist(h_probed_layer)

            mst = self.compute_mst(h_pdist)

            h_edges = self._get_pred_h_edges(h_probed_layer, mst.edges(), only_head=only_head)
            rel_preds_nums = knn_model.predict(h_edges.cpu().detach().numpy())
            rel_preds = [inverse_mapping[rel_num] for rel_num in rel_preds_nums]

            mst = self.compute_labeled_mst(mst,rel_preds)
            mst = self.compute_directed_mst(mst,h_edges,centroids,rel_preds_nums, rel_preds)
            

            gold_edges_labeled = {(u, v, d['rel_type']) for u, v, d in G[0].edges(data=True)}
            predicted_edges_labeled = {(u, v, d['rel_type']) for u, v, d in mst.edges(data=True)}

            correct_labeled_dependencies = gold_edges_labeled.intersection(predicted_edges_labeled)
            las_value = (len(correct_labeled_dependencies) / len(gold_edges_labeled)) * 100
            las_list.append(las_value)

            gold_edges_directed = {(u, v) for u, v, d in G[0].edges(data=True)}
            predicted_edges_directed = {(u, v) for u, v, d in mst.edges(data=True)}
            correct_directed_dependencies = gold_edges_directed.intersection(predicted_edges_directed)
            uas_value = (len(correct_directed_dependencies) / len(gold_edges_directed)) * 100
            uas_list.append(uas_value)

            gold_edges = {(min(u, v), max(u, v)) for u, v in G[0].edges(data=False)}
            predicted_edges = {(min(u, v), max(u, v)) for u, v in mst.edges(data=False)}
            correct_unlabeled_dependencies = gold_edges.intersection(predicted_edges)
            uuas_value = (len(correct_unlabeled_dependencies) / len(gold_edges)) * 100
            uuas_list.append(uuas_value)


        las = np.mean(las_list)
        uas = np.mean(uas_list)
        uuas = np.mean(uuas_list)

        return las, uas, uuas
    
    def evaluate_root(self, eval_set="dev"):
        dev_loader = Loader(eval_set,batch_size=1,skip_long=False)
        root_emb = self.model.probe.root_learned.weight

        pred_roots = []
        gold_roots = []
        
        for txt, pdist, G in dev_loader.generate():
            h_probed_layer = self._extract_h(txt)[0]

            distances = torch.norm(h_probed_layer - root_emb, dim=1)

            pred_root = torch.argmin(distances).item()
            gold_root = self._find_batch_roots(G)[0] - 1

            pred_roots.append(pred_root)
            gold_roots.append(gold_root)

        root_acc = accuracy_score(pred_roots,gold_roots)

        return root_acc
        
    def extract_prototypical(self, only_head=False, total=5000):
        proto_loader = Loader("train",batch_size=1,skip_long=False)

        all_edges = []
        all_rels = []
        for i, (batch_txt, batch_pdist, batch_G) in enumerate(proto_loader.generate()):

            probed_h_layer = self._extract_h(batch_txt)
            edges, rels = self._extract_edges(probed_h_layer, batch_G, only_head=only_head)

            all_edges.append(edges.detach().cpu())
            all_rels.extend(rels)

            if i==total:
                break

        all_edges = torch.vstack(all_edges)#.to(self.device)

        unique_strings = list(set(all_rels))
        mapping = {string: i for i, string in enumerate(unique_strings)}

        all_rels = [mapping[string] for string in all_rels]
        all_rels = torch.tensor(all_rels)#.to(self.device)

        centroids = []
        centroid_rels = []

        unique_rels = torch.unique(all_rels)
        for unique_rel in unique_rels:
            rel_indexes = all_rels==unique_rel
            
            ########## NEW DURING LAST FIGURE DEVELOPMENT
            # number_of_rels = len(rel_indexes)
            # if (number_of_rels/len(all_rels))<0.05:
            #     continue
            ########## NEW DURING LAST FIGURE DEVELOPMENT  

            centroid = all_edges[rel_indexes].mean(0)

            centroids.append(centroid.detach().cpu())
            centroid_rels.append(unique_rel.detach().cpu())
        
        centroids = torch.vstack(centroids)#.to(self.device)
        centroid_rels = torch.tensor(centroid_rels)#.to(self.device)

        return centroids, centroid_rels, mapping

    def construct_knn(self, X_cen, y_cen, metric="cosine"):
        X_cen = X_cen.detach().cpu().numpy()
        y_cen = y_cen.detach().cpu().numpy()

        knn_model = KNeighborsClassifier(n_neighbors=1, metric=metric)
        knn_model.fit(X_cen, y_cen)

        return knn_model
    
    def evaluate_knn(self, knn_model, X, y, mapping, metric):
        y = [mapping.get(string, -1) for string in y]
        y = torch.tensor(y)

        X = X.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        y_pred = knn_model.predict(X)
        acc = metric(y, y_pred)
        return acc

    def train(self):

        optimizer = optim.Adam(self.model.probe.parameters(), lr=self.lr) 
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        print("SETUP: LR:",self.lr,"     LAM:",self.lam,"     BETA:",self.beta,"     RANK:",self.model.probe.output_size, "     MODEL:",self.model.model_name, "     LAYER:",self.layer)
        print("WANDB FILENAME: ", wandb.run.dir)

        best_score = 0
        best_probe = None

        for epoch in range(self.epoch_num):
            print("Epoch: ", epoch)

            for batch_txt, batch_pdist, batch_G in tqdm(self.loader.generate()):
                optimizer.zero_grad()

                probed_h_layer = self._extract_h(batch_txt)
                
                edges, rels = self._extract_edges(probed_h_layer,batch_G, only_head=False)
                ang_loss = self.angular_loss(edges, rels)

                preds_dist, golds_dist = self._flatten_batch(probed_h_layer, batch_pdist)
                dist_loss = self.distance_loss(preds_dist,golds_dist)
                root_loss = self.root_loss(probed_h_layer, batch_G)

                loss = dist_loss + self.beta*root_loss + self.lam*ang_loss
                print("LOSS: ", ang_loss.item(), dist_loss.item(), root_loss.item())
                wandb.log({"dist_loss": dist_loss, "ang_loss": ang_loss, "root_loss":root_loss, "loss":loss})

                # loss = ang_loss
                # print("LOSS: ", ang_loss.item())
                # wandb.log({"ang_loss": ang_loss})
                
                loss.backward()
                optimizer.step()
            
            spr_dist = self.evaluate_dist()

            centroids, centroid_rels, mapping = self.extract_prototypical(only_head=False, total=5000)
            acc, balanced_acc = self.evaluate_ang(centroids, centroid_rels, mapping, only_head=False)
            las, uas, uuas = self.evaluate_las_uuas(centroids, centroid_rels, mapping, only_head=False)

            model_score = las

            if model_score > best_score:
                best_score = model_score
                best_probe = self.model.probe
                best_probe.save_probe(f"angular_probe_{self.model.model_name}_lam_{self.lam}_beta_{self.beta}_{self.layer}_{self.model.probe.output_size}.params")

            wandb.log({"spr_dist": spr_dist, "balanced_acc": balanced_acc, "acc":acc, "las":las, "uuas": uuas, "uas":uas})
            print("EVALUATION:\n Spr: ", spr_dist, " Balanced Acc: ", balanced_acc, "Acc: ",acc, "LAS: ",las,"UAS: ",uas, "UUAS: ", uuas)
            # scheduler.step()
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="machine setup")
    parser.add_argument("--machine",default="local",help="machine")
    parser.add_argument("--device",default="cpu",help="device")
    parser.add_argument("--lam",default=0.01,type=float, help="lambda")
    parser.add_argument("--beta",default=0.01,type=float, help="beta")
    parser.add_argument("--lr",default=0.001,type=float, help="learning rate")
    parser.add_argument("--rank",default=1024,type=int, help="rank")
    parser.add_argument("--layer",default=16,type=int, help="layer")
    parser.add_argument("--model",default="bert",type=str, help="model")

    args = parser.parse_args()

    if args.machine=="jz":
        mode = "offline"
    else:
        mode = "online"        

    wandb.init(project='Polar Probe', entity='itspablito',config={}, mode=mode)

    probe_config ={

        "mode":"rank",
        "rank": args.rank
    }

    loader = Loader("train",batch_size=200,skip_long=False)
    model = Model(args.model,probe_config=probe_config,device=args.device,machine=args.machine)
    trainer = Trainer(loader=loader,model=model,layer=args.layer,lr=args.lr, lam=args.lam, beta=args.beta)
    trainer.train()