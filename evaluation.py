from load import Loader
from net import Model
from train import Trainer
import torch

import argparse
from tqdm import tqdm
import json
import pickle

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="machine setup")
    parser.add_argument("--machine",default="",help="machine")
    parser.add_argument("--device",default="cpu",help="device")
    parser.add_argument("--rank",default=1024,type=int, help="rank")
    parser.add_argument("--layer",default=16,type=int, help="layer")
    parser.add_argument("--lam",default=10,type=float, help="lambda")
    parser.add_argument("--model_name",default="mistral",type=str, help="layer")
    parser.add_argument("--lang",default="english",type=str, help="language")

    args = parser.parse_args()

    class Evaluator:
        def __init__(self, model_name, lam, layer, rank, lang="english", machine="", device="cpu"):

            self.model_name = model_name
            self.lam = lam
            self.layer = layer
            self.rank = rank
            self.lang = lang

            self.machine = machine
            self.device = device
            
            self.model = None
            self.trainer_class = None

        def evaluate(self):
            centroids, centroid_rels, mapping = self.trainer_class.extract_prototypical(only_head=False)
            acc, balanced_acc = self.trainer_class.evaluate_ang(centroids, centroid_rels, mapping, only_head=False,eval_set="test")
            las, uas ,uuas = self.trainer_class.evaluate_las_uuas(centroids, centroid_rels, mapping, only_head=False,eval_set="test")
            return dict({"las":las, "uas":uas, "uuas":uuas, "acc":acc, "balanced_acc":balanced_acc})
        
        def evaluate_layers(self):

            del self.model
            del self.trainer_class
            torch.cuda.empty_cache()

            results = {}
            probe_config = {"mode":"rank", "rank":self.rank}
            self.model = Model(self.model_name,probe_config=probe_config,device=self.device,machine=self.machine,untrained_model=False)
            self.trainer_class = Trainer(loader=None,model=self.model,layer=self.layer, lr=None, lam=None, beta=None)

            for layer in tqdm(range(16,17,1)):
                self.layer = layer 
                # self.model.probe.load_state_dict(torch.load(f'probe_params_noisy/probe_{self.model_name}_lam_{float(self.lam)}_beta_0.0_{self.layer}_{self.rank}.params',map_location=torch.device(args.device)))
                self.trainer_class = Trainer(loader=None,model=self.model,layer=args.layer,lr=None, lam=None, beta=None)
                results_layer = self.evaluate()
                print(results_layer)
                results[str(layer)] = results_layer

            # with open(f'probe_params/results_{self.model_name}_structural_layers.json', 'w') as json_file:
            #     json.dump(results, json_file)

        def evaluate_rank(self):
            results = {}

            for rank in tqdm([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]):
                self.rank = rank 
                probe_config = {"mode":"rank", "rank":self.rank}
                del self.model
                del self.trainer_class
                torch.cuda.empty_cache()

                self.model = Model(self.model_name,probe_config=probe_config,device=self.device,machine=self.machine,untrained_model=False)
                self.model.probe.load_state_dict(torch.load(f'probe_params_noisy/probe_{self.model_name}_lam_{float(self.lam)}_beta_0.0_16_{self.rank}.params',map_location=torch.device(args.device)))
                self.trainer_class = Trainer(loader=None,model=self.model,layer=args.layer,lr=None, lam=None, beta=None)

                results_rank = self.evaluate()
                results[str(rank)] = results_rank
                print(results_rank)

            with open(f'probe_params_noisy/results_{self.model_name}_rank.json', 'w') as json_file:
                json.dump(results, json_file)

        def evaluate_types(self):
            del self.model
            del self.trainer_class
            torch.cuda.empty_cache()
            
            probe_config = {"mode":"rank", "rank":1024}
            self.model = Model(self.model_name,probe_config=probe_config,device=self.device,machine=self.machine,untrained_model=False)
            self.model.probe.load_state_dict(torch.load(f'probe_params/probe_{self.model_name}_lam_{float(self.lam)}_beta_0.0_16_1024.params',map_location=torch.device(args.device)))

            self.trainer_class = Trainer(loader=None,model=self.model,layer=16, lr=None, lam=None, beta=None)
            test_loader = Loader("test",batch_size=1)
            all_edges = []
            all_rels = []
            results = {}

            for i, (batch_txt, batch_pdist, batch_G) in enumerate(test_loader.generate()):

                probed_h_layer = self.trainer_class._extract_h(batch_txt)
                edges, rels = self.trainer_class._extract_edges(probed_h_layer, batch_G, only_head=False)
                all_edges = all_edges + edges.detach().cpu().tolist()
                all_rels.extend(rels)

                if len(all_rels)>10000:
                    break

            unique_strings = list(set(all_rels))
            mapping = {string: i for i, string in enumerate(unique_strings)}
            all_rels = [mapping[string] for string in all_rels]

            results["types"] = all_rels
            results["edges"] = all_edges
            results["mapping"] = mapping

            with open(f'probe_params_noisy/results_{self.model_name}_types.json', 'w') as json_file:
                json.dump(results, json_file)
        
        def run_controlled(self):
            del self.trainer_class
            del self.model
            torch.cuda.empty_cache()

            probe_config = {"mode":"rank", "rank":self.rank}
            self.model = Model(self.model_name,probe_config=probe_config,device=self.device,machine=self.machine,untrained_model=False)
            self.model.probe.load_state_dict(torch.load(f'probe_params_noisy/probe_{self.model_name}_lam_{float(self.lam)}_beta_0.0_16_1024.params',map_location=torch.device(args.device)))

            self.trainer_class = Trainer(loader=None,model=self.model,layer=self.layer, lr=None, lam=None, beta=None)
            centroids, centroid_rels, mapping = self.trainer_class.extract_prototypical(only_head=False, total=10000)

            if self.lang=="english":
                dataset_name = "gpt-paper.txt"
            elif self.lang=="spanish":
                dataset_name = "gpt-paper-spanish.txt"
            elif self.lang=="controlled":
                dataset_name = "controlled_dataset_unique.txt"


            with open(dataset_name, "r") as file:
                sentences = file.readlines()
            
            all_sentences = []
            msts = []

            for sentence in sentences:
                sentence = sentence.replace("\n","")
                probed_h_layer = self.trainer_class._extract_h([sentence])
                sentence_embeddings = probed_h_layer[0].detach().cpu()
                mst = self._get_prediction(sentence_embeddings,centroids,centroid_rels,mapping)        
                all_sentences.append(sentence_embeddings.tolist())
                msts.append(mst.edges(data=True))

            results = {}
            results["dataset"] = all_sentences
            # results["msts"] = msts
            
            with open(f'probe_params_noisy/results_{self.model_name}_dataset_{self.lang}.json', 'w') as json_file:
                json.dump(results, json_file)

            with open(f'probe_params_noisy/results_{self.model_name}_dataset_mst_{self.lang}.pkl', 'wb') as f:
                pickle.dump(msts, f)
        
        def _get_prediction(self,h_probed_layer,centroids,centroid_rels,mapping):
            knn_model = self.trainer_class.construct_knn(centroids, centroid_rels, metric=self.trainer_class.abs_sim_dist)
            inverse_mapping = {value: key for key, value in mapping.items()}
            print(h_probed_layer.shape)
            h_pdist = self.trainer_class._calc_pdist(h_probed_layer)
            mst = self.trainer_class.compute_mst(h_pdist)
            print(h_pdist.shape)
            print(mst)

            h_edges = self.trainer_class._get_pred_h_edges(h_probed_layer, mst.edges(), only_head=False)
            rel_preds_nums = knn_model.predict(h_edges.cpu().detach().numpy())
            rel_preds = [inverse_mapping[rel_num] for rel_num in rel_preds_nums]

            mst = self.trainer_class.compute_labeled_mst(mst,rel_preds)
            mst = self.trainer_class.compute_directed_mst(mst,h_edges,centroids,rel_preds_nums, rel_preds)
            return mst


    evaluator = Evaluator(model_name=args.model_name,lam=args.lam,layer=args.layer,lang=args.lang,rank=args.rank,machine=args.machine,device=args.device)
    # evaluator.evaluate_layers()
    # evaluator.evaluate_rank()
    # evaluator.evaluate_types()
    evaluator.run_controlled()




# TRAINED PROBE (complete test)
# ACC:  0.8506994261119082  BALANCED ACC:  0.6666907089769746  ACC ROOT:  0.83475935828877  LAS:  67.96314673894912  UAS:  76.1421570597521  UUAS:  79.26106588795344

# TRAINED PROBE (test)
# ACC:  0.8576282423137482  BALANCED ACC:  0.7030857692932353  LAS:  70.10503035000396  UAS:  78.45959971238791  UUAS:  81.66811099012746    

# IDENTITY PROBE (test)
# ACC:  0.7519681776746499  BALANCED ACC:  0.7184076076784779  LAS:  30.95901631221193  UAS:  36.407806764453355  UUAS:  41.55315588598236

# RANDOM PROBE (test)
# ACC:  0.7275213391895251  BALANCED ACC:  0.6943608916366153  LAS:  29.39796237677201  UAS:  35.407365624914256  UUAS:  40.734269505097245

# ROOTED STRUCTURAL PROBE (test)
# ACC:  0.6076075246540151  BALANCED ACC:  0.5456633521688928  ACC ROOT:  0.8412073490813649  LAS:  51.14225310131065  UAS:  72.75480445832217  UUAS:  85.99573086332917

# RANDOM MODEL + IDENTITY PROBE
# ACC:  0.2927819673489683  BALANCED ACC:  0.2446838308366027  ACC ROOT:  0.19881889763779528  LAS:  5.7121571131544675  UAS:  20.529953421963  UUAS:  36.09820182584742




###################################################
# MULTIBERT CROSSED
# ACC:  0.4093809563271733  BALANCED ACC:  0.38195907768315646  ACC ROOT:  0.19619422572178477  LAS:  14.795851692936415  UAS:  28.614115443749956  UUAS:  41.26027662672765
    
# MULTIBERT TRAINED PROBE
# ACC:  0.8103091074832187  BALANCED ACC:  0.6627316366213509  ACC ROOT:  0.12335958005249344  LAS:  60.04802253127044  UAS:  70.43949778760913  UUAS:  77.5206122645163
    
# MULTIBERT RANDOM PROBE
# ACC:  0.6277450899146433  BALANCED ACC:  0.593597683380092  ACC ROOT:  0.11745406824146981  LAS:  24.885639522555675  UAS:  32.77614353468794  UUAS:  42.042001859313416
