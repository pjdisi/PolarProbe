from torch import nn
import torch
from transformers import BertModel, RobertaModel, GPT2Model, AutoTokenizer, BertTokenizer, RobertaTokenizer, BertConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoConfig
from collections import defaultdict
from torch.nn.parallel import DataParallel



class PolarProbe(nn.Module):
    def __init__(self, input_size, output_size, device="cpu"):
        super(PolarProbe, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.device = device

        self.root_learned = nn.Linear(self.output_size, 1, bias=False, device=self.device)
        self.linear = nn.Linear(self.input_size, self.output_size, bias=False, device=self.device)
        nn.init.uniform_(self.linear.weight, -0.00001, 0.00001)

    def forward(self, x):
        probed = self.linear(x)
        return probed
    
    def save_probe(self, path):
        torch.save(self.state_dict(), path)

    def make_identity(self):
        assert self.input_size == self.input_size
        identity_matrix = torch.eye(self.linear.in_features, device=self.device)
        self.linear.weight.data = identity_matrix
        return self

class Model(nn.Module):
    def __init__(self,model_name,probe_config,device="cpu",machine="",untrained_model=False):
        super(Model, self).__init__()
        self.model_name = model_name
        self.probe_config = probe_config
        self.untrained_model = untrained_model
        self.device = device
        if machine=="jz":
            self.root = "/gpfsdswork/dataset/HuggingFace_Models/"
        else:
            self.root = ""

        self.initialize_llm()
        self.initialize_probe()

        # self.llm = self.llm.to(self.device).eval() DELETED
        self.freeze_llm()
    
    def freeze_llm(self):
        for param in self.llm.parameters():
            param.requires_grad = False

    def initialize_llm(self):
        if self.model_name=="bert":
            # self.tokenizer = BertTokenizer.from_pretrained(self.root+'bert-large-cased')
            self.tokenizer = AutoTokenizer.from_pretrained(self.root+'bert-large-cased',use_fast=True)
            if self.untrained_model:
                config = BertConfig.from_pretrained('bert-large-cased')
                self.llm = BertModel(config).to(self.device).eval()
            else:
                self.llm = AutoModel.from_pretrained(self.root+'bert-large-cased').to(self.device).eval()
            self.llm_dim = 1024

        elif self.model_name=="gpt2":
            self.tokenizer = AutoTokenizer.from_pretrained(self.root+'gpt2-medium',use_fast=True)
            self.llm = AutoModelForCausalLM.from_pretrained(self.root+'gpt2-medium').to(self.device).eval()
            self.llm_dim = 1024

        elif self.model_name=="roberta":
            self.tokenizer = AutoTokenizer.from_pretrained(self.root+'roberta-large',use_fast=True)
            self.llm = AutoModel.from_pretrained(self.root+'roberta-large').to(self.device).eval()
            self.llm_dim = 1024

        elif self.model_name=="mistral":
            self.tokenizer = AutoTokenizer.from_pretrained(self.root+'mistralai/Mistral-7B-v0.1')
            self.llm = AutoModelForCausalLM.from_pretrained(self.root+'mistralai/Mistral-7B-v0.1').to(self.device).eval()
            self.llm_dim = 4096
        
        elif self.model_name=="llama":
            self.tokenizer = AutoTokenizer.from_pretrained(self.root+'meta-llama/Llama-2-7b-hf')
            self.llm = AutoModelForCausalLM.from_pretrained(self.root+'meta-llama/Llama-2-7b-hf').to(self.device).eval()
            self.llm_dim = 4096
        
        elif self.model_name=="mistral_untrained":
            self.tokenizer = AutoTokenizer.from_pretrained(self.root+'mistralai/Mistral-7B-v0.1')
            config = AutoConfig.from_pretrained(self.root+'mistralai/Mistral-7B-v0.1')
            self.llm = AutoModel.from_config(config).to(self.device).eval()
            self.llm_dim = 4096
        
        elif self.model_name=="llama_untrained":
            self.tokenizer = AutoTokenizer.from_pretrained(self.root+'meta-llama/Llama-2-7b-hf')
            config = AutoConfig.from_pretrained(self.root+'meta-llama/Llama-2-7b-hf')
            self.llm = AutoModel.from_config(config).to(self.device).eval()
            self.llm_dim = 4096
        
        elif "bert_" in self.model_name:
            seed = str(self.model_name.split("_")[1])
            self.tokenizer = BertTokenizer.from_pretrained('models/multiberts-seed_'+seed)
            self.llm = BertModel.from_pretrained("models/multiberts-seed_"+seed)
            self.llm_dim = 768
    
        else:
            raise Exception("Model not recognized")
        
    def initialize_probe(self):
        self.mode = self.probe_config["mode"]
        self.rank = self.probe_config["rank"]

        if self.mode=="identity":
            probe = PolarProbe(self.llm_dim,self.llm_dim,device=self.device)
            self.probe = probe.make_identity()
        elif self.mode=="full":
            self.probe = PolarProbe(self.llm_dim,self.llm_dim,device=self.device)
        elif self.mode=="rank":
            self.probe = PolarProbe(self.llm_dim,self.rank,device=self.device)
        else:
            raise NameError
    
    def forward(self,txt):
        untok_sent = txt.split(" ")
        # tok_sent = self.tokenizer.wordpiece_tokenizer.tokenize("[CLS] " + txt + " [SEP]")

        enc = self.tokenizer(txt, add_special_tokens=True)
        tok_sent = self.tokenizer.convert_ids_to_tokens(enc.input_ids)
        input_ids = self.tokenizer.convert_tokens_to_ids(tok_sent)

        input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.device)
        segment_ids = torch.tensor([1 for x in tok_sent]).to(self.device)

        with torch.no_grad():
            # outputs = self.llm(input_ids=input_ids,attention_mask=segment_ids, output_hidden_states=True)
            outputs = self.llm(input_ids=input_ids, output_hidden_states=True)
        
        probed_activations = torch.cat(outputs.hidden_states,dim=0)
        assert probed_activations.shape[1] == len(tok_sent)

        mapping = self._get_mapping(tok_sent,untok_sent)
        avg_activations = self._avg_subtokens(probed_activations, mapping)

        probed_activations = self.probe(avg_activations)

        assert probed_activations.shape[1] == len(untok_sent), print(probed_activations.shape[1], len(untok_sent), len(list(mapping.keys())))
        assert not probed_activations.isnan().any()

        return probed_activations
  
    def _get_mapping(self, tokenized_sent, untokenized_sent):

        mapping = defaultdict(list)

        if self.model_name=="bert":
            untokenized_sent_index = 0
            tokenized_sent_index = 1
            while untokenized_sent_index < len(untokenized_sent) and tokenized_sent_index < len(tokenized_sent):
                while tokenized_sent_index + 1 < len(tokenized_sent) and tokenized_sent[tokenized_sent_index + 1].startswith("##"):
                    mapping[untokenized_sent_index].append(tokenized_sent_index)
                    tokenized_sent_index += 1
                mapping[untokenized_sent_index].append(tokenized_sent_index)
                untokenized_sent_index += 1
                tokenized_sent_index += 1
        
        elif self.model_name=="roberta":
            untokenized_sent_index = 0
            tokenized_sent_index = 1
            while untokenized_sent_index < len(untokenized_sent) and tokenized_sent_index < len(tokenized_sent):
                while tokenized_sent_index + 1 < len(tokenized_sent) and not tokenized_sent[tokenized_sent_index + 1].startswith("Ġ"):
                    mapping[untokenized_sent_index].append(tokenized_sent_index)
                    tokenized_sent_index += 1
                mapping[untokenized_sent_index].append(tokenized_sent_index)
                untokenized_sent_index += 1
                tokenized_sent_index += 1

        elif self.model_name=="gpt2":
            untokenized_sent_index = 0
            tokenized_sent_index = 0
            while untokenized_sent_index < len(untokenized_sent) and tokenized_sent_index < len(tokenized_sent):
                while tokenized_sent_index + 1 < len(tokenized_sent) and not tokenized_sent[tokenized_sent_index + 1].startswith("Ġ"):
                    mapping[untokenized_sent_index].append(tokenized_sent_index)
                    tokenized_sent_index += 1
                mapping[untokenized_sent_index].append(tokenized_sent_index)
                untokenized_sent_index += 1
                tokenized_sent_index += 1

        elif self.model_name in ["mistral", "llama", "mistral_untrained", "llama_untrained"]:
            untokenized_sent_index = 0
            tokenized_sent_index = 1
            while untokenized_sent_index < len(untokenized_sent) and tokenized_sent_index < len(tokenized_sent):
                while tokenized_sent_index + 1 < len(tokenized_sent) and not (ord(tokenized_sent[tokenized_sent_index + 1][0]) == 9601):
                    mapping[untokenized_sent_index].append(tokenized_sent_index)
                    tokenized_sent_index += 1
                mapping[untokenized_sent_index].append(tokenized_sent_index)
                untokenized_sent_index += 1
                tokenized_sent_index += 1
        else:
            raise NotImplemented
        
        return mapping

    def _avg_subtokens(self, activations, mapping):
        averaged = []
        for i in range(len(mapping)):
            start_idx = mapping[i][0]
            end_idx = mapping[i][-1]+1

            avg_token = activations[:,start_idx:end_idx,:].mean(1)
            averaged.append(avg_token)
        
        averaged = torch.stack(averaged, dim=1)
        return averaged