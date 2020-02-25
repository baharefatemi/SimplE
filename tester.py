import torch
from dataset import Dataset
import numpy as np
from measure import Measure
from os import listdir
from os.path import isfile, join

class Tester:
    def __init__(self, dataset, model_path, valid_or_test):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path, map_location = self.device)
        self.model.eval()
        self.dataset = dataset
        self.valid_or_test = valid_or_test
        self.measure = Measure(self.dataset)
        self.all_facts_as_set_of_tuples = set(self.allFactsAsTuples())

    def get_rank(self, sim_scores):#assuming the test fact is the first one
        return (sim_scores >= sim_scores[0]).sum()

    def create_queries(self, fact, head_or_tail):
        head, rel, tail = fact
        if head_or_tail == "head":
            return [(i, rel, tail) for i in range(self.dataset.num_ent())]
        elif head_or_tail == "tail":
            return [(head, rel, i) for i in range(self.dataset.num_ent())]

    def add_fact_and_shred(self, fact, queries, raw_or_fil):
        if raw_or_fil == "raw":
            result = [tuple(fact)] + queries
        elif raw_or_fil == "fil":
            result = [tuple(fact)] + list(set(queries) - self.all_facts_as_set_of_tuples)

        return self.shred_facts(result)

    def test(self):
        # settings = ["raw", "fil"] if self.valid_or_test == "test" else ["fil"]
        settings = ["fil"]
        running_min = 0.0
        running_mean = 0.0
        running_max = 0.0
        norm_ = 0.0
        for i, fact in enumerate(self.dataset.data[self.valid_or_test]):
            for head_or_tail in ["head", "tail"]:
                queries = self.create_queries(fact, head_or_tail)
                for raw_or_fil in settings:
                    h, r, t = self.add_fact_and_shred(fact, queries, raw_or_fil)
                    sim_scores = self.model(h, r, t).cpu().data.numpy()


                    rank = self.get_rank(sim_scores)
                    self.measure.update(rank, raw_or_fil)

                    if self.valid_or_test == "test":
                        if self.dataset.bin_setting == 0:
                            self.measure.update_per_frequency(min(self.dataset.entity_counter[fact[0]], self.dataset.entity_counter[fact[2]]), rank)
                        elif self.dataset.bin_setting == 1:  
                            self.measure.update_per_frequency(self.dataset.entity_counter[fact[0]] + self.dataset.entity_counter[fact[2]], rank)
                        elif self.dataset.bin_setting == 2:
                            if head_or_tail == "head":
                                self.measure.update_per_frequency(self.dataset.entity_counter[fact[0]], rank)
                            elif head_or_tail == "tail":
                                self.measure.update_per_frequency(self.dataset.entity_counter[fact[2]], rank)
                        elif self.dataset.bin_setting == 3:
                            if head_or_tail == "head":
                                self.measure.update_per_frequency(self.dataset.head_relation_counter[(fact[0], fact[1])], rank)
                            elif head_or_tail == "tail":
                                self.measure.update_per_frequency(self.dataset.relation_tail_counter[(fact[1], fact[2])], rank)

        self.measure.normalize(len(self.dataset.data[self.valid_or_test]))
        self.measure.print_()
        if self.valid_or_test == "test":
            self.measure.normalize_per_frequency(self.model.emb_dim)

        return self.measure.mrr["fil"]

    def shred_facts(self, triples):
        heads  = [triples[i][0] for i in range(len(triples))]
        rels   = [triples[i][1] for i in range(len(triples))]
        tails  = [triples[i][2] for i in range(len(triples))]
        return torch.LongTensor(heads).to(self.device), torch.LongTensor(rels).to(self.device), torch.LongTensor(tails).to(self.device)

    def allFactsAsTuples(self):
        tuples = []
        for spl in self.dataset.data:
            for fact in self.dataset.data[spl]:
                tuples.append(tuple(fact))
        
        return tuples



    
    
