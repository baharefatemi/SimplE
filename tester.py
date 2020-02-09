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
        self.measure = Measure()
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

    # def replace_and_shred(self, fact, raw_or_fil, head_or_tail):
    #     ret_facts = []
    #     head, rel, tail = fact
    #     for i in range(self.dataset.num_ent()):
    #         if head_or_tail == "head" and i != head:
    #             ret_facts.append((i, rel, tail))
    #         if head_or_tail == "tail" and i != tail:
    #             ret_facts.append((head, rel, i))

    #     if raw_or_fil == "raw":
    #         ret_facts = [tuple(fact)] + ret_facts
    #     elif raw_or_fil == "fil":
    #         ret_facts = [tuple(fact)] + list(set(ret_facts) - self.all_facts_as_set_of_tuples)

    #     return self.shred_facts(ret_facts)
    
    def test(self):
        settings = ["raw", "fil"] if self.valid_or_test == "test" else ["fil"]
        
        for i, fact in enumerate(self.dataset.data[self.valid_or_test]):
            for head_or_tail in ["head", "tail"]:
                queries = self.create_queries(fact, head_or_tail)
                for raw_or_fil in settings:
                    h, r, t = self.add_fact_and_shred(fact, queries, raw_or_fil)
                    sim_scores = self.model(h, r, t).cpu().data.numpy()
                    rank = self.get_rank(sim_scores)
                    self.measure.update(rank, raw_or_fil)

        self.measure.normalize(len(self.dataset.data[self.valid_or_test]))
        self.measure.print_()
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



    
    
