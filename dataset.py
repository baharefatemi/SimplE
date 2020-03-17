import numpy as np
import random
import torch
import math
import collections
from scipy import stats

class Dataset:
    def __init__(self, ds_name, bin_setting=-1, nbins=-1):
        self.name = ds_name
        self.dir = "datasets/" + ds_name + "/"
        self.ent2id = {}
        self.rel2id = {}
        self.entity_counter = collections.Counter()
        self.head_relation_counter = collections.Counter()
        self.relation_tail_counter = collections.Counter()

        self.freqs_test = []
        self.data = {"train": self.read_train(self.dir + "train.txt"), "valid": self.read_val(self.dir + "valid.txt"), "test": self.read_test(self.dir + "test.txt")}
        self.batch_index = 0
        self.bin_setting = bin_setting
        self.nbins = nbins

        self.read_test_counter(self.dir + "test.txt")
        self.bin_edges = self.find_bins()

        print("bin edges:", self.bin_edges)

    def read_train(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
        triples = np.zeros((len(lines), 3))
        for i, line in enumerate(lines):
            triples[i] = np.array(self.triple2ids(line.strip().split("\t")))
        return triples
    
    def read_val(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        triples = np.zeros((len(lines), 3))
        for i, line in enumerate(lines):
            triples[i] = np.array(self.triple2ids_val(line.strip().split("\t")))
        return triples

    def read_test(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        triples = np.zeros((len(lines), 3))
        for i, line in enumerate(lines):
            triples[i] = np.array(self.triple2ids_val(line.strip().split("\t")))
        return triples


    def read_test_counter(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        triples = np.zeros((len(lines), 3))
        for i, line in enumerate(lines):
            triples[i] = np.array(self.triple2ids_test_counter(line.strip().split("\t")))
        return triples


    def num_ent(self):
        return len(self.ent2id)
    
    def num_rel(self):
        return len(self.rel2id)
                     
    def triple2ids(self, triple):
        self.head_relation_counter[(self.get_ent_id(triple[0]), self.get_rel_id(triple[1]))] += 1
        self.relation_tail_counter[(self.get_rel_id(triple[1]), self.get_ent_id(triple[2]))] += 1
        return [self.get_ent_id(triple[0]), self.get_rel_id(triple[1]), self.get_ent_id(triple[2])]
    
    def triple2ids_val(self, triple):
        return [self.ent2id[triple[0]], self.get_rel_id(triple[1]), self.ent2id[triple[2]]]
    
    def triple2ids_test_counter(self, triple):
        if self.bin_setting == 0:
            # min(freq[a], freq[b])
            self.freqs_test.append(min(self.entity_counter[self.ent2id[triple[0]]], self.entity_counter[self.ent2id[triple[2]]]))
        elif self.bin_setting == 1:
            # freq[a] + freq[b]
            self.freqs_test.append(self.entity_counter[self.ent2id[triple[0]]] + self.entity_counter[self.ent2id[triple[2]]])
        elif self.bin_setting == 2:
            # freq[a], freq[b]
            self.freqs_test.append(self.entity_counter[self.ent2id[triple[0]]])
            self.freqs_test.append(self.entity_counter[self.ent2id[triple[2]]])
        elif self.bin_setting == 3:
            self.freqs_test.append(self.head_relation_counter[(self.ent2id[triple[0]], self.rel2id[triple[1]])])
            self.freqs_test.append(self.relation_tail_counter[(self.rel2id[triple[1]], self.ent2id[triple[2]])])

   
    def get_ent_id(self, ent):
        if not ent in self.ent2id:
            self.ent2id[ent] = len(self.ent2id)
        self.entity_counter[self.ent2id[ent]] += 1
        return self.ent2id[ent]


    def get_rel_id(self, rel):
        if not rel in self.rel2id:
            self.rel2id[rel] = len(self.rel2id)
        return self.rel2id[rel]
                     
    def rand_ent_except(self, ent):
        rand_ent = random.randint(0, self.num_ent() - 1)
        while(rand_ent == ent):
            rand_ent = random.randint(0, self.num_ent() - 1)
        return rand_ent
                     
    def next_pos_batch(self, batch_size):
        if self.batch_index + batch_size < len(self.data["train"]):
            batch = self.data["train"][self.batch_index: self.batch_index+batch_size]
            self.batch_index += batch_size
        else:
            batch = self.data["train"][self.batch_index:]
            self.batch_index = 0
        return np.append(batch, np.ones((len(batch), 1)), axis=1).astype("int") #appending the +1 label
                     
    def generate_neg(self, pos_batch, neg_ratio):
        neg_batch = np.repeat(np.copy(pos_batch), neg_ratio, axis=0)
        for i in range(len(neg_batch)):
            if random.random() < 0.5:
                neg_batch[i][0] = self.rand_ent_except(neg_batch[i][0]) #flipping head
            else:
                neg_batch[i][2] = self.rand_ent_except(neg_batch[i][2]) #flipping tail
        neg_batch[:,-1] = -1
        return neg_batch

    def next_batch(self, batch_size, neg_ratio, device):
        pos_batch = self.next_pos_batch(batch_size)
        neg_batch = self.generate_neg(pos_batch, neg_ratio)
        batch = np.append(pos_batch, neg_batch, axis=0)
        np.random.shuffle(batch)
        heads  = torch.tensor(batch[:,0]).long().to(device)
        rels   = torch.tensor(batch[:,1]).long().to(device)
        tails  = torch.tensor(batch[:,2]).long().to(device)
        labels = torch.tensor(batch[:,3]).float().to(device)
        return heads, rels, tails, labels
    
    def was_last_batch(self):
        return (self.batch_index == 0)

    def num_batch(self, batch_size):
        return int(math.ceil(float(len(self.data["train"])) / batch_size))


    def find_bins(self):
        percentage = []
        for counter in range(self.nbins + 1):
            percentage.append(counter/self.nbins)
        
        bin_edges = stats.mstats.mquantiles(list(self.freqs_test), percentage)

        bin_counter = collections.Counter()

        for val in self.freqs_test:
            bin_counter[self.which_bin(val, bin_edges)] += 1
        print("bin counter:", bin_counter)

        return bin_edges

    def which_bin(self, frequency, bin_edges):
        for ind, val in enumerate(bin_edges[1:]):
            if frequency < val:
                return ind
        return len(bin_edges) - 2

