import collections
# import matplotlib.pylab as plt
class Measure:
    def __init__(self, dataset):
        self.hit1  = {"raw": 0.0, "fil": 0.0}
        self.hit3  = {"raw": 0.0, "fil": 0.0}
        self.hit10 = {"raw": 0.0, "fil": 0.0}
        self.mrr   = {"raw": 0.0, "fil": 0.0}
        self.mr    = {"raw": 0.0, "fil": 0.0}
        self.dataset = dataset
        self.nbins = dataset.nbins

        self.hit1_per_frequency = {}
        self.hit3_per_frequency = {}
        self.hit10_per_frequency = {}
        self.mr_per_frequency = {}
        self.mrr_per_frequency = {}
        self.normalizer_per_frequency = {}
        for i in range(self.nbins):
            self.hit1_per_frequency[i] = 0
            self.hit3_per_frequency[i] = 0
            self.hit10_per_frequency[i] = 0
            self.mr_per_frequency[i] = 0
            self.mrr_per_frequency[i] = 0
            self.normalizer_per_frequency[i] = 0

        self.mrr_for_all_test = 0.0
        self.normalizer_for_all_test = 0.0
        self.frequency_counter = collections.Counter()

    def update(self, rank, raw_or_fil):
        if rank == 1:
            self.hit1[raw_or_fil] += 1.0
        if rank <= 3:
            self.hit3[raw_or_fil] += 1.0
        if rank <= 10:
            self.hit10[raw_or_fil] += 1.0

        self.mr[raw_or_fil]  += rank
        self.mrr[raw_or_fil] += (1.0 / rank)
    
    def normalize(self, num_facts):
        for raw_or_fil in ["raw", "fil"]:
            self.hit1[raw_or_fil]  /= (2 * num_facts)
            self.hit3[raw_or_fil]  /= (2 * num_facts)
            self.hit10[raw_or_fil] /= (2 * num_facts)
            self.mr[raw_or_fil]    /= (2 * num_facts)
            self.mrr[raw_or_fil]   /= (2 * num_facts)

    def print_(self):
        for raw_or_fil in ["raw", "fil"]:
            print(raw_or_fil.title() + " setting:")
            print("\tHit@1 =",  self.hit1[raw_or_fil])
            print("\tHit@3 =",  self.hit3[raw_or_fil])
            print("\tHit@10 =", self.hit10[raw_or_fil])
            print("\tMR =",     self.mr[raw_or_fil])
            print("\tMRR =",    self.mrr[raw_or_fil])
            print("")

    def update_per_frequency(self, frequency, rank):

        self.frequency_counter[frequency] += 1

        bin_= self.which_bin(frequency)

        if rank == 1:
            self.hit1_per_frequency[bin_] += 1.0
        if rank <= 3:
            self.hit3_per_frequency[bin_] += 1.0
        if rank <= 10:
            self.hit10_per_frequency[bin_] += 1.0
        self.mr_per_frequency[bin_]  += rank        
        self.mrr_per_frequency[bin_] += (1.0 / rank)
        self.normalizer_per_frequency[bin_] += 1

        self.mrr_for_all_test += (1.0 / rank)
        self.normalizer_for_all_test += 1

    def which_bin(self, frequency):
        bin_edges = self.dataset.bin_edges
        for ind, val in enumerate(bin_edges[1:]):
            if frequency < val:
                return ind
        return len(bin_edges) - 2


    def normalize_per_frequency(self, fname):

        for key in self.mrr_per_frequency:
            if self.normalizer_per_frequency[key] == 0:
                self.hit1_per_frequency[key] = 0
                self.hit3_per_frequency[key] = 0
                self.hit10_per_frequency[key] = 0
                self.mrr_per_frequency[key] = 0
                self.mr_per_frequency[key] = 0
            else:
                self.hit1_per_frequency[key] /= self.normalizer_per_frequency[key]
                self.hit3_per_frequency[key] /= self.normalizer_per_frequency[key]
                self.hit10_per_frequency[key] /= self.normalizer_per_frequency[key]
                self.mrr_per_frequency[key] /= self.normalizer_per_frequency[key]
                self.mr_per_frequency[key] /= self.normalizer_per_frequency[key]


        print("all mrr for test:", self.mrr_for_all_test / self.normalizer_for_all_test)
        print("hit@1:", self.hit1_per_frequency)
        print("hit@3:", self.hit3_per_frequency)
        print("hit@10:", self.hit10_per_frequency)
        print("mrr:", self.mrr_per_frequency)
        print("normalizer:", self.normalizer_per_frequency)



    