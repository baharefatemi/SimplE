class Measure:
    def __init__(self):
        self.hit1  = {"raw": 0.0, "fil": 0.0}
        self.hit3  = {"raw": 0.0, "fil": 0.0}
        self.hit10 = {"raw": 0.0, "fil": 0.0}
        self.mrr   = {"raw": 0.0, "fil": 0.0}
        self.mr    = {"raw": 0.0, "fil": 0.0}

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
    