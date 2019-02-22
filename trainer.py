from dataset import Dataset
from SimplE import SimplE
import torch
import torch.nn as nn
import torch.nn.functional as F
import os 

class Trainer:
    def __init__(self, dataset, args):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = SimplE(dataset.num_ent(), dataset.num_rel(), args.emb_dim, self.device)
        self.dataset = dataset
        self.args = args
        
    def train(self):
        self.model.train()

        optimizer = torch.optim.Adagrad(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay= 0,
            initial_accumulator_value= 0.1 #this is added because of the consistency to the original tensorflow code
        )

        for epoch in range(1, self.args.ne + 1):
            last_batch = False
            total_loss = 0.0

            while not last_batch:
                h, r, t, l = self.dataset.next_batch(self.args.batch_size, neg_ratio=self.args.neg_ratio, device = self.device)
                last_batch = self.dataset.was_last_batch()
                optimizer.zero_grad()
                scores = self.model(h, r, t)
                loss = torch.sum(F.softplus(-l * scores))+ (self.args.reg_lambda * self.model.l2_loss() / self.dataset.num_batch(self.args.batch_size))
                loss.backward()
                optimizer.step()
                total_loss += loss.cpu().item()

            print("Loss in iteration " + str(epoch) + ": " + str(total_loss) + "(" + self.dataset.name + ")")
        
            if epoch % self.args.save_each == 0:
                self.save_model(epoch)

    def save_model(self, chkpnt):
        print("Saving the model")
        directory = "models/" + self.dataset.name + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.model, directory + str(chkpnt) + ".chkpnt")

