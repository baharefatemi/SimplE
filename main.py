from trainer import Trainer
from tester import Tester
from dataset import Dataset
import argparse
import time
def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ne', default=1000, type=int, help="number of epochs")
    parser.add_argument('-lr', default=0.1, type=float, help="learning rate")
    parser.add_argument('-reg_lambda', default=0.03, type=float, help="l2 regularization parameter")
    parser.add_argument('-dataset', default="WN18", type=str, help="wordnet dataset")
    parser.add_argument('-emb_dim', default=200, type=int, help="embedding dimension")
    parser.add_argument('-neg_ratio', default=1, type=int, help="number of negative examples per positive example")
    parser.add_argument('-batch_size', default=1415, type=int, help="batch size")
    parser.add_argument('-save_each', default=50, type=int, help="validate every k epochs")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parameter()
    dataset = Dataset(args.dataset)

    print("~~~~ Training ~~~~")
    trainer = Trainer(dataset, args)
    trainer.train()

    print("~~~~ Select best epoch on validation set ~~~~")
    epochs2test = [str(int(args.save_each * (i + 1))) for i in range(args.ne // args.save_each)]
    dataset = Dataset(args.dataset)
    
    best_mrr = -1.0
    best_epoch = "0"
    for epoch in epochs2test:
        start = time.time()
        print(epoch)
        model_path = "models/" + args.dataset + "/" + epoch + ".chkpnt"
        tester = Tester(dataset, model_path, "valid")
        mrr = tester.test()
        if mrr > best_mrr:
            best_mrr = mrr
            best_epoch = epoch
        print(time.time() - start)

    print("Best epoch: " + best_epoch)

    print("~~~~ Testing on the best epoch ~~~~")
    best_model_path = "models/" + args.dataset + "/" + best_epoch + ".chkpnt"
    tester = Tester(dataset, best_model_path, "test")
    tester.test()
