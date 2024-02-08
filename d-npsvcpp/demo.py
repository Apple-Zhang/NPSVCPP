from npsvcpp import *
from npsvcpp_utils import *

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ============ General setting of training ===============
    parser.add_argument("--dataset", default="cifar10", type=str, help="Use which dataset.")
    parser.add_argument("--backbone", default="resnet34", type=str, help="Use which pretrained model for training.")
    parser.add_argument("--valfreq", default=1, type=int, help="validation frequency")
    parser.add_argument("--batch_size", default=64, type=int, help="training batchsize")
    parser.add_argument("--device", default="cuda:0", type=str, help="Device for training")
    parser.add_argument("--criterion", default="npsvcpp", choices=["npsvcpp", "dtwsvn", "margin", "ce"], type=str, help="choose the learning criterion")

    # ============ hyperparameters of NPSVC++ ===============
    parser.add_argument("--disloss", default="hinge-square", choices=["hinge", "hinge-square", "square"], 
                        type=str, help="Dissimilar loss, optional: ['hinge', 'hinge-square', 'square']")
    parser.add_argument("--simloss", default="square", choices=["absolute", "square"],
                        type=str, help="Similar loss, optional: ['absolute', 'square']")
    parser.add_argument("--deltaSim", default=3.1415926, type=float, help="Similar loss hyperparameter")
    parser.add_argument("--deltaDis", default=2.7182818, type=float, help="Dissimilar loss hyperparameter")
    parser.add_argument("--margin", default=16.0, type=float, help="Margin of the dissimilar loss")
    parser.add_argument("--epoch", default=5, type=int, help="Dissimilar loss hyperparameter")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--wd", default=1e-4, type=float, help="weight decay")
    parser.add_argument("--c", default=1.0, type=float, help="npsvcpp penalty")
    parser.add_argument("--gamma", default=1.0, type=float, help="trade-off parameter in dual optimization.")
    parser.add_argument("--finetunepoch", default=10, type=int, help="Use how many epochs to tune the prior encoder.")
    parser.add_argument("--type", default="ovr", choices=["ovr", "rvo"], type=str, help="multiclass type")
    args = parser.parse_args()
    
    train_loader, test_loader, num_classes = use_data(args.dataset, batch_size=args.batch_size)
    prior, dim_out = use_fea_extractor(args.backbone)

    net = NPSVCPP_Net(prior, dim_out, num_classes, skip_conn=True)

    npsvcpp = NPSVCPP(net, name="NPSVC++",num_classes=num_classes,  
                    simloss=SimilarLoss(args.simloss, delta=args.deltaSim),
                    dissimloss=DissimilarLoss(args.disloss, delta=args.deltaDis, margin=args.margin),
                    device=args.device,
                    c=args.c, gamma=args.gamma, weight_decay=args.wd, lr=args.lr)
    npsvcpp.train(train_loader, test_loader, verbose=True, n_epoch=args.epoch, val_freq=args.valfreq, ID=args.dataset, ftepoch=args.finetunepoch)
    res = npsvcpp.test(test_loader, verbose=True)