# Implementation of D-NPSVC++
# Written by Apple Zhang, 2023.

# pytorch things
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.utils.data as data
from torchvision import datasets, transforms
from torchvision import models
from torch import nn

# aux stuff
from tqdm import tqdm
from typing import Any, Tuple, Dict
from datetime import datetime
import math
import copy

# cvxopt
import cvxopt as cvx
import numpy  as np
cvx.solvers.options['show_progress'] = False

# npsvcpp module
from npsvcpp_loss import *

class NPSVCPP_Net(nn.Module):
    __constants__ = ["hidden", "out_z", "drop_p"]
    hidden: int
    out_z:  int
    drop_p: float

    def __init__(self, prior: nn.Module, dim_in: int, num_classes: int, skip_conn: bool = True):
        super(NPSVCPP_Net, self).__init__()
        self.num_classes = num_classes
        self.in_features = dim_in
        self.prior = prior
        self.skip_conn = skip_conn

        self.hidden = 512
        self.out_z  = 128
        self.drop_p = 0.25

        self.mlp = nn.Sequential(
            nn.Linear(self.in_features, self.hidden),
            nn.Dropout(self.drop_p, inplace=True),
            nn.BatchNorm1d(self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, self.out_z),
            nn.Dropout(self.drop_p, inplace=True),
            nn.BatchNorm1d(self.out_z),
            nn.ReLU(inplace=True),
        )
        
        if skip_conn:
            self.cls = nn.Linear(self.out_z + self.in_features, num_classes)
            self.ln  = nn.LayerNorm(self.out_z + self.in_features)
        else:
            self.cls = nn.Linear(self.out_z, num_classes)
            self.ln  = nn.LayerNorm(self.out_z)

        self.allow_grad_prior = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.allow_grad_prior:
            x = self.prior(x)
        else:
            with torch.no_grad():
                x = self.prior(x)

        # skip connection
        z_x = self.mlp(x)
        if self.skip_conn:
            z_x = torch.cat((z_x, x), dim=1)
            z_x = self.ln(z_x)
        else:
            z_x = self.ln(z_x)
        
        return self.cls(z_x)

    def stop_tune_prior(self):
        self.allow_grad_prior = False

    def activate_tune_prior(self):
        self.allow_grad_prior = True

class NPSVCPP_Runtime:
    def __init__(self,
                 net: NPSVCPP_Net):
        self._net = net
        self._tau = torch.ones(net.num_classes) / net.num_classes

    def __call__(self, input: torch.Tensor):
        return self._net.forward(input)
    
    @property
    def net(self):
        return self._net
    
    @property
    def tau(self):
        return self._tau
    
    @tau.setter
    def tau(self, value: torch.Tensor):
        self._tau = value

    def train_mode(self):
        self._net.train()

    def eval_mode(self):
        self._net.eval()

    def snapshot(self):
        return {"net": self._net.to("cpu").state_dict(), "tau": self.tau.data}

    def load_snapshot(self, state):
        self._net.load_state_dict(state["net"])
        self._tau = state["tau"].data

    def to(self, device: torch.device):
        self._net.to(device)
        self._tau = self._tau.to(device)
        return self

class GammaSchedule:
    __constants__ = ["init_gamma", "gamma_min"]
    def __init__(self, gamma: float, method="cos", T_max=25, decay=0.965, gamma_min=1e-6):
        assert method in ["cos", "exp", "none"]

        self.init_gamma = gamma
        self.gamma_min  = gamma_min

        self.gamma = self.init_gamma
        self.epoch = 0
        
        if method == "cos":
            self.update_gamma = (lambda: self.init_gamma * (math.cos(self.epoch * math.pi / T_max) + 1) / 2 + gamma_min) 
        elif method == "exp":
            self.update_gamma = (lambda: self.init_gamma * decay ** self.epoch + gamma_min )
        else:
            self.update_gamma = lambda: self.init_gamma

    def step(self):
        self.gamma = self.update_gamma()
        self.epoch += 1

class NPSVCPP:
    def __init__(self, net: NPSVCPP_Net,
                 num_classes: int,
                 lr=1e-3,
                 weight_decay=1e-3,
                 name="NPSVCPP",
                 simloss=SimilarLoss("square"),
                 dissimloss=DissimilarLoss("hinge-square"),
                 normW=False,
                 device="cuda:0",
                 beta=0.4,
                 mtype="rvo",
                 **hyperparamNPSVCPP: Dict[str, torch.Tensor]):

        assert mtype in ["ovr", "rvo"]
        self.mtype = mtype
        
        self.runtime = NPSVCPP_Runtime(net)
        # self.net = net
        self.name = name

        self.num_classes = num_classes
        self.lr = lr
        self.normW = normW

        # optimizer of Neural Network and tau
        self.optimizer = optim.SGD(self.runtime.net.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        # self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.lr_sch = lrs.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1.0e-5, verbose=True)
        # self.lr_sch = lrs.StepLR(self.optimizer, step_size=20, gamma=0.5)

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.simloss  = simloss
        self.disloss  = dissimloss

        # hyperparameter
        self.param = hyperparamNPSVCPP
        self.logger = []
        self.val = []

        # state dict
        self.best_state = self.runtime.snapshot()
        self.best_res = 0.0

        self.gsch = GammaSchedule(self.param["gamma"])
        self.tau_mean = self.runtime.tau
        self.beta = beta

    def train(self, tr_data: data.DataLoader, val_data: data.DataLoader, n_epoch=50, val_freq=5, verbose=False, ID="-", ftepoch=10):
        # training routine
        self.runtime.to(self.device)
        print("Trainer is: %s" % self.name)
        print("Training ID: %s" % ID)
        print("train with device: ", self.device)

        # start training
        start_time = datetime.now()
        for epoch in tqdm(range(n_epoch)):
            if epoch == ftepoch:
                self.runtime.net.stop_tune_prior()
            loss, feasible = self._epoch_train(tr_data)

            # recording
            infm = {"loss": loss.to("cpu"), 
                    "tau": self.runtime.tau.to("cpu"),
                    "dual": torch.dot(self.runtime.tau.to("cpu"), loss.to("cpu")),
                    "primal": torch.max(loss),
                    "feasible": feasible}
            
            if verbose:
                print(f"\n=======================\
                        \nPrimal  = {infm['primal']}\
                        \nDual    = {infm['dual']} \
                        \nDualGap = {infm['primal']-infm['dual']}\
                        \nTau     = {infm['tau']}\
                        \nObj     = {infm['loss']}\
                        \nFeasi   = {infm['feasible']}\
                        \n=======================")
            
            # try validation
            if val_data is not None and (1+epoch) % val_freq == 0:
                valResult = self._epoch_val(val_data)
                self.val.append(valResult)
                print(valResult)
                infm["acc"] = valResult["acc"]
            else:
                infm["acc"] = -1.0

            self.logger.append(infm)
        
        # save the running state
        current_time = datetime.now()
        with open("./log/%s-%s-log-%s.csv" % (ID, self.name, current_time.strftime("%Y%m%d%H%M")), "w+") as f:
            f.write("trainer_name:%s,n_epoch:%d,training_time:%4f\n" % (self.name, n_epoch, (current_time - start_time).total_seconds()))
            f.write("primal,dual,gap,feasible,valacc\n")
            # f.write("n_epoch:%d\n" % n_epoch)
            for infm in self.logger:
                f.write(f"{infm['primal']},{infm['dual']},{infm['primal']-infm['dual']},{infm['feasible']},{infm['acc']}\n")

    def test(self, te_data: data.DataLoader, verbose=False):
        self.runtime.to(self.device)
        res = self._epoch_val(te_data)
        self.best_res = res["acc"]
        if verbose:
            print(res)
        return res
    
    def save_model(self, ID: str, use_best_state=True):
        current_time = datetime.now()
        current_time.strftime("%Y%m%d%H%M")
        filename = './model/%s_model_%s_%s-acc=%.4f.pth' % (self.name, current_time.strftime("%Y%m%d%H%M"), ID, self.best_res)
        if use_best_state:
            self.runtime.load_snapshot(self.best_state)
        torch.save(self.runtime.net, filename)
        print("model saved. file: %s." % filename)
    
    def predict(self, x: torch.Tensor, y: torch.Tensor):
        self.runtime.eval_mode()

        # computing the ``score''
        self.runtime.to(self.device)
        x = x.to(self.device)
        target = self.runtime(x)
        # the farthest hyperplane
        if self.mtype == "rvo":
            pred = torch.argmax(target.abs(), dim=1)
        else:
            pred = torch.argmin(target.abs(), dim=1)
        return pred
    
    def save_best_state(self, filename=None):
        self.best_state = copy.deepcopy(self.runtime.snapshot())
        if filename is not None:
            torch.save(self.best_state, filename)
        self.runtime.to(self.device)
    
    def load_best_state(self, filename=None):
        self.best_state = torch.load(filename)
        self.runtime.load_snapshot(self.best_state)

    def rewind_best_model(self):
        if self.best_state is not None:
            self.runtime.load_snapshot(self.best_state)

#### PRIVATE METHODS   
    """
    
    """
    def _epoch_train(self, tr_data: data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        self.runtime.train_mode()
        train_loss = torch.zeros(self.num_classes).to(self.device)
        self.runtime.tau.to(self.device)
        feasible = 0.0

        cnt = 1
        n_sample = 0
        for x, y in tr_data:
            x = x.to(self.device)   # x: (batch_size, H, W, C)
            y = y.to(self.device)

            n = x.shape[0]
            n_sample += n

            ### phase 1: theta(t+1) = argmin tau(t)'J(theta)
            self.runtime.net.cls.requires_grad_(True)
            self.optimizer.zero_grad()
            target  = self.runtime(x)
            loss, _ = self._losses_with_grad(target, y)
            h = torch.dot(loss, self.runtime.tau)
            h.backward()
            self.optimizer.step()

            ### phase 2: tau(t+1) = argmax tau'J(theta(t+1)), s.t. pareto stationary
            self.runtime.net.cls.requires_grad_(False)
            self.optimizer.zero_grad()
            target = self.runtime(x)
            loss, grad = self._losses_with_grad(target, y, ask_grad=True)

            # compute feasible constraint and duality objective
            # self.optimizer.zero_grad()
            Q_feas = grad @ grad.T
            h = loss.detach()
            tau = self._qpcvxopt_tau(Q_feas, -self.gsch.gamma * h)

            self.runtime.tau = self.beta * tau + (1-self.beta) * self.runtime.tau

            # average the tau
            # self.tau_mean = self.tau_mean + tau / cnt

            feasible += 0.5*torch.sum((grad.T @ self.runtime.tau.unsqueeze(1)).pow(2)).item()

            # again descent
            loss = torch.dot(loss, self.runtime.tau)
            loss.backward()
            self.optimizer.step()

            train_loss += h * n

        print("\n", self.lr_sch.get_last_lr())
        self.lr_sch.step()
        self.gsch.step()
        print(self.gsch.gamma)

        # update runtime tau
        # self.runtime.tau = self.tau_mean

        # update the reweighting parameter
        # self.param["gamma"] = 0.75*self.param["gamma"] + 2.5*self.lr_sch.get_last_lr()[0]
        return train_loss / n_sample, feasible / len(tr_data)
    
    def _epoch_val(self, val_data: data.DataLoader):
        self.runtime.eval_mode()
        valid_loss = torch.zeros(self.num_classes)
        correct = 0
        n_sample  = 0
        for x, y in val_data:
            x = x.to(self.device)   # x: (batch_size, H, W, C)
            y = y.to(self.device)

            n_batch = x.shape[0]

            target = self.runtime(x)
            loss, _ = self._losses_with_grad(target, y)
            valid_loss += loss.data.to("cpu") * n_batch

            pred = self.predict(x, y)
            correct += torch.sum(pred == y).item()
            n_sample  += n_batch
        acc = correct / n_sample

        return {"acc": acc, "loss": valid_loss / n_sample}


    def _losses_with_grad(self, target: torch.Tensor, gt: torch.Tensor, ask_grad=False) -> Tuple[torch.Tensor, torch.Tensor]:
        y = F.one_hot(gt, num_classes=self.num_classes)
        if self.mtype == "rvo":
            loss = (1-y) * self.simloss(target) / (self.num_classes - 1) + \
                    y  * self.disloss(target) * self.param["c"]
            loss_mean = torch.mean(loss, dim=0)

            with torch.no_grad():
                if ask_grad:
                    grad_zx = (1-y) * self.simloss.grad / (self.num_classes - 1) + \
                                y  * self.disloss.grad * self.param["c"]
                    w = self.runtime.net.cls.weight[:, 0:self.runtime.net.in_features]

                    grad = torch.bmm(w.unsqueeze(dim=2), grad_zx.T.unsqueeze(dim=1))
                    grad = grad.view(self.num_classes, -1)
                else:
                    grad = None
        else:
            loss =    y  * self.simloss(target) + \
                   (1-y) * self.disloss(target) * self.param["c"] / (self.num_classes - 1)
            loss_mean = torch.mean(loss, dim=0)

            with torch.no_grad():
                if ask_grad:
                    grad_zx =    y  * self.simloss.grad + \
                              (1-y) * self.disloss.grad * self.param["c"] / (self.num_classes - 1)
                    w = self.runtime.net.cls.weight[:, 0:self.runtime.net.in_features]

                    grad = torch.bmm(w.unsqueeze(dim=2), grad_zx.T.unsqueeze(dim=1))
                    grad = grad.view(self.num_classes, -1)
                else:
                    grad = None
        
        return loss_mean, grad

    def _qpcvxopt_tau(self, Q: torch.Tensor, h: torch.Tensor, mu=0.85):
        Q_np = Q.to("cpu").numpy().astype(np.double)
        h_np = h.to("cpu").numpy().astype(np.double)

        n = self.num_classes

        Q_cvx = cvx.matrix(Q_np)
        h_cvx = cvx.matrix(h_np)

        ones = cvx.matrix(1.0, (1, n))
        b    = cvx.matrix(1.0)
        zero = cvx.matrix(0.0, (n,1))
        eye  = cvx.matrix(0.0, (n,n))
        eye[::n+1] = -1.0

        sol = cvx.solvers.qp(Q_cvx, h_cvx, eye, zero, ones, b)
        tau = np.ravel(sol["x"]).astype(np.float32)

        return torch.from_numpy(tau).to(self.device)