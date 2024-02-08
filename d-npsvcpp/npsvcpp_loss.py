import torch
import torch.nn as nn
import torch.nn.functional as F

class SimilarLoss:
    def __init__(self, loss_name="square", delta=0.5, require_grad=True):
        lossDict = {"absolute":    self.__absolute,
                    "square":      self.__square}
        self.delta = delta
        self.loss  = lossDict[loss_name.lower()]
        self.name  = loss_name
        self.__require_grad = require_grad
        self.__grad_x = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.loss(x)

    def __absolute(self, x: torch.Tensor) -> torch.Tensor:
        if self.__require_grad:
            self.__grad_x = self.delta * torch.sign(x)
        return x.abs() * self.delta
    
    def __square(self, x: torch.Tensor) -> torch.Tensor:
        if self.__require_grad:
            self.__grad_x = self.delta * x
        return 0.5*x.pow(2) * self.delta
    
    @property
    def grad(self):
        return self.__grad_x
    

class DissimilarLoss:
    __constants__ = ["delta", "margin", "name"]
    
    def __init__(self, loss_name="hinge-softplus", delta=0.5, margin=2.714, require_grad=True):
        lossDict = {"hinge":          self.__hinge,
                    "hinge-square":   self.__hinge_square,
                    "square":         self.__square}
        self.delta  = delta
        self.loss   = lossDict[loss_name.lower()]
        self.name   = loss_name
        self.__require_grad = require_grad
        self.__grad_x = None

        self.margin = margin

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        loss = self.loss((self.margin-x).abs())
        if self.__require_grad:
            self.__grad_x *= (self.margin-x).sign()
        return loss
    
    def __hinge(self, x: torch.Tensor) -> torch.Tensor:
        if self.__require_grad:
            self.__grad_x = -(x>0).float()
        return F.relu(x)
    
    def __hinge_square(self, x: torch.Tensor) -> torch.Tensor:
        if self.__require_grad:
            self.__grad_x = -(x>0).float() * x * self.delta
        return F.relu(x).pow(2) * 0.5 * self.delta
    
    def __square(self, x: torch.Tensor) -> torch.Tensor:
        if self.__require_grad:
            self.__grad_x = -x * self.delta
        return x.pow(2) * 0.5 * self.delta
    
    @property
    def grad(self):
        return self.__grad_x
    
    def z_grad(self, asNone=True):
        if self.__grad_x is None:
            return
        elif asNone:
            self.__grad_x = None
        else:
            self.__grad_x.zero_()