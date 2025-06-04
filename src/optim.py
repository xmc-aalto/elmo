from abc import abstractmethod

from torch.optim import Optimizer
from typing import Callable, Iterable, Optional, Tuple, Union
from transformers.utils.versions import require_version
import torch.nn as nn
import torch
import warnings
from transformers.utils import logging
import math
from triton_kernels.stochastic_rounding_kernel import stochastic_rounding
import random


class AdamW_SR(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                p: torch.Tensor

                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.bfloat16)

                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.bfloat16)
                    
                beta1, beta2 = group["betas"]
                state["step"] += 1

                # Update first and second moment
                #  1) convert from internal representation to fp32
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"].to(torch.float32)

                # 2) do update in fp32
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                p.grad = None
                
                denom = exp_avg_sq.sqrt().add_(group["eps"])
                # 3) update state using internal representation
                state["exp_avg"] = exp_avg
                # stochastic_rounding(exp_avg, state["exp_avg"], random.randint(0, 1000000))
                stochastic_rounding(exp_avg_sq, state["exp_avg_sq"], random.randint(0, 1000000))

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                
                weights = p.to(torch.float32)

                weights.addcdiv_(exp_avg, denom, value=-step_size)
                # handle weight decay
                if group["weight_decay"] > 0.0:
                    weights.add_(weights, alpha=(-group["lr"] * group["weight_decay"]))

                stochastic_rounding(weights, p[...] , random.randint(0, 1000000))

        return loss
