import torch
from torch.nn import functional as F
from torch import Tensor

class CompressedLinear(torch.nn.Module):
    def __init__(self, 
                 in_features:  int,
                 rank:         int, 
                 out_features: int,
                 bias:         bool,
                 device=None,
                 dtype=None) -> None:
    
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.V = torch.nn.Parameter(torch.empty((rank, in_features), **factory_kwargs))
        self.U = torch.nn.Parameter(torch.empty((out_features, rank), **factory_kwargs))
        self.S = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
    
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def forward(self, x:Tensor) -> Tensor:
        return F.linear(x, self.S, self.bias) + F.linear(F.linear(x, self.V), self.U)
    
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, rank={self.rank}, out_features={self.out_features}, bias={self.bias is not None}"

class CompressedQKV(torch.nn.Module):
    def __init__(self, 
                 in_features:  int,
                 q_rank:       int, 
                 q_out:        int,
                 k_rank:       int, 
                 k_out:        int,
                 v_rank:       int, 
                 v_out:        int,
                 bias:         bool,
                 device=None,
                 dtype=None) -> None:
    
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        self.in_features = in_features
        self.q_rank = q_rank
        self.k_rank = k_rank
        self.v_rank = v_rank

        self.q_out = q_out
        self.k_out = k_out
        self.v_out = v_out

        self.q_V = torch.nn.Parameter(torch.empty((q_rank, in_features), **factory_kwargs))
        self.q_U = torch.nn.Parameter(torch.empty((q_out, q_rank), **factory_kwargs))
        self.q_S = torch.nn.Parameter(torch.empty((q_out, in_features), **factory_kwargs))

        self.k_V = torch.nn.Parameter(torch.empty((v_rank, in_features), **factory_kwargs))
        self.k_U = torch.nn.Parameter(torch.empty((k_out, v_rank), **factory_kwargs))
        self.k_S = torch.nn.Parameter(torch.empty((k_out, in_features), **factory_kwargs))

        self.v_V = torch.nn.Parameter(torch.empty((k_rank, in_features), **factory_kwargs))
        self.v_U = torch.nn.Parameter(torch.empty((v_out, k_rank), **factory_kwargs))
        self.v_S = torch.nn.Parameter(torch.empty((v_out, in_features), **factory_kwargs))
    
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(q_out+k_out+v_out, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def forward(self, x:Tensor) -> Tensor:
        query = F.linear(x, self.q_S) + F.linear(F.linear(x, self.q_V), self.q_U)
        key = F.linear(x, self.k_S) + F.linear(F.linear(x, self.k_V), self.k_U)
        value = F.linear(x, self.v_S) + F.linear(F.linear(x, self.v_V), self.v_U)

        return torch.cat([query, key, value], dim=2) + self.bias if self.bias is not None else torch.cat([query, key, value], dim=2)
    
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, q_out={self.q_out}, q_rank={self.q_rank}, k_out={self.k_out}, k_rank={self.k_rank}, v_out={self.v_out}, v_rank={self.v_rank}, bias={self.bias is not None}"