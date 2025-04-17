import torch 
import torch.nn as nn 
import torch.nn.init as init 
class FiLMBlock(nn.Module):
    def __init__(self):
        super(FiLMBlock, self).__init__()
        
    def forward(self, x, gamma, beta):
        beta = beta.view(x.size(0), x.size(1), 1, 1)
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)
        
        x = gamma * x + beta
        
        return x

class FiLMConv(nn.Module):
    def __init__(self, batchNorm, cin, cout, k=3, stride=1, pad=-1, brdf_emb_dim=512, FiLM_hidden_dim=512):
        super(FiLMConv, self).__init__()
        self.batchNorm = batchNorm
        self.cin       = cin
        self.cout      = cout
        self.k         = k
        self.stride    = stride
        self.pad = pad = (k - 1) // 2 if pad < 0 else pad
        if batchNorm:
            self.conv = nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False) 
            self.bn   = nn.BatchNorm2d(cout)
            self.relu = nn.LeakyReLU(0.1, inplace=True)
            self.FiLMGnerator = nn.Linear(brdf_emb_dim, cout * 2)
            self.FiLMBlock = FiLMBlock()
        else:
            self.conv = nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True) 
            self.relu = nn.LeakyReLU(0.1, inplace=True)
            self.FiLMGnerator = nn.Linear(brdf_emb_dim, cout * 2)
            self.FiLMBlock = FiLMBlock()
        self._initialize_film_generator()

    def _initialize_film_generator(self):
        """Initializes the final layer of the FiLM generator,
         so that the gamma and beta are 1s and 0s. 
        """
        layer = self.FiLMGnerator

        # Initialize weights to zero (or very small values)
        init.zeros_(layer.weight)
        # Alternative: Small random values
        # init.normal_(final_layer.weight, mean=0.0, std=0.01)

        # Initialize bias: gamma part to 1, beta part to 0
        if layer.bias is not None:
            # Initialize all biases to zero first
            init.zeros_(layer.bias)
            # Set the first 'cout' bias elements (for gamma) to 1
            layer.bias.data[:self.cout].fill_(1.0)
            # The second 'cout' bias elements (for beta) remain 0
            
    def forward(self, x, brdf_emb_vector):
        if self.batchNorm:
            x = self.conv(x)
            x = self.bn(x)
        else:
            x = self.conv(x)
        
        gamma_beta = self.FiLMGnerator(brdf_emb_vector)
        gamma, beta = torch.split(gamma_beta, self.cout, dim=-1)
        
        x = self.FiLMBlock(x, gamma, beta)
        x = self.relu(x)
        return x