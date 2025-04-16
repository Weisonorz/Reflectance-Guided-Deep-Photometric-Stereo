import torch
import torch.nn as nn


'''
CBN (Conditional Batch Normalization layer)
    uses an MLP to predict the beta and gamma parameters in the batch norm equation
    Reference : https://papers.nips.cc/paper/7237-modulating-early-visual-processing-by-language.pdf
'''

class CBN(nn.Module):

    def __init__(self, 
                 brdf_emb_size, 
                 emb_size, 
                 channels, 
                 use_betas=True, 
                 use_gammas=True, 
                 eps=1.0e-5,
                 momentum=0.1,
                 track_running_stats=True):
        super(CBN, self).__init__()

        self.brdf_emb_size = brdf_emb_size # size of the brdf emb which is input to MLP
        self.emb_size = emb_size # size of hidden layer of MLP
        self.out_size = channels
        self.use_betas = use_betas
        self.use_gammas = use_gammas

        self.channels = channels
        self.momentum = momentum
        self.track_running_stats = track_running_stats


        # beta and gamma parameters for each channel - defined as trainable parameters
        self.betas = nn.Parameter(torch.zeros((1, self.channels)))
        self.gammas = nn.Parameter(torch.ones((1, self.channels)))
        self.eps = eps

        # MLP used to predict betas and gammas
        self.fc_gamma = nn.Sequential(
            nn.Linear(self.brdf_emb_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.out_size),
            )

        self.fc_beta = nn.Sequential(
            nn.Linear(self.brdf_emb_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.out_size),
            )
        
        self.delta_betas, self.delta_gammas = None, None

        # Add running mean and variance like in PyTorch BatchNorm
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(channels))
            self.register_buffer('running_var', torch.ones(channels))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)

        # initialize weights using Xavier initialization and biases with constant value
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    '''
    Predicts the value of delta beta and delta gamma for each channel

    Arguments:
        brdf_emb : brdf embedding of the question

    Returns:
        delta_betas, delta_gammas : for each layer
    '''
    def create_cbn_input(self, brdf_emb):

        if self.use_betas:
            delta_betas = self.fc_beta(brdf_emb)
        else:
            delta_betas = torch.zeros(1, self.channels, device=brdf_emb.device)

        if self.use_gammas:
            delta_gammas = self.fc_gamma(brdf_emb)
        else:
            delta_gammas = torch.zeros(1, self.channels, device=brdf_emb.device)

        return delta_betas, delta_gammas

    '''
    Computer Normalized feature map with the updated beta and gamma values

    Arguments:
        feature : feature map from the previous layer
        brdf_emb : brdf embedding of the question

    Returns:
        out : beta and gamma normalized feature map
        brdf_emb : brdf embedding of the question (unchanged)

    Note : brdf_emb needs to be returned since CBN is defined within nn.Sequential
           and subsequent CBN layers will also require brdf question embeddings
    '''
    def forward(self, feature, brdf_emb):
        self.batch_size, self.channels, self.height, self.width = feature.shape

        self.delta_betas, self.delta_gammas = self.create_cbn_input(brdf_emb)


        # update the values of beta and gamma
        betas_modified = self.betas + self.delta_betas
        gamma_modified = self.gammas + self.delta_gammas

        # Reshape feature for statistics calculation: [N,C,H,W] -> [N,H,W,C] -> [N*H*W,C]
        feature_flattened = feature.permute(0, 2, 3, 1).contiguous().view(-1, self.channels)
        
        if self.training and self.track_running_stats:
            # Calculate batch statistics
            batch_mean = feature_flattened.mean(0)
            batch_var = feature_flattened.var(0, unbiased=False)
            
            # Update running statistics
            if self.num_batches_tracked == 0:
                self.running_mean.copy_(batch_mean)
                self.running_var.copy_(batch_var)
            else:
                self.running_mean.mul_(1 - self.momentum).add_(batch_mean * self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(batch_var * self.momentum)
            
            self.num_batches_tracked += 1
        else:
            # Use running statistics in evaluation mode
            batch_mean = self.running_mean
            batch_var = self.running_var

        # Expand mean and var from [C] to [N,C,H,W]
        mean_expanded = batch_mean.view(1, self.channels, 1, 1).expand_as(feature)
        var_expanded = batch_var.view(1, self.channels, 1, 1).expand_as(feature)
        
        # Normalize the feature map
        feature_normalized = (feature - mean_expanded) / torch.sqrt(var_expanded + self.eps)

        # Expand betas and gammas for final transformation
        betas_expanded = betas_modified.view(self.batch_size, self.channels, 1, 1).expand_as(feature)
        gammas_expanded = gamma_modified.view(self.batch_size, self.channels, 1, 1).expand_as(feature)

        # Apply conditional scaling and shifting
        out = torch.mul(feature_normalized, gammas_expanded) + betas_expanded

        return out, brdf_emb
    
