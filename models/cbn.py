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
                 out_size, 
                 batch_size, 
                 channels, 
                 use_betas=True, 
                 use_gammas=True, 
                 eps=1.0e-5):
        super(CBN, self).__init__()

        self.brdf_emb_size = brdf_emb_size # size of the brdf emb which is input to MLP
        self.emb_size = emb_size # size of hidden layer of MLP
        self.out_size = out_size # output of the MLP - for each channel
        self.use_betas = use_betas
        self.use_gammas = use_gammas

        self.batch_size = batch_size
        self.channels = channels

        # beta and gamma parameters for each channel - defined as trainable parameters
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.betas = nn.Parameter(torch.zeros((self.batch_size, self.channels)))
        self.gammas = nn.Parameter(torch.ones((self.batch_size, self.channels)))
        self.eps = eps

        # MLP used to predict betas and gammas
        self.fc_gamma = nn.Sequential(
            nn.Linear(self.brdf_emb_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.out_size),
            ).cuda()

        self.fc_beta = nn.Sequential(
            nn.Linear(self.brdf_emb_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.out_size),
            ).cuda()
        
        self.delta_betas, self.delta_gammas = None, None

        # initialize weights using Xavier initialization and biases with constant value
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0.1)

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
            delta_betas = torch.zeros(self.batch_size, self.channels).cuda()

        if self.use_gammas:
            delta_gammas = self.fc_gamma(brdf_emb)
        else:
            delta_gammas = torch.zeros(self.batch_size, self.channels).cuda()

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
        # print("DEBUG: feature.shape = ", feature.shape)
        # print("DEBUG: brdf_emb.shape = ", brdf_emb.shape)
        self.batch_size, self.channels, self.height, self.width = feature.shape

        if self.delta_betas is None or self.delta_gammas is None:
        # get delta values
            self.delta_betas, self.delta_gammas = self.create_cbn_input(brdf_emb)

        betas_cloned = self.betas.clone()
        gammas_cloned = self.gammas.clone()

        # update the values of beta and gamma
        betas_cloned += self.delta_betas
        gammas_cloned += self.delta_gammas

        # get the mean and variance for the batch norm layer
        batch_mean = torch.mean(feature)
        batch_var = torch.var(feature)

        # extend the betas and gammas of each channel across the height and width of feature map
        betas_expanded = torch.stack([betas_cloned]*self.height, dim=2)
        betas_expanded = torch.stack([betas_expanded]*self.width, dim=3)

        gammas_expanded = torch.stack([gammas_cloned]*self.height, dim=2)
        gammas_expanded = torch.stack([gammas_expanded]*self.width, dim=3)

        # normalize the feature map
        feature_normalized = (feature-batch_mean)/torch.sqrt(batch_var+self.eps)

        # get the normalized feature map with the updated beta and gamma values
        out = torch.mul(feature_normalized, gammas_expanded) + betas_expanded

        return out, brdf_emb
    
    def __call__(self, x, brdf_embed):
        return self.forward(x, brdf_embed)