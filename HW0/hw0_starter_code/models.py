"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to zero.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    sparse: boolean, optional
        Use sparse gradients.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(self, num_users, num_items, embedding_dim=32, layer_sizes=[96, 64],
                 sparse=False, embedding_sharing=True):

        super().__init__()

        self.embedding_dim = embedding_dim

        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************

        self.U = ScaledEmbedding(num_users, embedding_dim, sparse=sparse)
        self.Q = ScaledEmbedding(num_items, embedding_dim, sparse=sparse)

        self.A = ZeroEmbedding(num_users, 1)
        self.B = ZeroEmbedding(num_items, 1)

        if not embedding_sharing:
            self.U_not_shared = ScaledEmbedding(num_users, embedding_dim, sparse=sparse)
            self.Q_not_shared = ScaledEmbedding(num_items, embedding_dim, sparse=sparse)

        self.layer_sizes = layer_sizes

        self.embedding_sharing = embedding_sharing

        #********************************************************
        #********************************************************
        #********************************************************

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of shape (batch,)
        score: tensor
            Tensor of user-item score predictions of shape (batch,)
        """
        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************

        if self.embedding_sharing:
            predictions = torch.bmm(torch.transpose(self.U(user_ids).unsqueeze(2), 1, 2), self.Q(item_ids).unsqueeze(2)).squeeze() + self.A(user_ids).squeeze() + self.B(item_ids).squeeze()
        else:
            predictions = torch.bmm(torch.transpose(self.U_not_shared(user_ids).unsqueeze(2), 1, 2), self.Q_not_shared(item_ids).unsqueeze(2)).squeeze() + self.A(user_ids).squeeze() + self.B(item_ids).squeeze()

        model_score = nn.Sequential(
            nn.Linear(self.layer_sizes[0], self.layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(self.layer_sizes[1], 1),
        )
        score = model_score(torch.cat((self.U(user_ids), self.Q(item_ids), torch.mul(self.U(user_ids), self.Q(item_ids))), axis=1)).squeeze()
        
        #********************************************************
        #********************************************************
        #********************************************************
        ## Make sure you return predictions and scores of shape (batch,)
        if (len(predictions.shape) > 1) or (len(score.shape) > 1):
            raise ValueError("Check your shapes!")
        
        return predictions, score