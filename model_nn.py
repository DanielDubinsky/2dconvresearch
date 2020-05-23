########################################################################################################################
#                                                                                                                      #
# The main body of this code is taken from:                                                                            #
# https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/01-basics/feedforward_neural_network                #
#                                                                                                                      #
# Adaptations by Maartje ter Hoeve.                                                                                    #
# Comments about adaptations specifically to run this code with Sacred start with 'SACRED'                             #
#                                                                                                                      #
# Please have a look at the Sacred documentations for full details about Sacred itself: https://sacred.readthedocs.io/ #
#                                                                                                                      #
########################################################################################################################

import torch
import torch.nn as nn

from typing import Optional


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class EncDec(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()
        self.enc = ConvGRUCell(input_channels, hidden_channels, kernel_size)
        self.dec = ConvGRUCell(hidden_channels, input_channels, kernel_size)

    def forward(self, input: torch.Tensor):
        batch_size = input.size(0)
        seq_size = input.size(1)

        if (len(input.size()) == 4):
            # Add channel
            input = torch.unsqueeze(input, 2)

        h = None
        for i in range(seq_size):
            x = input[:, i]
            h = self.enc(x, h)

        return self.dec(h)


class ConvGRUCell(nn.Module):
    """
        .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    """

    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size[0] // 2
        self.Wr = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
        self.Wz = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
        self.Wi = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)

        self.Sr = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.Sz = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.Si = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)

    def forward(self, input:torch.Tensor, hidden: Optional[torch.Tensor] = None):
        n, c, h, w = input.size()
        if hidden is None:
            hidden = torch.zeros(n, self.hidden_channels, h, w, device=input.device)
        r_t = torch.sigmoid(self.Wr(input) + self.Sr(hidden))
        z_t = torch.sigmoid(self.Wz(input) + self.Sz(hidden))
        n_t = torch.tanh(self.Wi(input) + r_t * self.Si(hidden))
        h_t = (1 - z_t) * n_t + z_t * hidden

        return h_t


# def gru_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
#     gi = torch.mm(input, w_ih.t()) + b_ih
#     gh = torch.mm(hidden, w_hh.t()) + b_hh
#     i_r, i_i, i_n = gi.chunk(3, 1)
#     h_r, h_i, h_n = gh.chunk(3, 1)
#
#     resetgate = torch.sigmoid(i_r + h_r)
#     inputgate = torch.sigmoid(i_i + h_i)
#     newgate = torch.tanh(i_n + resetgate * h_n)
#     hy = newgate + inputgate * (hidden - newgate)
#
#     return hy
#
#
# def lstm_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
#     # type: (Tensor, Tuple[Tensor, Tensor], Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
#     hx, cx = hidden
#     gates = torch.mm(input, w_ih.t()) + torch.mm(hx, w_hh.t()) + b_ih + b_hh
#
#     ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
#
#     ingate = torch.sigmoid(ingate)
#     forgetgate = torch.sigmoid(forgetgate)
#     cellgate = torch.tanh(cellgate)
#     outgate = torch.sigmoid(outgate)
#
#     cy = (forgetgate * cx) + (ingate * cellgate)
#     hy = outgate * torch.tanh(cy)
#
#     return hy, cy
