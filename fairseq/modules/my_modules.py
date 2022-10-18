import torch
import torch.nn.functional as F

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 side_padding,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = side_padding * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        # print(id(result))  # printing object refrence
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


class Soft_Gate():
    def __init__(self, distance):
        """

        :param distance: n-1 * B * 2
        """
        self.distance = distance
    def soft_gate(self):

        distances = self.distance
        N, bsz, _ = distances.shape
        src_len = N + 1

        if distances is not None:
            pretext_dist = distances[:, :, 0].permute(1, 0)  # (B, T-1)
            alpha_pre = (
                    (F.hardtanh(
                        (pretext_dist[:, None, :] - pretext_dist[:, :, None]) / 0.1 * 2 + 1
                    ) + 1) / 2
            ).tril()  # B, T-1, T-1
            alpha_pre = torch.cat((
                torch.zeros([bsz, 1, src_len],
                            dtype=alpha_pre.dtype,
                            device=alpha_pre.device),
                torch.cat((alpha_pre,
                           torch.zeros([bsz, src_len - 1, 1],
                                       dtype=alpha_pre.dtype,
                                       device=alpha_pre.device)
                           ), dim=-1)
            ), dim=-2)
            alpha_pre = alpha_pre + torch.ones_like(alpha_pre).triu(diagonal=0)
            gate_pre = torch.cumprod(alpha_pre.flip(dims=[-1]), dim=-1).flip(dims=[-1])

            postext_dist = distances[:, :, 1].permute(1, 0)
            alpha_pos = (
                    (F.hardtanh(
                        (postext_dist[:, None, :] - postext_dist[:, :, None]) / 0.1 * 2 + 1
                    ) + 1) / 2
            ).triu()
            alpha_pos = torch.cat((
                torch.cat((torch.zeros([bsz, src_len - 1, 1],
                                       dtype=alpha_pos.dtype,
                                       device=alpha_pos.device),
                           alpha_pos
                           ), dim=-1),
                torch.zeros([bsz, 1, src_len],
                            dtype=alpha_pos.dtype,
                            device=alpha_pos.device)
            ), dim=-2)
            alpha_pos = alpha_pos + torch.ones_like(alpha_pos).tril(diagonal=0)
            gate_pos = torch.cumprod(alpha_pos, dim=-1)


            gate_pos -= (1 - torch.ones(gate_pos.shape).triu()).to(gate_pos.device)

            gate_pre -= (1 - torch.ones(gate_pre.shape).tril()).to(gate_pre.device)


            return gate_pos, gate_pre


