import torch
import torch.nn.functional as F


def cal_sal_rank_loss(real_pred, lite_pred, target, margin=0):
    B, T, K = real_pred.shape

    # TODO(shape) B * T
    b_idx = [[x] * T for x in range(B)]
    t_idx = [list(range(T)) for _ in range(B)]
    k_idx = [[tgt] * T for tgt in target[:, 0].cpu().numpy()]

    # TODO(shape) B * T
    real_cfd = real_pred[b_idx, t_idx, k_idx]
    lite_cfd = lite_pred[b_idx, t_idx, k_idx]

    x, y = torch.triu_indices(T - 1, T - 1) + torch.tensor([[0], [1]])

    # TODO(shape) B * (T*(T-1)/2)
    real_cfd_x = real_cfd[:, x]
    real_cfd_y = real_cfd[:, y]
    lite_cfd_x = lite_cfd[:, x]
    lite_cfd_y = lite_cfd[:, y]

    psu_label = (real_cfd_x > real_cfd_y).double() * 2 - 1

    return F.margin_ranking_loss(lite_cfd_x, lite_cfd_y, psu_label, margin=margin, reduction="mean")


if __name__ == "__main__":
    # B=2, T=3, K=4
    a = torch.tensor([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.2, 0.1, 0.2], [0.3, 0.3, 0.3, 0.1]],
                      [[0.3, 0.1, 0.1, 0.1], [0.2, 0.2, 0.5, 0.3], [0.4, 0.4, 0.3, 0.1]]])
    b = torch.tensor([[[0.0, 0.0, 0.0, 0.3], [0.0, 0.0, 0.0, 0.2], [0.0, 0.0, 0.0, 0.3]],
                      [[0.0, 0.0, 0.1, 0.0], [0.0, 0.0, 0.3, 0.0], [0.0, 0.0, 0.2, 0.0]]])
    target = torch.tensor([3, 2])
    margin = 0

    print("Expect: 0.0000, reality:", cal_sal_rank_loss(a, a, target, 0))  # TODO(yue) expect to become 0
    print("Expect: 0.0167, reality:", cal_sal_rank_loss(a, b, target, 0))  # TODO(yue) expect to become 0.1/6=0.0167
    print("Expect: 0.9333, reality:", cal_sal_rank_loss(a, b, target, 1))  # TODO(yue) expect to become 5.6/6=0.9333
