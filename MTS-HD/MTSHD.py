import torch
import torch.nn as nn
import torchhd as hd

class HDC_TS(nn.Module):
    """
    Hyperdimensional Computing for Multivariate Time Series
    Compatible with main.py template: fit(), accuracy()
    """
    def __init__(self, n_features, n_dimensions, n_classes, n_levels=256, n_gram=3, eta=0.5, max_channels=64, device='cpu'):
        super().__init__()
        self.D = n_dimensions
        self.n_classes = n_classes
        self.n_levels = n_levels
        self.n_gram = n_gram
        self.eta = eta
        self.max_channels = max_channels
        self.device = device

        # 初始化类别向量
        self.W = torch.zeros(n_classes, n_dimensions, device=device)

        # Level hypervectors
        self.levels = torch.linspace(0, 1, n_levels, device=device)
        self.basis_hvs = hd.random_hv(n_dim=n_dimensions, batch=n_levels, device=device)

        # 通道 hypervectors
        self.channel_hvs = hd.random_hv(n_dim=n_dimensions, batch=max_channels, device=device)

    def channel_encode(self, x):
        """
        x: [batch, T, C] float tensor
        return: [batch, T, C, D]
        """
        B, T, C = x.shape
        xmin, _ = x.view(B, -1).min(dim=1, keepdim=True)
        xmax, _ = x.view(B, -1).max(dim=1, keepdim=True)
        x_norm = (x - xmin.unsqueeze(-1)) / (xmax - xmin + 1e-8)

        idx_float = x_norm * (self.n_levels - 1)
        idx_low = idx_float.floor().long().clamp(0, self.n_levels - 2)
        idx_high = idx_low + 1
        lam = idx_float - idx_low.float()

        H_li = self.basis_hvs[idx_low]
        H_li1 = self.basis_hvs[idx_high]
        phi = (1 - lam.unsqueeze(-1)) * H_li + lam.unsqueeze(-1) * H_li1

        channel_vecs = self.channel_hvs[:C].unsqueeze(0).unsqueeze(0)
        phi_ch = phi * channel_vecs
        return phi_ch

    def temporal_encode(self, phi_ch):
        """
        phi_ch: [batch, T, C, D]
        return: [batch, D] binary hypervector
        """
        B, T, C, D_dim = phi_ch.shape
        v_t = phi_ch.sum(dim=2)
        v_t = hd.normalize(v_t)

        s = torch.zeros(B, D_dim, device=self.device)
        for t in range(T - self.n_gram + 1):
            window = []
            for j in range(self.n_gram):
                permuted = hd.permute(v_t[:, t + j], n_positions=self.n_gram - j - 1)
                window.append(permuted)
            window_hv = torch.prod(torch.stack(window, dim=0), dim=0)
            s += window_hv

        s_b = torch.sign(s)
        s_b[s_b == 0] = 1
        return s_b

    def forward(self, x):
        phi_ch = self.channel_encode(x.to(self.device))
        s_b = self.temporal_encode(phi_ch)
        return s_b

    def update_class(self, s_b, labels):
        for i in range(s_b.shape[0]):
            self.W[labels[i]] += self.eta * s_b[i]

    def fit(self, dataloader):
        for batch_x, batch_y in dataloader:
            s_b = self.forward(batch_x)
            self.update_class(s_b, batch_y.to(self.device))

    def infer(self, s_b):
        W_norm = hd.normalize(self.W)
        scores = torch.matmul(s_b, W_norm.T)
        pred = scores.argmax(dim=1)
        return pred

    def accuracy(self, dataloader):
        correct = 0
        total = 0
        for batch_x, batch_y in dataloader:
            s_b = self.forward(batch_x)
            pred = self.infer(s_b)
            correct += (pred == batch_y.to(self.device)).sum().item()
            total += batch_x.shape[0]
        return correct / total