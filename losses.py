import torch
import torch.nn.functional as F


class FeatureQueue:
    def __init__(self, dim, size, device):
        self.size = size
        self.device = device
        self.features = torch.empty(0, dim, device=device)

    def ready(self):
        return self.features.size(0) > 0

    @torch.no_grad()
    def enqueue(self, z):
        z = F.normalize(z.detach(), dim=1)
        self.features = torch.cat([self.features, z], dim=0)
        if self.features.size(0) > self.size:
            self.features = self.features[-self.size:]

    def get(self):
        return self.features.detach()


def info_nce(z1, z2, temperature):
    b = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.t()) / temperature
    mask = torch.eye(2 * b, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)
    targets = torch.cat([torch.arange(b, 2 * b), torch.arange(0, b)]).to(z.device)
    return F.cross_entropy(sim, targets)


def similarity_distribution(z, refs, temperature):
    logits = torch.matmul(F.normalize(z, dim=1), F.normalize(refs, dim=1).t()) / temperature
    return F.softmax(logits, dim=1)


def kl_distribution(p_teacher, p_student):
    return F.kl_div(torch.log(p_student.clamp_min(1e-8)), p_teacher.detach(), reduction="batchmean")


def covariance(z):
    zc = z - z.mean(dim=0, keepdim=True)
    return torch.matmul(zc.t(), zc) / max(1, z.size(0) - 1)


def statistic_loss(z_teacher, z_student):
    mean_loss = F.mse_loss(z_student.mean(dim=0), z_teacher.detach().mean(dim=0))
    cov_loss = F.mse_loss(covariance(z_student), covariance(z_teacher.detach()))
    return mean_loss + cov_loss


def add_patch_trigger(x, size=4, value=1.0):
    y = x.clone()
    y[:, :, -size:, -size:] = value
    return y


def flatten_grads(params):
    chunks = []
    for p in params:
        if p.grad is None:
            chunks.append(torch.zeros_like(p).reshape(-1))
        else:
            chunks.append(p.grad.detach().reshape(-1))
    return torch.cat(chunks)


def assign_flat_grad(params, flat):
    offset = 0
    for p in params:
        n = p.numel()
        g = flat[offset:offset + n].view_as(p).to(p.device)
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        p.grad.copy_(g)
        offset += n


def feasible_interval(a0, a1):
    if abs(a1) < 1e-12:
        if a0 >= 0:
            return 0.0, 1.0
        return 1.0, 0.0
    root = -a0 / a1
    if a1 > 0:
        return max(0.0, root), 1.0
    return 0.0, min(1.0, root)


def collaborative_gradient(g_benign, g_backdoor):
    gbgb = torch.dot(g_benign, g_benign).item()
    gbgd = torch.dot(g_benign, g_backdoor).item()
    gdgd = torch.dot(g_backdoor, g_backdoor).item()
    l1, u1 = feasible_interval(gbgd, gbgb - gbgd)
    l2, u2 = feasible_interval(gdgd, gbgd - gdgd)
    low = max(0.0, l1, l2)
    high = min(1.0, u1, u2)
    if low > high:
        w = 0.5
    else:
        w = min(high, max(low, 0.5))
    return w * g_benign + (1.0 - w) * g_backdoor, w
