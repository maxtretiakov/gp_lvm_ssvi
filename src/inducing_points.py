import torch

def sample_U_batch(m_u: torch.Tensor, C_u: torch.Tensor, S: int) -> torch.Tensor:
    eps   = torch.randn(S, *m_u.shape, device=m_u.device).unsqueeze(-1)   # (S,D,M,1)
    C_exp = C_u.unsqueeze(0).expand(S, -1, -1, -1)                        # (S,D,M,M)
    return m_u.unsqueeze(0) + (C_exp @ eps).squeeze(-1)                   # (S,D,M)
