import torch, math
import torch.nn.functional as F
from einops import rearrange


class FlowMatchScheduler():
    
    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003/1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        exponential_shift=False,
        exponential_shift_mu=None,
        shift_terminal=None,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal
        self.set_timesteps(num_inference_steps)
        self.rho = 4.0

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, shift=None, dynamic_shift_len=None):
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        if self.exponential_shift:
            mu = self.calculate_shift(dynamic_shift_len) if dynamic_shift_len is not None else self.exponential_shift_mu
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing
            self.training = True
        else:
            self.training = False

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample, sigma
    
    def d_metric(self, x, y):
        return torch.sqrt(torch.sum((x - y) ** 2, dim=-1, keepdim=True))
    
    def normalize(self, x):
        return F.normalize(x, p=2, dim=-1)

    def calculate_loss(self, sample, noise, xt, vt_predict, vt_target):
        
        vt = -vt_predict.clone().to(dtype=vt_predict.dtype, device=vt_predict.device)
        rho = self.rho
        
        bb, cc, ff, hh, ww = sample.shape
        device, dtype = sample.device, sample.dtype
        sample_f = rearrange(sample, 'b c f h w -> f (b c h w)').float()
        noise_f = rearrange(noise, 'b c f h w -> f (b c h w)').float()
        xt_f = rearrange(xt, 'b c f h w -> f (b c h w)').float()
        vt_f = rearrange(vt, 'b c f h w -> f (b c h w)').float()
        vt_f_norm = torch.norm(vt_f, p=2, dim=-1, keepdim=True)
        
        vt_target_f = rearrange(-vt_target, 'b c f h w -> f (b c h w)').float()
        vt_target_f_direction = self.normalize(vt_target_f)
        vt_f_direction = self.normalize(vt_f)
        loss_1 = (1 - torch.nn.functional.cosine_similarity(vt_target_f_direction, vt_f_direction, dim=-1)).mean()
        
        C_star = self.d_metric(sample_f[:-2], sample_f[1:-1]) + self.d_metric(sample_f[1:-1], sample_f[2:]) \
                    - self.d_metric(noise_f[:-2], noise_f[1:-1]) - self.d_metric(noise_f[1:-1], noise_f[2:]) \
                        - rho * self.d_metric(noise_f[1:-1], sample_f[1:-1])
        C_star_begin = (self.d_metric(sample_f[0], sample_f[1]) - self.d_metric(noise_f[0], noise_f[1]) - rho * self.d_metric(noise_f[0], sample_f[0])).unsqueeze(1)
        C_star_end = (self.d_metric(sample_f[-2], sample_f[-1]) - self.d_metric(noise_f[-2], noise_f[-1]) - rho * self.d_metric(noise_f[-1], sample_f[-1])).unsqueeze(1)
        C_star = torch.cat([C_star_begin, C_star, C_star_end], dim=0) # d_i
        
        A_star = self.normalize(sample_f - noise_f)
        
        A_star_main = rho * self.normalize(noise_f[1:-1] - sample_f[1:-1]) + self.normalize(xt_f[1:-1] - xt_f[:-2]) + self.normalize(xt_f[1:-1] - xt_f[2:])
        A_star_main_start = (rho * self.normalize(noise_f[0] - sample_f[0]) + self.normalize(xt_f[0] - xt_f[1])).unsqueeze(0)
        A_star_main_end = (rho * self.normalize(noise_f[-1] - sample_f[-1]) + self.normalize(xt_f[-1] - xt_f[-2])).unsqueeze(0)
        A_star_main = torch.cat([A_star_main_start, A_star_main, A_star_main_end], dim=0)
        A_star_main = (A_star_main * (A_star * vt_f_norm)).sum(dim=1) # b_i
        
        A_star_below = self.normalize(xt_f[:-1] - xt_f[1:])
        A_star_below = (A_star_below * (A_star[:-1] * vt_f_norm[:-1])).sum(dim=1) # a_i
        
        A_star_above = self.normalize(xt_f[1:] - xt_f[:-1]) 
        A_star_above = (A_star_above * (A_star[1:] * vt_f_norm[1:])).sum(dim=1) # c_i
        
        A_star = torch.cat([
            (A_star_main[0] + A_star_above[0]).unsqueeze(0),
            A_star_main[1:-1] + A_star_above[1:] + A_star_below[:-1],
            (A_star_main[-1] + A_star_below[-1]).unsqueeze(0)
        ], dim=0).unsqueeze(1)

        loss_2 = torch.mean(torch.abs(torch.div(A_star.abs(), C_star.abs() + 1e-6) - 1))
        
        beta = 1.0
        loss = loss_1 + beta * loss_2
        
        return loss

    def training_target_FM(self, sample, noise, timestep):
        target = noise - sample
        return target
    
    def training_target(self, sample, noise, timestep):
        dtype = sample.dtype
        sample = sample.float()
        noise = noise.float()
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sigmas = reversed(torch.arange(sigma, 1.0, 0.002, dtype=sigma.dtype)) # max 500 step
        xt = noise.clone()
        for it, jt in zip(sigmas[:-1], sigmas[1:]):
            dt = jt - it
            vt = self.calculate_u(sample, noise, xt, self.rho)
            xt = xt + vt * dt
        ut = self.calculate_u(sample, noise, xt, self.rho)

        return xt.to(dtype=dtype), ut.to(dtype=dtype)
        
    def calculate_u(self, sample, noise, xt, rho):
        bb, cc, ff, hh, ww = sample.shape
        device, dtype = sample.device, sample.dtype
        
        sample_f = rearrange(sample, 'b c f h w -> f (b c h w)')
        noise_f = rearrange(noise, 'b c f h w -> f (b c h w)')
        xt_f = rearrange(xt, 'b c f h w -> f (b c h w)')
    
        C_star = self.d_metric(sample_f[:-2], sample_f[1:-1]) + self.d_metric(sample_f[1:-1], sample_f[2:]) \
            - self.d_metric(noise_f[:-2], noise_f[1:-1]) - self.d_metric(noise_f[1:-1], noise_f[2:]) \
                - rho * self.d_metric(noise_f[1:-1], sample_f[1:-1])
        C_star_begin = (self.d_metric(sample_f[0], sample_f[1]) - self.d_metric(noise_f[0], noise_f[1]) - rho * self.d_metric(noise_f[0], sample_f[0])).unsqueeze(1)
        C_star_end = (self.d_metric(sample_f[-2], sample_f[-1]) - self.d_metric(noise_f[-2], noise_f[-1]) - rho * self.d_metric(noise_f[-1], sample_f[-1])).unsqueeze(1)
        C_star = torch.cat([C_star_begin, C_star, C_star_end], dim=0) # d_i
        
        A_star = self.normalize(sample_f - noise_f)
        
        A_star_main = rho * self.normalize(noise_f[1:-1] - sample_f[1:-1]) + self.normalize(xt_f[1:-1] - xt_f[:-2]) + self.normalize(xt_f[1:-1] - xt_f[2:])
        A_star_main_start = (rho * self.normalize(noise_f[0] - sample_f[0]) + self.normalize(xt_f[0] - xt_f[1])).unsqueeze(0)
        A_star_main_end = (rho * self.normalize(noise_f[-1] - sample_f[-1]) + self.normalize(xt_f[-1] - xt_f[-2])).unsqueeze(0)
        A_star_main = torch.cat([A_star_main_start, A_star_main, A_star_main_end], dim=0)
        A_star_main = (A_star_main * A_star).sum(dim=1)
        
        A_star_below = self.normalize(xt_f[:-1] - xt_f[1:])
        A_star_below = (A_star_below * A_star[:-1]).sum(dim=1)
        
        A_star_above = self.normalize(xt_f[1:] - xt_f[:-1]) 
        A_star_above = (A_star_above * A_star[1:]).sum(dim=1)
        
        CC = torch.zeros((ff - 1, ), device=device, dtype=dtype)
        CC[0] = A_star_above[0] / A_star_main[0]
        DD = torch.zeros((ff, ), device=device, dtype=dtype)
        DD[0] = C_star[0] / A_star_main[0]
        for i in range(1, ff - 1):
            tmp = A_star_main[i] - A_star_below[i - 1] * CC[i - 1]
            CC[i] = A_star_above[i] / tmp
            DD[i] = (C_star[i] - A_star_below[i - 1] * DD[i - 1]) / tmp
        DD[ff - 1] = (C_star[ff - 1] - A_star_below[ff - 2] * DD[ff - 2]) / (A_star_main[ff - 1] - A_star_below[ff - 2] * CC[ff - 2])
        
        HH = torch.zeros((ff, ), device=device, dtype=dtype)
        HH[-1] = DD[-1]
        for i in range(ff - 2, -1, -1):
            HH[i] = DD[i] - CC[i] * HH[i + 1]
        
        u_stars = HH.unsqueeze(1) * A_star
        u_stars = rearrange(u_stars, 'f (b c h w) -> b c f h w', b=bb, c=cc, h=hh, w=ww)
 
        return -u_stars
        

    def training_weight(self, timestep):
        timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        weights = self.linear_timesteps_weights[timestep_id]
        return weights
    
    
    def calculate_shift(
        self,
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 8192,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu
