import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import logging
from functools import partial
from evaluation.metrics import sample_fid_and_is
from utils import get_norm, batch_forward
import os


@torch.no_grad()
def PID_ALD(x_mod, scorenet, sigmas, n_steps_each, step_lr,
                    k_p, k_i, k_d, k_i_decay, k_d_decay, 
                    recording_hook, saving_hook, evaluation_hook, visualization_hook,
                    device, batch_size, verbose=True, denoise=True,
    ):
    """
    PID-controlled anneal lagevin dynamics. Sample saving, evaluation and recording can be customized by hooks.

    Args:
        x_mod (torch.Tensor): Initial samples of shape (n_samples, n_channels, height, width), typically (10000,3,32,32).
        scorenet (nn.Module): Score network, which takes a batch of samples of shape (batch_size, n_channels, height, width)
            and a batch of labels of shape (batch_size,), and outputs a batch of gradients
            of shape (batch_size, n_channels, height, width).
        sigmas (np.ndarray): 1-d numpy array of noise levels.
        n_steps_each (int): Number of steps for each noise level.
        step_lr (float): Step size constant.
        
        k_p (float): Coefficient for Proportional gain.
        k_i (float): Coefficient for Integral gain.
        k_d (float): Coefficient for Derivative gain.
        k_i_decay (float): Decay rate for integral gain. The integral gain is multiplied by `k_i_decay` every noise level.
        k_d_decay (float): Decay rate for derivative gain. The derivative gain is multiplied by `k_d_decay` every noise level.
        
        recording_hook (Callable): A function that manages recording at each step.
        saving_hook (Callable): A function that manages sample saving at each step.
        evaluation_hook (Callable): A function that manages evaluation at each step.
        visualization_hook (Callable): A function that manages visualization at each step.

        device (torch.device): Device to run the model.
        batch_size (int): Batch size for data loading in score network gradient computation.
        verbose (bool): Whether to print the logging information.
        denoise (bool): Whether to add an additional step to denoise the final sample.


    Returns:
        x_mod (torch.Tensor): The final image after sampling.
    """
    x_mod = x_mod.to(device)
    e_int=torch.zeros_like(x_mod).to(x_mod.device) # The mean of historical gradients within the current window
    e_prev=torch.zeros_like(x_mod).to(x_mod.device)
    e_diff=torch.zeros_like(x_mod).to(x_mod.device)
    e_t=torch.zeros_like(x_mod).to(x_mod.device)
    
    global_step = 0 # Total number of sampling steps
    for c, sigma in enumerate(sigmas): # Iterate over noise levels
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        if verbose:
            logging.info("level: {:>4}, k_p={:>7.4f}, k_i={:>7.4f}, k_d={:>7.4f}, sigma={:>7.4f}".format(c, k_p, k_i, k_d, sigma))

        for t in range(n_steps_each): # Iterate over steps within each noise level

            # Proportional gain
            grad = batch_forward(scorenet, x_mod, c, batch_size=batch_size)
            
            # Integral gain
            e_int = (e_int * global_step + grad) / (global_step + 1) # Update the mean of historical gradients
            
            # Derivative gain
            e_prev=e_t
            e_t = grad
            e_diff = e_t - e_prev
            
            # !IMPORTANT: Updating formula
            noise = torch.randn_like(x_mod) # (n_samples, *sample_shape)
            x_mod = x_mod + step_size * (k_p * grad + k_i * e_int + k_d * e_diff) + noise * np.sqrt(step_size * 2)
            
            recording_hook(x_mod, c, t, global_step,
                           k_p=k_p, k_i=k_i, k_d=k_d, step_size=step_size, 
                           grad=grad, e_int=e_int, e_diff=e_diff, noise=noise)
            saving_hook(x_mod, c, t, global_step, end=False)
            evaluation_hook(x_mod, c, t, global_step, end=False)
            visualization_hook(x_mod, c, t, global_step, end=False)

            k_i = k_i * k_i_decay
            k_d = k_d * k_d_decay
            global_step = global_step + 1

    # Final denoising step
    if denoise:
        grad = batch_forward(scorenet, x_mod, len(sigmas)-1, batch_size=batch_size)
        x_mod = x_mod + sigmas[-1] ** 2 * grad

        saving_hook(x_mod, c, t, global_step, end=True)
        evaluation_hook(x_mod, c, t, global_step, end=True)
        visualization_hook(x_mod, c, t, global_step, end=True)
    
    return x_mod.to('cpu')


class SavingHook:
    """"""
    def __init__(self, save, freq, last_only, sample_save_dir, verbose):
        self.save = save
        self.freq = freq
        self.last_only = last_only
        self.sample_save_dir = sample_save_dir
        self.verbose = verbose

    def __call__(self, x_mod, level, step, global_step, end=False):
        if self.save:
            sample_save_path = None
            if end==False and not self.last_only and global_step % self.freq == 0:
                sample_save_path = os.path.join(self.sample_save_dir, 'samples_level_{:03d}_step_{:03d}.pth'.format(level, step))
            elif end==True:
                sample_save_path = os.path.join(self.sample_save_dir, 'samples_final_denoised.pth')
            if sample_save_path is not None:
                torch.save(x_mod.detach().cpu(), sample_save_path)
                if self.verbose:
                    logging.info("level: {:>4}, step: {:>4}, Sample saved to '{}'".format(level, step, sample_save_path))


class EvaluationHook:
    """"""
    def __init__(self, inception_v3_model, mu_real, sigma_real, device,
                    batch_size, num_workers,
                    evaluate, freq, last_only,
                    verbose=True
                ):
        self.evaluate = evaluate
        self.evaluate_func = partial(sample_fid_and_is, inception_v3_model=inception_v3_model,
                                        mu_real=mu_real, sigma_real=sigma_real,
                                        device=device, batch_size=batch_size, num_workers=num_workers)
        self.freq = freq
        self.last_only = last_only

        self.metric_record_dict = {
            'fids': [],
            'is_means': [],
            'is_stds': []
        }

        self.verbose = verbose
    
    def __call__(self, x_mod, level, step, global_step, end=False):
        flag=False
        if self.evaluate:
            if end==False and not self.last_only and global_step % self.freq == 0:
                flag=True
            elif end==True:
                flag=True
        if flag:
            fid, is_mean, is_std = self.evaluate_func(x_mod)
            self.metric_record_dict['fids'].append(fid)
            self.metric_record_dict['is_means'].append(is_mean)
            self.metric_record_dict['is_stds'].append(is_std)
            if self.verbose:
                logging.info("level: {:>4}, step: {:>4}, FID: {:>11.6f}, IS_mean: {:>11.6f}, IS_std: {:>11.6f}".format(
                    level, step, fid, is_mean, is_std))


class RecordingHook:
    """"""
    def __init__(self, verbose=True):
        self.sampler_record_dict = {
            'grad_norms': [], 'e_int_norms': [], 'e_diff_norms': [],
            'P_term_norms': [], 'I_term_norms': [], 'D_term_norms': [],
            'IP_ratios': [], 'DP_ratios': [],
            'PID_term_norms': [], 'noise_term_norms': [], 'delta_term_norms': [],
            'snrs': [],
            'image_norms': [],
        }
        self.verbose=verbose

    def __call__(self, x_mod, level, step, global_step,
                k_p, k_i, k_d, step_size,
                grad, e_int, e_diff, noise):
        # Compute the norms
        grad_norm = get_norm(grad)
        noise_norm = get_norm(noise)
        image_norm = get_norm(x_mod)
        e_int_norm = get_norm(e_int)
        e_diff_norm = get_norm(e_diff)

        P_term = step_size * k_p * grad
        I_term = step_size * k_i * e_int
        D_term = step_size * k_d * e_diff
        PID_term = P_term + I_term + D_term
        noise_term = noise * np.sqrt(step_size * 2)
        delta_term = PID_term + noise_term
                
        P_term_norm = get_norm(P_term)
        I_term_norm = get_norm(I_term)
        D_term_norm = get_norm(D_term)
        IP_ratio = I_term_norm/P_term_norm
        DP_ratio = D_term_norm/P_term_norm
        PID_term_norm = get_norm(PID_term)
        noise_term_norm = get_norm(noise_term)
        delta_term_norm = get_norm(delta_term)
        snr = PID_term_norm/noise_term_norm # Signal to Noise Ratio

        # Recording
        self.sampler_record_dict['grad_norms'].append(grad_norm)
        self.sampler_record_dict['e_int_norms'].append(e_int_norm)
        self.sampler_record_dict['e_diff_norms'].append(e_diff_norm)
        self.sampler_record_dict['P_term_norms'].append(P_term_norm)
        self.sampler_record_dict['I_term_norms'].append(I_term_norm)
        self.sampler_record_dict['D_term_norms'].append(D_term_norm)
        self.sampler_record_dict['IP_ratios'].append(IP_ratio)
        self.sampler_record_dict['DP_ratios'].append(DP_ratio)
        self.sampler_record_dict['PID_term_norms'].append(PID_term_norm)
        self.sampler_record_dict['noise_term_norms'].append(noise_term_norm)
        self.sampler_record_dict['delta_term_norms'].append(delta_term_norm)
        self.sampler_record_dict['snrs'].append(snr)
        self.sampler_record_dict['image_norms'].append(image_norm)

        if self.verbose:
            message = "level: {:>4}, step: {:>4}".format(level, step)
            message += ", image_norm: {:>13.8f}, step_size: {:>13.8f}".format(image_norm, step_size)
            message += ", grad_norm: {:>13.8f}, noise_norm: {:>13.8f}, snr: {:>13.8f}".format(
                grad_norm, noise_norm, snr)
            message += ", e_int_norm: {:>13.8f}, e_diff_norm: {:>13.8f}".format(
                e_int_norm, e_diff_norm)
            message += ", P_norm: {:>13.8f}, I_norm: {:>13.8f}, D_norm: {:>13.8f}".format(
                P_term_norm, I_term_norm, D_term_norm)
            message += ", IP_ratio: {:>13.8f}, DP_ratio: {:>13.8f}".format(
                IP_ratio, DP_ratio)
            message += ", PID_norm: {:>13.8f}, noise_term_norm: {:>13.8f}, delta_term_norm: {:>13.8f}".format(
                PID_term_norm, noise_term_norm, delta_term_norm)
            logging.info(message)


class VisualizationHook:
    """"""
    def __init__(self, save, freq, last_only, nrow, sample_save_dir, verbose):
        self.save = save
        self.freq = freq
        self.last_only = last_only
        self.nrow = nrow
        self.sample_save_dir = sample_save_dir
        self.verbose = verbose

    def __call__(self, x_mod, level, step, global_step, end=False):
        if self.save:
            sample_save_path = None
            nrow=self.nrow
            if end==False and not self.last_only and global_step % self.freq == 0:
                sample_save_path = os.path.join(self.sample_save_dir, 'image_grid_{}x{}_level_{:03d}_step_{:03d}.png'.format(nrow, nrow, level, step))
            elif end==True:
                sample_save_path = os.path.join(self.sample_save_dir, 'image_grid_final_denoised.png')
            if sample_save_path is not None:
                plt.figure(figsize=(8,8))
                grid=torchvision.utils.make_grid(x_mod.detach().cpu()[:nrow*nrow], nrow=nrow, padding=2).permute(1,2,0).numpy()
                plt.imshow(grid)
                plt.xticks([])
                plt.yticks([])
                plt.savefig(sample_save_path, bbox_inches='tight')
                plt.close()
                image_grid_save_path = sample_save_path.replace('.png', '.pth')
                torch.save(grid, image_grid_save_path)
                if self.verbose:
                    logging.info("level: {:>4}, step: {:>4}, figure saved to '{}'".format(level, step, sample_save_path))
                    logging.info("level: {:>4}, step: {:>4}, image grid tensor saved to '{}'".format(level, step, image_grid_save_path))


#--------------------
# deprecated
@torch.no_grad()
def MILD(x_mod, scorenet, inception_v3_model,
                            sigmas, device,
                            mu_real, sigma_real,
                            gamma_dynamic=False, gamma=0.0,
                            n_steps_each=200, step_lr=0.000008,
                            final_only=True, verbose=True, denoise=True,
                            batch_size=100, num_workers=0,
                            evaluate_sample=False, evaluation_freq=10, evaluate_last=True,
                            save_sample=True, sample_save_dir='fcald_samples', sample_save_freq=20,
                            ):
    """
    Momentum-Imbued Langevin Dynamics (MILD).

    Args:
        x_mod (torch.Tensor): Initial samples of shape (n_samples, n_channels, height, width), typically (10000,3,32,32).
        scorenet (nn.Module): Score network, which takes a batch of samples of shape (batch_size, n_channels, height, width)
            and a batch of labels of shape (batch_size,), and outputs a batch of gradients.
            of shape (batch_size, n_channels, height, width).
        inception_v3_model (nn.Module): Inception v3 model used for FID and IS evaluation.
        sigmas (np.ndarray): 1-d numpy array of noise levels.
        device (torch.device): Device to run the model.
        mu_real (np.ndarray): Mean of the inception features of the real samples.
        sigma_real (np.ndarray): Covariance matrix of the inception features of the real samples.
        gamma_dynamic (bool): Whether to use dynamic gamma.
        gamma (float): Coefficient for momentum term.
        n_steps_each (int): Number of steps for each noise level.
        step_lr (float): Step size constant.
        final_only (bool): Deprecated, always True.
        verbose (bool): Whether to print the logging information.
        denoise (bool): Whether to add an additional step to denoise the final sample.
        batch_size (int): Batch size for data loading in score network gradient computation and evaluation.
        num_workers (int): Number of workers for data loading in evaluation.
        evaluate_sample (bool): Whether to calculate FID and IS of the intermediate samples.
        evaluation_freq (int): Number of sampling steps of the interval of FID and IS evaluation.
        evaluate_last (bool): Whether to calculate FID and IS of the final sample.
        save_sample (bool): Whether to save the intermediate samples.
        sample_save_dir (str): Directory to save the intermediate samples.
        sample_save_freq (int): Number of sampling steps of sample saving interval.

    Returns:
        (x_mod, sampler_record_dict) (tuple): Final samples and evaluation results.
        - x_mod (torch.Tensor): Final samples.
        - sampler_record_dict (dict): Dictionary of sampling record.
    """
    assert final_only == True
    
    if save_sample and not os.path.exists(sample_save_dir):
        os.makedirs(sample_save_dir, exist_ok=True)
        logging.info("Created directory '{}' to save samples.".format(sample_save_dir))

    def get_norm(x):
        """Compute the average Frobenius norm over a batch of tensors."""
        return torch.norm(x.view(x.shape[0], -1), p='fro', dim=-1).mean().item()
    
    def dynamicGamma(t,start):
        """Reference: https://github.com/mani-312/mild/blob/main/models/__init__.py, line 28."""
        # Celeba
        # T = 1
        # 1 --  0.4(<200),0.5(<300),0.6(<400),0.7
        # 2 -- 0.4(<300),0.5(<400),0.6

        # T = 2
        # 1 -- 0.2(<300),0.3(<400),0.4
        # 2 -- 0.2(<300),0.25(<400),0.3

        # T = 3(500*3)
        # 1 -- 0.1(<500),0.15(<750),0.2(<1000),0.25

        # Cifar 10
        # 232*5
        # 1 -- 0.1(<250),0.15(<500),0.2(<750),0.25(<1k),0.3
        if t<500:
            return 0.1
        elif t<600:
            return 0.12
        elif t<800:
            return 0.15
        elif t<1000:
            return 0.17
        elif t<1250:
            return 0.2
        return 0.25

        # p = 1/(1-start)
        # a = 1-(p**(-1-math.log2(t//250 + 1)))
        # return min(a,0.9)

    sampler_record_dict = {
        'grad_norms': [],
        'signal_term_norms': [], 'noise_term_norms': [], 'delta_term_norms': [],
        'snrs': [],
        'image_norms': [],
        'fids': [], 'is_means': [], 'is_stds': []
    }

    with torch.no_grad():
        V = torch.zeros_like(x_mod).to(x_mod.device)
        time_step = 0 # Total number of sampling steps.
        
        for c, sigma in enumerate(sigmas): # Iterate over noise levels
            step_size = step_lr * (sigma / sigmas[-1]) ** 2

            for t in range(n_steps_each): # Iterate over steps within each noise level
                
                # Compute gradients
                grad=[]
                num_batches = (len(x_mod) + batch_size - 1) // batch_size
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = (i+1) * batch_size
                    if end_idx > len(x_mod):
                        end_idx = len(x_mod)
                    x = x_mod[start_idx:end_idx]
                    v = V[start_idx:end_idx]
                    labels = torch.ones(x.shape[0], device=x_mod.device) * c
                    labels = labels.long()
                    grad.append(scorenet(x + gamma * v, labels))
                
                # Proportional gain
                grad = torch.cat(grad, dim=0) # Shape: (n_samples, n_channels, image_size, image_size)
                
                if gamma_dynamic:
                    gamma = dynamicGamma(time_step,0.4)
                if verbose:
                    logging.info("gamma: {:.6f}".format(gamma))
                
                V = gamma*V + step_size * grad

                # !IMPORTANT: Updating formula
                noise = torch.randn_like(x_mod) # (n_samples, *sample_shape)
                x_mod = x_mod + V + noise * np.sqrt(step_size * 2)
                
                # Compute the norms
                grad_norm = get_norm(grad)
                noise_norm = get_norm(noise)
                image_norm = get_norm(x_mod)

                signal_term_norm = get_norm(V)
                noise_term = noise * np.sqrt(step_size * 2)
                delta_term = V + noise_term
                
                noise_term_norm = get_norm(noise_term)
                delta_term_norm = get_norm(delta_term)
                snr = signal_term_norm/noise_term_norm # Signal to Noise Ratio

                # Recording
                sampler_record_dict['grad_norms'].append(grad_norm)
                sampler_record_dict['signal_term_norms'].append(signal_term_norm)
                sampler_record_dict['noise_term_norms'].append(noise_term_norm)
                sampler_record_dict['delta_term_norms'].append(delta_term_norm)
                sampler_record_dict['snrs'].append(snr)
                sampler_record_dict['image_norms'].append(image_norm)
                
                # Evaluating and Recording
                if evaluate_sample and time_step % evaluation_freq == 0:
                    fid, is_mean, is_std = sample_fid_and_is(x_mod, inception_v3_model, mu_real, sigma_real, device, batch_size, num_workers)
                    sampler_record_dict['fids'].append(fid)
                    sampler_record_dict['is_means'].append(is_mean)
                    sampler_record_dict['is_stds'].append(is_std)

                # Logging
                if verbose:
                    message = "level: {:3}, step: {:3}".format(c, t)
                    message += ", image_norm: {:>13.8f}, step_size: {:>13.8f}".format(image_norm, step_size)
                    message += ", grad_norm: {:>13.8f}, noise_norm: {:>13.8f}, snr: {:>13.8f}".format(
                        grad_norm, noise_norm, snr)
                    message += ", signal_term_norm: {:>13.8f}".format(signal_term_norm)
                    message += ", noise_term_norm: {:>13.8f}, delta_term_norm: {:>13.8f}".format(
                        noise_term_norm, delta_term_norm)
                    logging.info(message)

                    if evaluate_sample and time_step % evaluation_freq == 0:
                        message1 = "level: {}, step: {}".format(c, t)
                        message1 += "image_norm: {:.6f}, fid: {:.6f}, is_mean: {:.6f}, is_std: {:.6f}".format(image_norm, fid, is_mean, is_std)
                        logging.info(message1)

                # Saving
                if save_sample and time_step % sample_save_freq == 0:
                    sample_save_path = os.path.join(sample_save_dir, 'samples_level_{:03d}_step_{:03d}.pth'.format(c, t))
                    torch.save(x_mod.detach().cpu(), sample_save_path)
                    logging.info("Sample saved to {}".format(sample_save_path))

                time_step += 1

        # Saving the final samples before the last denoising step (if any)
        if save_sample:
            sample_save_path = os.path.join(sample_save_dir, 'samples_final.pth')
            torch.save(x_mod.detach().cpu(), sample_save_path)
            logging.info("Sample saved to {}".format(sample_save_path))

        # Final denoising step
        if denoise:
            grad = []
            num_batches = (len(x_mod) + batch_size - 1) // batch_size
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = (i+1) * batch_size
                if end_idx > len(x_mod):
                    end_idx = len(x_mod)
                x = x_mod[start_idx:end_idx]
                last_noise = (len(sigmas) - 1) * torch.ones(x.shape[0], device=x.device)
                last_noise = last_noise.long()
                grad.append(scorenet(x, last_noise))
            grad = torch.cat(grad, dim=0) # Shape: (n_samples, channels, image_size, image_size)
            x_mod = x_mod + sigmas[-1] ** 2 * grad
    
            if verbose:
                image_norm = get_norm(x_mod)
                logging.info("Final denoising step: image_norm: {:>13.8f}".format(image_norm))
            
            if save_sample:
                sample_save_path = os.path.join(sample_save_dir, 'samples_final_denoised.pth')
                torch.save(x_mod.detach().cpu(), sample_save_path)
                logging.info("Final denoised sample saved to {}".format(sample_save_path))

            if evaluate_last:
                fid, is_mean, is_std = sample_fid_and_is(x_mod, inception_v3_model, mu_real, sigma_real, device, batch_size, num_workers)
                sampler_record_dict['fids'].append(fid)
                sampler_record_dict['is_means'].append(is_mean)
                sampler_record_dict['is_stds'].append(is_std)
                if verbose:
                    logging.info("Final denoising step, fid: {:.6f}, is_mean: {:.6f}, is_std: {:.6f}".format(fid, is_mean, is_std))

        return x_mod.to('cpu'), sampler_record_dict


