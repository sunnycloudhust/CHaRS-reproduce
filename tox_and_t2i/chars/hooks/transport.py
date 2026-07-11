# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import abc
import copy
import logging
import typing as t
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
torch._dynamo.config.capture_dynamic_output_shape_ops = True

from act.hooks.intervention_hook import InterventionHook
from act.optimal_transport.archs import LinearProj
from act.optimal_transport.ot_maps import solve_ot_1d
from act.utils.quantiles import compute_quantiles
from act.utils.nonparam import *
import einops

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def mse_loss(x: torch.Tensor, y: torch.Tensor) -> t.List[torch.Tensor]:
    return [
        torch.mean(torch.pow((x - y), 2)),
    ]


class XYDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)


def optimize_loop(
    x: torch.Tensor,
    y: torch.Tensor,
    net: nn.Module,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: t.Optional[int] = None,
    loss_fn: t.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = mse_loss,
):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    dataset = XYDataset(x, y)
    if batch_size is not None and batch_size < len(x):
        loader = DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=0, shuffle=True
        )
    else:
        loader = DataLoader(
            dataset=dataset, batch_size=len(x), num_workers=0, shuffle=False
        )
    net.train()
    for epoch in range(epochs):
        # print(epoch)
        for i, batch in enumerate(loader):
            optimizer.zero_grad()
            x_batch, y_batch = batch
            x_proj = net(x_batch)
            loss_terms = loss_fn(x_proj, y_batch)
            loss = sum(loss_terms)
            # Backprop
            loss.backward()
            optimizer.step()
        if epoch == 0 or (epoch % 50) == 0 or epoch == (epochs - 1):
            loss_items_float = [f"{l.item():0.5f}" for l in loss_terms]
            print(
                f"epoch {epoch}, loss {loss.item():0.5f} [{', '.join(loss_items_float)}]"
            )
    return net


class GaussianOTHook(InterventionHook):
    def __init__(
        self,
        module_name: str,
        device: str = None,
        intervention_position: str = "all",
        dtype: torch.dtype = torch.float32,
        strength: float = 1.0,
        std_eps: float = 1e-4,
        quantiles_src: str = "q_all",
        hook_onlymean: bool = False,
        **kwargs,
    ):
        """
        Gaussian Optimal Transport hook. Assumes the output of each neuron is independent and Gaussianly distributed.
        Transports from N(mu1, std1) to N(mu2, std2) if hook_forward is True, and N(mu2, std2) to N(mu1, std1) otherwise.

        Args:
            module_name (str): Module name on which the hook is applied.
            device (torch.device): Torch device.
            dtype (str): The torch dtype for this hook.
            intervention_position (str): The position of the token to intervene upon (all, last)
            **kwargs: Any extra arguments, for compatibility with other hooks.
        """
        super().__init__(
            module_name=module_name,
            device=device,
            intervention_position=intervention_position,
            dtype=dtype,
        )

        self.strength = float(strength)
        self.onlymean = bool(int(hook_onlymean))
        self.std_eps = float(std_eps)
        self.quantiles_src = str(quantiles_src)

        # Register buffer placeholders
        for buffer_name in ["mu1", "mu2", "std1", "std2"]:
            self.register_buffer(buffer_name, torch.empty(0))

        # Register non-buffer placeholders
        self.quantiles_dict_src = None

    def __str__(self):
        txt = (
            f"GaussianOT("
            f"module_name={self.module_name}, "
            f"quantiles_src={self.quantiles_src}, "
            f"strength={self.strength:0.2f}, "
            f"onlymean={self.hook_onlymean}"
            f")"
        )
        return txt

    def state_dict(self, *args, **kwargs) -> t.Dict:
        d = super().state_dict()
        # Merge non-buffers into dict
        d.update(
            {
                "quantiles_dict_src": self.quantiles_dict_src,
            }
        )
        # Merge hook_args required for incremental learning
        d.update(
            {
                "quantiles_src": self.quantiles_src,
            }
        )
        return d

    def load_state_dict(
        self,
        state_dict: t.Mapping[str, t.Any],
        strict: bool = True,
        assign: bool = False,
    ) -> None:
        # Fill in registered buffers
        for buffer_name, _ in self.named_buffers():
            setattr(
                self,
                buffer_name,
                state_dict[buffer_name].to(self.device).to(self.dtype),
            )

        # Fill in non buffers
        self.quantiles_dict_src = {
            k: [v[0].to(self.device), v[1].to(self.device)]
            for k, v in state_dict["quantiles_dict_src"].items()
        }

        # Non tensor parameters. If these exist in state_dict, they will override the ones passed through constructor.
        # This is useful for hooks learnt for some specific args (eg. incremental hooks).
        for hook_arg in [attr for attr in dir(self) if attr.startswith("hook_")]:
            if hook_arg in state_dict:
                # We know the type, __init__() cas been called.
                vartype = type(getattr(self, hook_arg))
                varval = vartype(state_dict[hook_arg])
                logging.warning(
                    f"Overriding {hook_arg} to {varval} in {self.__class__.__name__}."
                )
                setattr(self, hook_arg, varval)
        self._post_load()

    def fit(self, responses: torch.Tensor, labels=torch.Tensor, **kwargs) -> None:
        # Typecasting to float64 to avoid overflow in the computation of the mean/std.
        z_f64 = responses.to(torch.float64)
        labels = labels.to(torch.bool)
        self.mu1 = torch.mean(z_f64[labels], dim=0).to(self.dtype)
        self.std1 = torch.std(z_f64[labels], dim=0).to(self.dtype)
        self.mu2 = torch.mean(z_f64[~labels], dim=0).to(self.dtype)
        self.std2 = torch.std(z_f64[~labels], dim=0).to(self.dtype)
        self.quantiles_dict_src = compute_quantiles(z_f64[labels])
        self._post_load()

    def _post_load(self) -> None:
        """
        This method should be called after loading the states of the hook.
        So calls must be placed at the end of .fit() and at the end of .load_state_dict().
        """
        super()._post_load()

        self.mask = (
            torch.ones_like(
                self.mu1, dtype=torch.bool
            )  # just here to be able to comment/uncomment masks below easily]
            # TODO: We've had this all the time, remove?
            & (self.std1 > self.std_eps)
            & (self.std2 > self.std_eps)
        )

        # Pre-computing things beforehand
        self.mu1_m = self.mu1[self.mask]
        self.mu2_m = self.mu2[self.mask]
        self.std1_m = self.std1[self.mask]
        self.std2_m = self.std2[self.mask]
        self.std1_2_m = self.std2_m / self.std1_m

        # TODO: Move this to some post_load method to avoid doing it every time.
        if self.quantiles_src == "q_all":
            self.quantiles_src = [-1e6, 1e6]
        else:
            self.quantiles_src = copy.deepcopy(
                self.quantiles_dict_src[self.quantiles_src]
            )
            self.quantiles_src[0] = self.quantiles_src[0][self.mask].view(1, -1)
            self.quantiles_src[1] = self.quantiles_src[1][self.mask].view(1, -1)

    def forward(self, module, input, output) -> t.Any:
        output_shape = output.shape
        if len(output_shape) == 3:
            output = output.reshape(-1, output_shape[2])
            self.mu1_m = self.mu1_m.view(1, -1)
            self.mu2_m = self.mu2_m.view(1, -1)
            self.std1_m = self.std1_m.view(1, -1)
            self.std2_m = self.std2_m.view(1, -1)
            self.std1_2_m = self.std1_2_m.view(1, -1)
            z_unit = output[:, self.mask]  # B,  U
            z_mask = z_unit
        elif len(output_shape) == 4:
            # Handles image model case
            self.mu1_m = self.mu1_m.view(1, -1, 1, 1)
            self.mu2_m = self.mu2_m.view(1, -1, 1, 1)
            self.std1_m = self.std1_m.view(1, -1, 1, 1)
            self.std2_m = self.std2_m.view(1, -1, 1, 1)
            self.std1_2_m = self.std1_2_m.view(1, -1, 1, 1)
            z_unit = output[:, self.mask]  # B,  U
            z_mask = z_unit.mean(dim=(2, 3), keepdim=False)  # B, C
        else:
            raise NotImplementedError(f"Can't handle tensors with dim=2.")

        # Select outputs that fall in statistics of pooled
        pool_mask = (
            (self.quantiles_src[0] < z_mask) & (z_mask < self.quantiles_src[1])
        ).to(z_unit.dtype)

        # Transport the outputs
        if self.onlymean:
            z_ot = (z_unit - self.mu1_m) + self.mu2_m
        else:
            z_ot = self.std1_2_m * (z_unit - self.mu1_m) + self.mu2_m

        # Apply transport with specific strength
        z_ot = self.strength * z_ot + (1 - self.strength) * z_unit

        if len(output_shape) == 4:
            pool_mask = pool_mask[..., None, None]

        # Only apply masked outputs
        output[:, self.mask] = pool_mask * z_ot + (1 - pool_mask) * z_unit
        output = output.view(*output_shape)
        return output


class OnlyMeanHook(GaussianOTHook):
    def __init__(
        self,
        module_name: str,
        device: str = None,
        intervention_position: str = "all",
        dtype: torch.dtype = torch.float32,
        strength: float = 1.0,
        std_eps: float = 1e-4,
        quantiles_src: str = "q_all",
        **kwargs,
    ):
        # Hardcoded to only mean.
        kwargs.update({"hook_onlymean": True})

        super().__init__(
            module_name=module_name,
            device=device,
            intervention_position=intervention_position,
            dtype=dtype,
            strength=strength,
            std_eps=std_eps,
            quantiles_src=quantiles_src,
            **kwargs,
        )

    def __str__(self):
        txt = (
            f"OnlyMeanOT("
            f"module_name={self.module_name}, "
            f"quantiles_src={self.quantiles_src}, "
            f"strength={self.strength:0.2f}, "
            f")"
        )
        return txt


class LearnableOTHook(abc.ABC, InterventionHook):
    def __init__(
        self,
        module_name: str,
        device: str = None,
        intervention_position: str = "all",
        dtype: torch.dtype = torch.float32,
        strength: float = 1.0,
        std_eps: float = 1e-4,
        quantiles_src: str = "q_all",
        **kwargs,
    ):
        """
        Gaussian Optimal Transport hook. Assumes the output of each neuron is independent and Gaussianly distributed.
        Transports from N(mu1, std1) to N(mu2, std2) if hook_forward is True, and N(mu2, std2) to N(mu1, std1) otherwise.
        Args:
            module_name (str): Module name on which the hook is applied.
            device (torch.device): Torch device.
            **kwargs: Any extra arguments, for compatibility with other hooks.
        """
        super().__init__(
            module_name=module_name,
            device=device,
            intervention_position=intervention_position,
            dtype=dtype,
        )

        self.strength = float(strength)
        self.std_eps = float(std_eps)
        self.quantiles_src = str(quantiles_src)

        # Register non-buffer placeholders
        self.quantiles_dict_src = None
        self.net = None

    def __str__(self):
        txt = (
            f"LearnableOT("
            f"module_name={self.module_name}, "
            f"quantiles_src={self.quantiles_src}, "
            f"strength={self.strength:0.2f}, "
            f")"
        )
        return txt

    @abc.abstractmethod
    def get_net(self) -> t.Type[torch.nn.Module]:
        pass

    def state_dict(self, *args, **kwargs) -> t.Dict:
        d = super().state_dict()
        # Merge non-buffers into dict
        d.update(
            {
                "quantiles_dict_src": self.quantiles_dict_src,
            }
        )
        # Merge hook_args required for incremental learning
        d.update(
            {
                "quantiles_src": self.quantiles_src,
            }
        )
        return d

    def load_state_dict(
        self,
        state_dict: t.Mapping[str, t.Any],
        strict: bool = True,
        assign: bool = False,
    ):
        # Fill in non buffers
        self.quantiles_dict_src = {
            k: [v[0].to(self.device), v[1].to(self.device)]
            for k, v in state_dict["quantiles_dict_src"].items()
        }

        # Fill in the net
        net_type = self.get_net()

        dim = state_dict["net.w1"].shape[-1]
        self.net = net_type(dim=dim)

        # This loads the net statedict.
        net_dict = {k: v for k, v in state_dict.items() if k.startswith("net.")}
        super().load_state_dict(net_dict)
        self.net = self.net.to(self.device)

        if not torch.allclose(self.net.w1, state_dict["net.w1"]) or (
            "net.A" in state_dict
            and not torch.allclose(self.net.A, state_dict["net.A"])
        ):
            msg = "Net load_state_dict() failed.\n"
            msg += "self.net:\n"
            for k, v in self.net.named_parameters():
                msg += f"{k}  {v.shape}\n"
            raise RuntimeError(msg)

        # Non tensor parameters. If these exist in state_dict, they will override the ones passed through constructor.
        # This is useful for hooks learnt for some specific args (eg. incremental hooks).
        for hook_arg in [attr for attr in dir(self) if attr.startswith("hook_")]:
            if hook_arg in state_dict:
                # We know the type, __init__() cas been called.
                vartype = type(getattr(self, hook_arg))
                varval = vartype(state_dict[hook_arg])
                logging.warning(
                    f"Overriding {hook_arg} to {varval} in {self.__class__.__name__}."
                )
                setattr(self, hook_arg, varval)
        self._post_load()

    @abc.abstractmethod
    def _fit_net(
        self,
        z_src_ot: torch.Tensor,
        z_dst_ot: torch.Tensor,
        use_gd: bool = False,
        **kwargs,
    ):
        pass

    def fit(
        self, responses: torch.Tensor, labels=torch.Tensor, use_gd=False, **kwargs
    ) -> None:
        z_f64 = responses.to(torch.float32)
        labels = labels.to(torch.bool)
        z_src = z_f64[labels]  # Bs, D
        z_dst = z_f64[~labels]  # Bd, D
        dim = z_src.shape[-1]

        # Find OT pairs "PER NEURON" (not jointly!).
        z_src_ot, z_dst_ot = torch.empty_like(z_src), torch.empty_like(z_dst)  # B, S
        for idx in range(dim):
            z_src_ot[:, idx], z_dst_ot[:, idx] = solve_ot_1d(
                z_src[:, idx], z_dst[:, idx]
            )  # D

        logger.info(
            f"Transport data after filtering {z_src_ot.shape} --> {z_dst_ot.shape}"
        )

        self.quantiles_dict_src = compute_quantiles(z_f64[labels])
        self._fit_net(
            z_src_ot=z_src_ot,
            z_dst_ot=z_dst_ot,
            use_gd=kwargs.get("use_gd", False),
            **kwargs,
        )
        self.net = self.net.to(self.dtype).to(self.device)
        self._post_load()

    def _post_load(self) -> None:
        """
        This method should be called after loading the states of the hook.
        So calls must be placed at the end of .fit() and at the end of .load_state_dict().
        """
        super()._post_load()

        # TODO: Move this to some post_load method to avoid doing it every time.
        if self.quantiles_src == "q_all":
            self.quantiles_src = [-1e6, 1e6]
        else:
            self.quantiles_src = copy.deepcopy(
                self.quantiles_dict_src[self.quantiles_src]
            )
            self.quantiles_src[0] = self.quantiles_src[0].view(1, -1)
            self.quantiles_src[1] = self.quantiles_src[1].view(1, -1)

    def forward(self, module, input, output) -> t.Any:
        output_shape = output.shape
        output_orig = output
        if len(output_shape) == 4:  # diffusion
            output = output.mean((2, 3)).view(output_shape[0], 1, output_shape[1])
        z_unit = output

        # Quantile filtering
        ot_mask = (
            (self.quantiles_src[0] < z_unit) & (z_unit < self.quantiles_src[1])
        ).to(z_unit.dtype)

        # Transport the outputs
        z_ot = self.net(z_unit, reverse=False).to(z_unit.dtype)

        # Apply transport with specific strength
        z_ot = self.strength * z_ot + (1 - self.strength) * z_unit

        # Only apply masked outputs
        if len(output_shape) == 4:
            z_ot = z_ot.view(output_shape[0], -1, 1, 1)
            z_ot = output_orig - output.view(output_shape[0], -1, 1, 1) + z_ot
            ot_mask = ot_mask.view(output_shape[0], -1, 1, 1)

        output = ot_mask * z_ot + (1 - ot_mask) * output_orig
        return output


class LinearOTHook(LearnableOTHook):
    def get_net(self) -> t.Type[LinearProj]:
        return LinearProj

    def _fit_net(
        self, z_src_ot: torch.Tensor, z_dst_ot: torch.Tensor, use_gd=False, **kwargs
    ):
        # The linear case
        dim = z_src_ot.shape[-1]
        self.net = LinearProj(dim=dim).to(self.device).to(torch.float32)
        if use_gd:
            self.net = optimize_loop(
                z_src_ot,
                z_dst_ot,
                self.net,
                epochs=kwargs.get("epochs"),
                lr=kwargs.get("learning_rate"),
            )
        else:
            opt_params, opt_extra = self.net.optimize(
                z_src_ot.cpu().numpy(),
                z_dst_ot.cpu().numpy(),
            )

class GaussianOTClustHook(InterventionHook):
    def __init__(
        self,
        module_name: str,
        device: str = None,
        intervention_position: str = "all",
        dtype: torch.dtype = torch.float32,
        strength: float = 1.0,
        std_eps: float = 1e-4,
        quantiles_src: str = "q_all",
        hook_onlymean: bool = True,
        k = 4,
        lmda = "adaptive",
        sim = "adaptive_gaussian",
        **kwargs,
    ):
        """
        Gaussian Optimal Transport hook. Assumes the output of each neuron is independent and Gaussianly distributed.
        Transports from N(mu1, std1) to N(mu2, std2) if hook_forward is True, and N(mu2, std2) to N(mu1, std1) otherwise.

        Args:
            module_name (str): Module name on which the hook is applied.
            device (torch.device): Torch device.
            dtype (str): The torch dtype for this hook.
            intervention_position (str): The position of the token to intervene upon (all, last)
            **kwargs: Any extra arguments, for compatibility with other hooks.
        """
        super().__init__(
            module_name=module_name,
            device=device,
            intervention_position=intervention_position,
            dtype=dtype,
        )
        # print('init hook start')

        self.strength = float(strength)
        self.onlymean = bool(int(hook_onlymean))
        self.std_eps = float(std_eps)
        self.quantiles_src = str(quantiles_src)
        self.k = int(k)
        self.lmda = str(lmda)
        self.sim = str(sim)

        # Register buffer placeholders
        for buffer_name in ["mu1", "mu2", "std1", "std2", "diff", "src_clusters", "tgt_clusters", "P_star"]:
            self.register_buffer(buffer_name, torch.empty(0))

        # Register non-buffer placeholders
        self.quantiles_dict_src = None

    def __str__(self):
        txt = (
            f"GaussianOTClust("
            f"module_name={self.module_name}, "
            f"quantiles_src={self.quantiles_src}, "
            f"strength={self.strength:0.2f}, "
            f"onlymean={self.hook_onlymean}"
            f")"
        )
        return txt

    def state_dict(self, *args, **kwargs) -> t.Dict:
        d = super().state_dict()
        # Merge non-buffers into dict
        d.update(
            {
                "quantiles_dict_src": self.quantiles_dict_src,
            }
        )
        # Merge hook_args required for incremental learning
        d.update(
            {
                "quantiles_src": self.quantiles_src,
            }
        )
        d.update(
            {
                "diff": self.diff
            }
        )
        d.update(
            {
                "src_clusters": self.src_clusters
            }
        )
        d.update(
            {
                "tgt_clusters": self.tgt_clusters
            }
        )
        d.update(
            {
                "P_star": self.P_star
            }
        )
        return d

    def load_state_dict(
        self,
        state_dict: t.Mapping[str, t.Any],
        state_path="",
        strict: bool = True,
        assign: bool = False,
    ) -> None:
        # Fill in registered buffers
        
        all_state_path = state_path.parent
        
        for buffer_name, _ in self.named_buffers():
            
            setattr(
                self,
                buffer_name,
                state_dict[buffer_name].to(self.device).to(self.dtype),
            )
        # breakpoint()
        
        self.quantiles_dict_src = {
            k: [v[0].to(self.device), v[1].to(self.device)]
            for k, v in state_dict["quantiles_dict_src"].items()
        }

        # Non tensor parameters. If these exist in state_dict, they will override the ones passed through constructor.
        # This is useful for hooks learnt for some specific args (eg. incremental hooks).
        for hook_arg in [attr for attr in dir(self) if attr.startswith("hook_")]:
            if hook_arg in state_dict:
                # We know the type, __init__() cas been called.
                vartype = type(getattr(self, hook_arg))
                varval = vartype(state_dict[hook_arg])
                logging.warning(
                    f"Overriding {hook_arg} to {varval} in {self.__class__.__name__}."
                )
                setattr(self, hook_arg, varval)
        self._post_load()

    def fit(self, responses: torch.Tensor, labels=torch.Tensor, **kwargs) -> None:

        # Typecasting to float64 to avoid overflow in the computation of the mean/std.
        z_f64 = responses.to(torch.float64)
        labels = labels.to(torch.bool)
        self.mu1 = torch.mean(z_f64[labels], dim=0).to(self.dtype)
        self.std1 = torch.std(z_f64[labels], dim=0).to(self.dtype)
        self.mu2 = torch.mean(z_f64[~labels], dim=0).to(self.dtype)
        self.std2 = torch.std(z_f64[~labels], dim=0).to(self.dtype)
        self.diff = self.mu2 - self.mu1
        
        # Clustering
        # TODO: Sanity check responses, sanity check labels
        src = z_f64[labels][None, :, :]
        tgt = z_f64[~labels][None, :, :]
        # TODO: Get src and tgt in the form (layer, ex_pt, batch, token, dim) 
        src_clusters, src_labels, src_inertia = get_clusters(src, self.k)
        tgt_clusters, tgt_labels, tgt_inertia = get_clusters(tgt, self.k)
        src_cluster_weights = compute_cluster_weights(src_labels, self.k)
        tgt_cluster_weights = compute_cluster_weights(tgt_labels, self.k)
        cost_matrix = calculate_cost_matrix(src_clusters, tgt_clusters)
        if self.lmda == "adaptive":
            kernel_matrix = calculate_kernel_matrix_adaptive(cost_matrix) 
        else:
            if self.lmda == "sqrtd":
                eps = np.sqrt(z_f64.shape[-1])
            else:
                eps = self.lmda
            kernel_matrix = calculate_kernel_matrix(cost_matrix, eps=eps)
        if self.k > 1:
            self.P_star = sinkhorn_knopp_multi(src_cluster_weights, tgt_cluster_weights, kernel_matrix, self.k, max_iter=1000, tau=1e-6)
        else:
            self.P_star = torch.Tensor([[1.0]])

        # Remove Floating Dimension
        src_clusters = src_clusters.squeeze(0)
        tgt_clusters = tgt_clusters.squeeze(0)
        if self.k > 1:
            self.P_star = self.P_star.squeeze(0)
        self.src_clusters = torch.Tensor(src_clusters).to(self.dtype)
        self.tgt_clusters = torch.Tensor(tgt_clusters).to(self.dtype)
        assert len(self.src_clusters.shape) == 2, "Problem with Source Clusters!"
        assert len(self.tgt_clusters.shape) == 2, "Problem with Target Clusters!"
        assert len(self.P_star.shape) == 2, "Problem with P_star!"

        self.quantiles_dict_src = compute_quantiles(z_f64[labels])
        self._post_load()

    def _post_load(self) -> None:
        """
        This method should be called after loading the states of the hook.
        So calls must be placed at the end of .fit() and at the end of .load_state_dict().
        """
        super()._post_load()

        self.mask = (
            torch.ones_like(
                self.mu1, dtype=torch.bool
            )  # just here to be able to comment/uncomment masks below easily]
            # TODO: We've had this all the time, remove?
            & (self.std1 > self.std_eps)
            & (self.std2 > self.std_eps)
        )

        # Pre-computing things beforehand
        self.mu1_m = self.mu1[self.mask]
        self.mu2_m = self.mu2[self.mask]
        self.std1_m = self.std1[self.mask]
        self.std2_m = self.std2[self.mask]
        
        self.std1_2_m = self.std2_m / self.std1_m

        # TODO: Move this to some post_load method to avoid doing it every time.
        if self.quantiles_src == "q_all":
            self.quantiles_src = [-1e6, 1e6]
        else:
            self.quantiles_src = copy.deepcopy(
                self.quantiles_dict_src[self.quantiles_src]
            )
            self.quantiles_src[0] = self.quantiles_src[0][self.mask].view(1, -1)
            self.quantiles_src[1] = self.quantiles_src[1][self.mask].view(1, -1)

    def forward(self, module, input, output) -> t.Any:
        output_shape = output.shape
        if len(output_shape) == 3:
            output = output.reshape(-1, output_shape[2])
            self.mu1_m = self.mu1_m.view(1, -1)
            self.mu2_m = self.mu2_m.view(1, -1)
            self.std1_m = self.std1_m.view(1, -1)
            self.std2_m = self.std2_m.view(1, -1)
            self.std1_2_m = self.std1_2_m.view(1, -1)
            z_unit = output[:, self.mask]  # B,  U
            z_mask = z_unit
        elif len(output_shape) == 4:
            # Handles image model case
            self.mu1_m = self.mu1_m.view(1, -1, 1, 1)
            self.mu2_m = self.mu2_m.view(1, -1, 1, 1)
            self.std1_m = self.std1_m.view(1, -1, 1, 1)
            self.std2_m = self.std2_m.view(1, -1, 1, 1)
            self.std1_2_m = self.std1_2_m.view(1, -1, 1, 1)
            z_unit = output[:, self.mask]  # B,  U
            z_mask = z_unit.mean(dim=(2, 3), keepdim=False)  # B, C
        else:
            raise NotImplementedError(f"Can't handle tensors with dim=2.")

        # Select outputs that fall in statistics of pooled
        pool_mask = (
            (self.quantiles_src[0] < z_mask) & (z_mask < self.quantiles_src[1])
        ).to(z_unit.dtype)

        # Transport the outputs
        if self.onlymean:
            feature_direction = get_steering_direction2_torch(self.src_clusters, self.tgt_clusters, self.P_star, output, self.sim)
            z_ot = z_unit + 1.0*feature_direction
        else:
            z_ot = self.std1_2_m * (z_unit - self.mu1_m) + self.mu2_m

        # Apply transport with specific strength
        z_ot = self.strength * z_ot + (1 - self.strength) * z_unit

        if len(output_shape) == 4:
            pool_mask = pool_mask[..., None, None]

        # Only apply masked outputs
        output[:, self.mask] = pool_mask * z_ot + (1 - pool_mask) * z_unit
        output = output.view(*output_shape)
        return output

class GaussianOTPCAHook(InterventionHook):
    def __init__(
        self,
        module_name: str,
        device: str = None,
        intervention_position: str = "all",
        dtype: torch.dtype = torch.float32,
        strength: float = 1.0,
        std_eps: float = 1e-4,
        quantiles_src: str = "q_all",
        hook_onlymean: bool = True,
        k = 4,
        k_pcs = 3,
        lmda = "adaptive",
        sim = "adaptive_gaussian",
        **kwargs,
    ):
        """
        Gaussian Optimal Transport hook. Assumes the output of each neuron is independent and Gaussianly distributed.
        Transports from N(mu1, std1) to N(mu2, std2) if hook_forward is True, and N(mu2, std2) to N(mu1, std1) otherwise.

        Args:
            module_name (str): Module name on which the hook is applied.
            device (torch.device): Torch device.
            dtype (str): The torch dtype for this hook.
            intervention_position (str): The position of the token to intervene upon (all, last)
            **kwargs: Any extra arguments, for compatibility with other hooks.
        """
        super().__init__(
            module_name=module_name,
            device=device,
            intervention_position=intervention_position,
            dtype=dtype,
        )

        self.strength = float(strength)
        self.onlymean = bool(int(hook_onlymean))
        self.std_eps = float(std_eps)
        self.quantiles_src = str(quantiles_src)
        self.k = int(k)
        self.k_pcs = int(k_pcs)
        self.lmda = str(lmda)
        self.sim = str(sim)

        # Register buffer placeholders
        for buffer_name in ["mu1", "mu2", "std1", "std2", "diff", "src_clusters", "tgt_clusters", "P_star", "v_bar", "pc_scores", "top_k_pc"]:
            self.register_buffer(buffer_name, torch.empty(0))

        # Register non-buffer placeholders
        self.quantiles_dict_src = None

    def __str__(self):
        txt = (
            f"GaussianOTPCA("
            f"module_name={self.module_name}, "
            f"quantiles_src={self.quantiles_src}, "
            f"strength={self.strength:0.2f}, "
            f"onlymean={self.hook_onlymean}"
            f")"
        )
        return txt

    def state_dict(self, *args, **kwargs) -> t.Dict:
        d = super().state_dict()
        # Merge non-buffers into dict
        d.update(
            {
                "quantiles_dict_src": self.quantiles_dict_src,
            }
        )
        # Merge hook_args required for incremental learning
        d.update(
            {
                "quantiles_src": self.quantiles_src,
            }
        )
        d.update(
            {
                "diff": self.diff
            }
        )
        d.update(
            {
                "src_clusters": self.src_clusters
            }
        )
        d.update(
            {
                "tgt_clusters": self.tgt_clusters
            }
        )
        d.update(
            {
                "P_star": self.P_star
            }
        )
        d.update(
            {
                "v_bar": self.v_bar
            }
        )
        d.update(
            {
                "pc_scores": self.pc_scores
            }
        )
        d.update(
            {
                "top_k_pc": self.top_k_pc
            }
        )
        return d

    def load_state_dict(
        self,
        state_dict: t.Mapping[str, t.Any],
        state_path="",
        strict: bool = True,
        assign: bool = False,
    ) -> None:
        # Fill in registered buffers
        
        all_state_path = state_path.parent
        
        for buffer_name, _ in self.named_buffers():
            
            setattr(
                self,
                buffer_name,
                state_dict[buffer_name].to(self.device).to(self.dtype),
            )
        # breakpoint()
        
        self.quantiles_dict_src = {
            k: [v[0].to(self.device), v[1].to(self.device)]
            for k, v in state_dict["quantiles_dict_src"].items()
        }

        # Non tensor parameters. If these exist in state_dict, they will override the ones passed through constructor.
        # This is useful for hooks learnt for some specific args (eg. incremental hooks).
        for hook_arg in [attr for attr in dir(self) if attr.startswith("hook_")]:
            if hook_arg in state_dict:
                # We know the type, __init__() cas been called.
                vartype = type(getattr(self, hook_arg))
                varval = vartype(state_dict[hook_arg])
                logging.warning(
                    f"Overriding {hook_arg} to {varval} in {self.__class__.__name__}."
                )
                setattr(self, hook_arg, varval)
        self._post_load()

    def fit(self, responses: torch.Tensor, labels=torch.Tensor, **kwargs) -> None:

        # Typecasting to float64 to avoid overflow in the computation of the mean/std.
        z_f64 = responses.to(torch.float64)
        labels = labels.to(torch.bool)
        self.mu1 = torch.mean(z_f64[labels], dim=0).to(self.dtype)
        self.std1 = torch.std(z_f64[labels], dim=0).to(self.dtype)
        self.mu2 = torch.mean(z_f64[~labels], dim=0).to(self.dtype)
        self.std2 = torch.std(z_f64[~labels], dim=0).to(self.dtype)
        self.diff = self.mu2 - self.mu1
        
        # Clustering
        # TODO: Sanity check responses, sanity check labels
        src = z_f64[labels][None, :, :]
        tgt = z_f64[~labels][None, :, :]
        # TODO: Get src and tgt in the form (layer, ex_pt, batch, token, dim) 
        src_clusters, src_labels, src_inertia = get_clusters(src, self.k)
        tgt_clusters, tgt_labels, tgt_inertia = get_clusters(tgt, self.k)
        src_cluster_weights = compute_cluster_weights(src_labels, self.k)
        tgt_cluster_weights = compute_cluster_weights(tgt_labels, self.k)
        cost_matrix = calculate_cost_matrix(src_clusters, tgt_clusters)
        if self.lmda == "adaptive":
            kernel_matrix = calculate_kernel_matrix_adaptive(cost_matrix) 
        else:
            if self.lmda == "sqrtd":
                eps = np.sqrt(z_f64.shape[-1])
            else:
                eps = self.lmda
            kernel_matrix = calculate_kernel_matrix(cost_matrix, eps=eps)
        if self.k > 1:
            self.P_star = sinkhorn_knopp_multi(src_cluster_weights, tgt_cluster_weights, kernel_matrix, self.k, max_iter=1000, tau=1e-6)
        else:
            self.P_star = torch.Tensor([[1.0]])

        src_clusters = src_clusters.squeeze(0)
        tgt_clusters = tgt_clusters.squeeze(0)
        if self.k > 1:
            self.P_star = self.P_star.squeeze(0)

        P_star = self.P_star.float().numpy()
        
        # Computation of Principal Components and Score
        v = tgt_clusters[None, :, :] - src_clusters[:, None, :]
        i_test, j_test = 2, 3
        correct_test = tgt_clusters[j_test] - src_clusters[i_test]
        computed_test = v[i_test, j_test]
        assert np.isclose(correct_test, computed_test).all()

        scaled_v = v * P_star[:, :, None]
        v_bar = np.sum(scaled_v, axis = (0, 1))
        

        # Centered v
        centered_v = v - v_bar[None, None, :]

        # Computation of Weighted Covariance Matrix
        centered_v_reshaped = centered_v.reshape(-1, v_bar.shape[-1])
        P_star_reshaped = P_star.reshape(-1, 1)
        sigma_total_ex_pt = centered_v_reshaped.transpose(1, 0) @ (P_star_reshaped * centered_v_reshaped)

        # EigenDecomposition
        ex_pt_eigdecomp_ex_pt = np.linalg.eigh(sigma_total_ex_pt)

        # Principal_components;
        eigenvecs_ex_pt = ex_pt_eigdecomp_ex_pt.eigenvectors[:, ::-1]
        top_k_pc_ex_pt = eigenvecs_ex_pt[:, :self.k_pcs].transpose().copy()

        # Compute pc scores:
        pc_scores_ex_pt = einops.einsum(top_k_pc_ex_pt, centered_v, "k d, i j d -> k i j")
        

        # Remove Floating Dimension
        
        self.src_clusters = torch.Tensor(src_clusters).to(self.dtype)
        self.tgt_clusters = torch.Tensor(tgt_clusters).to(self.dtype)
        self.v_bar = torch.Tensor(v_bar).to(self.dtype)
        self.top_k_pc = torch.Tensor(top_k_pc_ex_pt).to(self.dtype)
        self.pc_scores = torch.Tensor(pc_scores_ex_pt).to(self.dtype)
        
        assert len(self.src_clusters.shape) == 2, "Problem with Source Clusters!"
        assert len(self.tgt_clusters.shape) == 2, "Problem with Target Clusters!"
        assert len(self.P_star.shape) == 2, "Problem with P_star!"
        assert len(self.v_bar.shape) == 1, "Problem with V bar!"
        assert len(self.top_k_pc.shape) == 2, "Problem with Top K PC!"
        assert len(self.pc_scores.shape) == 3, "Problem with PC Scores!"

        self.quantiles_dict_src = compute_quantiles(z_f64[labels])
        self._post_load()

    def _post_load(self) -> None:
        """
        This method should be called after loading the states of the hook.
        So calls must be placed at the end of .fit() and at the end of .load_state_dict().
        """
        super()._post_load()

        self.mask = (
            torch.ones_like(
                self.mu1, dtype=torch.bool
            )  # just here to be able to comment/uncomment masks below easily]
            # TODO: We've had this all the time, remove?
            & (self.std1 > self.std_eps)
            & (self.std2 > self.std_eps)
        )

        # Pre-computing things beforehand
        self.mu1_m = self.mu1[self.mask]
        self.mu2_m = self.mu2[self.mask]
        self.std1_m = self.std1[self.mask]
        self.std2_m = self.std2[self.mask]
        
        self.std1_2_m = self.std2_m / self.std1_m

        # TODO: Move this to some post_load method to avoid doing it every time.
        if self.quantiles_src == "q_all":
            self.quantiles_src = [-1e6, 1e6]
        else:
            self.quantiles_src = copy.deepcopy(
                self.quantiles_dict_src[self.quantiles_src]
            )
            self.quantiles_src[0] = self.quantiles_src[0][self.mask].view(1, -1)
            self.quantiles_src[1] = self.quantiles_src[1][self.mask].view(1, -1)

    def forward(self, module, input, output) -> t.Any:
        output_shape = output.shape
        if len(output_shape) == 3:
            output = output.reshape(-1, output_shape[2])
            self.mu1_m = self.mu1_m.view(1, -1)
            self.mu2_m = self.mu2_m.view(1, -1)
            self.std1_m = self.std1_m.view(1, -1)
            self.std2_m = self.std2_m.view(1, -1)
            self.std1_2_m = self.std1_2_m.view(1, -1)
            z_unit = output[:, self.mask]  # B,  U
            z_mask = z_unit
        elif len(output_shape) == 4:
            # Handles image model case
            self.mu1_m = self.mu1_m.view(1, -1, 1, 1)
            self.mu2_m = self.mu2_m.view(1, -1, 1, 1)
            self.std1_m = self.std1_m.view(1, -1, 1, 1)
            self.std2_m = self.std2_m.view(1, -1, 1, 1)
            self.std1_2_m = self.std1_2_m.view(1, -1, 1, 1)
            z_unit = output[:, self.mask]  # B,  U
            z_mask = z_unit.mean(dim=(2, 3), keepdim=False)  # B, C
        else:
            raise NotImplementedError(f"Can't handle tensors with dim=2.")

        # Select outputs that fall in statistics of pooled
        pool_mask = (
            (self.quantiles_src[0] < z_mask) & (z_mask < self.quantiles_src[1])
        ).to(z_unit.dtype)

        # Transport the outputs
        if self.onlymean:
            feature_direction = pca_get_steering_direction2_torch(self.src_clusters, self.tgt_clusters, self.P_star, self.v_bar, self.pc_scores, self.top_k_pc, output, self.sim)
            z_ot = z_unit + 1.0*feature_direction
        else:
            z_ot = self.std1_2_m * (z_unit - self.mu1_m) + self.mu2_m

        # Apply transport with specific strength
        z_ot = self.strength * z_ot + (1 - self.strength) * z_unit

        if len(output_shape) == 4:
            pool_mask = pool_mask[..., None, None]

        # Only apply masked outputs
        output[:, self.mask] = pool_mask * z_ot + (1 - pool_mask) * z_unit
        output = output.view(*output_shape)
        return output