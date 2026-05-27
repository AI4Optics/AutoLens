"""Binary2 phase on a plane surface."""

import torch

from ..config import EPSILON
from .phase import Phase


class Binary2Phase(Phase):
    """Binary2 phase on a plane surface."""

    def __init__(
        self,
        r,
        d,
        order2=0.0,
        order4=0.0,
        order6=0.0,
        order8=0.0,
        order10=0.0,
        order12=0.0,
        order14=0.0,
        order16=0.0,
        norm_radii=None,
        mat2="air",
        pos_xy=None,
        vec_local=None,
        is_square=False,
        device="cpu",
    ):
        if pos_xy is None:
            pos_xy = [0.0, 0.0]
        if vec_local is None:
            vec_local = [0.0, 0.0, 1.0]
        super().__init__(
            r=r,
            d=d,
            norm_radii=norm_radii,
            mat2=mat2,
            pos_xy=pos_xy,
            vec_local=vec_local,
            is_square=is_square,
            device=device,
        )

        # Initialize polynomial coefficients
        self.order2 = torch.tensor(order2)
        self.order4 = torch.tensor(order4)
        self.order6 = torch.tensor(order6)
        self.order8 = torch.tensor(order8)
        self.order10 = torch.tensor(order10)
        self.order12 = torch.tensor(order12)
        self.order14 = torch.tensor(order14)
        self.order16 = torch.tensor(order16)

        self.to(device)
        self.init_param_model()

    @classmethod
    def init_from_dict(cls, surf_dict):
        """Initialize Binary2 phase surface from dictionary."""
        mat2 = surf_dict.get("mat2", "air")
        norm_radii = surf_dict.get("norm_radii", None)
        obj = cls(
            r=surf_dict["r"],
            d=surf_dict["d"],
            order2=surf_dict.get("order2", 0.0),
            order4=surf_dict.get("order4", 0.0),
            order6=surf_dict.get("order6", 0.0),
            order8=surf_dict.get("order8", 0.0),
            order10=surf_dict.get("order10", 0.0),
            order12=surf_dict.get("order12", 0.0),
            order14=surf_dict.get("order14", 0.0),
            order16=surf_dict.get("order16", 0.0),
            norm_radii=norm_radii,
            mat2=mat2,
        )
        return obj

    def init_param_model(self):
        """Initialize Binary2 parameters."""
        self.param_model = "binary2"

    def phi(self, x, y):
        """Reference phase map at design wavelength."""
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii
        r2 = x_norm * x_norm + y_norm * y_norm + EPSILON

        # Horner's method for phi(r^2) = sum_k order_{2k} * r^(2k), k = 1..8.
        phi = r2 * (
            self.order2
            + r2 * (
                self.order4
                + r2 * (
                    self.order6
                    + r2 * (
                        self.order8
                        + r2 * (
                            self.order10
                            + r2 * (
                                self.order12
                                + r2 * (self.order14 + r2 * self.order16)
                            )
                        )
                    )
                )
            )
        )

        phi = torch.remainder(phi, 2 * torch.pi)
        return phi

    def dphi_dxy(self, x, y):
        """Calculate phase derivatives (dphi/dx, dphi/dy) for given points."""
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii
        r2 = x_norm * x_norm + y_norm * y_norm + EPSILON

        # d/dr2 of polynomial, then chain rule: dphi/dx = dphi/dr2 * 2*x_norm / norm_radii
        # Horner's: o2 + r2*(2*o4 + r2*(3*o6 + ... + r2*(7*o14 + r2*8*o16)))
        dphidr2 = (
            self.order2
            + r2 * (
                2 * self.order4
                + r2 * (
                    3 * self.order6
                    + r2 * (
                        4 * self.order8
                        + r2 * (
                            5 * self.order10
                            + r2 * (
                                6 * self.order12
                                + r2 * (7 * self.order14 + r2 * 8 * self.order16)
                            )
                        )
                    )
                )
            )
        )
        dphidx = dphidr2 * 2 * x_norm / self.norm_radii
        dphidy = dphidr2 * 2 * y_norm / self.norm_radii

        return dphidx, dphidy

    def get_optimizer_params(self, lrs=[1e-4, 1e-2], optim_mat=False):
        """Generate optimizer parameters with per-order lr scaling.

        The gradient of phi w.r.t. order_{2n} scales as (r/norm_radii)^{2n}.
        Dividing the lr by this factor keeps the effective phase change per
        step approximately constant across all polynomial orders.
        """
        params = []

        # Optimize position
        self.d.requires_grad = True
        params.append({"params": [self.d], "lr": lrs[0]})

        # Optimize polynomial coefficients with r-normalised learning rates
        r_norm = self.r / self.norm_radii
        lr_base = lrs[1]

        for n, attr in enumerate(
            [
                "order2",
                "order4",
                "order6",
                "order8",
                "order10",
                "order12",
                "order14",
                "order16",
            ],
            start=1,
        ):
            p = getattr(self, attr)
            p.requires_grad = True
            order = 2 * n  # 2, 4, 6, 8, 10, 12, 14, 16
            lr_coeff = lr_base / r_norm**order
            params.append({"params": [p], "lr": lr_coeff})

        # We do not optimize material parameters for phase surface.
        assert optim_mat is False, (
            "Material parameters are not optimized for phase surface."
        )

        return params

    def save_ckpt(self, save_path="./binary2_doe.pth"):
        """Save Binary2 DOE parameters."""
        torch.save(
            {
                "param_model": self.param_model,
                "order2": self.order2.clone().detach().cpu(),
                "order4": self.order4.clone().detach().cpu(),
                "order6": self.order6.clone().detach().cpu(),
                "order8": self.order8.clone().detach().cpu(),
                "order10": self.order10.clone().detach().cpu(),
                "order12": self.order12.clone().detach().cpu(),
                "order14": self.order14.clone().detach().cpu(),
                "order16": self.order16.clone().detach().cpu(),
            },
            save_path,
        )

    def load_ckpt(self, load_path="./binary2_doe.pth"):
        """Load Binary2 DOE parameters.

        Older checkpoints may not contain order14 / order16; those default to 0.
        """
        self.diffraction = True
        ckpt = torch.load(load_path, weights_only=False)
        self.param_model = ckpt["param_model"]
        self.order2 = ckpt["order2"].to(self.device)
        self.order4 = ckpt["order4"].to(self.device)
        self.order6 = ckpt["order6"].to(self.device)
        self.order8 = ckpt["order8"].to(self.device)
        self.order10 = ckpt["order10"].to(self.device)
        self.order12 = ckpt["order12"].to(self.device)
        self.order14 = ckpt.get("order14", torch.tensor(0.0)).to(self.device)
        self.order16 = ckpt.get("order16", torch.tensor(0.0)).to(self.device)

    def zmx_str(self, surf_idx, d_next):
        """Return Zemax BINARY_2 surface string."""
        # XDAT1 = number of terms, XDAT2 = norm radius, XDAT3..10 = coefficients
        mat_str = ""
        if self.mat2.get_name() != "air":
            mat_str = f"    GLAS ___BLANK 1 0 {self.mat2.n} {self.mat2.V}\n"

        zmx_str = f"""SURF {surf_idx}
    TYPE BINARY_2
    CURV 0.0
    DISZ {d_next.item()}
{mat_str}    PARM 0 1
    PARM 1 0
    PARM 2 0
    PARM 3 0
    PARM 4 0
    PARM 5 0
    PARM 6 0
    PARM 7 0
    PARM 8 0
    XDAT 1 8.000000000000E+00 0 0 1.000000000000E+00 0.000000000000E+00 0 ""
    XDAT 2 {self.norm_radii:.12E} 0 0 1.000000000000E+00 0.000000000000E+00 0 ""
    XDAT 3 {self.order2.item():.12E} 0 0 1.000000000000E+00 0.000000000000E+00 0 ""
    XDAT 4 {self.order4.item():.12E} 0 0 1.000000000000E+00 0.000000000000E+00 0 ""
    XDAT 5 {self.order6.item():.12E} 0 0 1.000000000000E+00 0.000000000000E+00 0 ""
    XDAT 6 {self.order8.item():.12E} 0 0 1.000000000000E+00 0.000000000000E+00 0 ""
    XDAT 7 {self.order10.item():.12E} 0 0 1.000000000000E+00 0.000000000000E+00 0 ""
    XDAT 8 {self.order12.item():.12E} 0 0 1.000000000000E+00 0.000000000000E+00 0 ""
    XDAT 9 {self.order14.item():.12E} 0 0 1.000000000000E+00 0.000000000000E+00 0 ""
    XDAT 10 {self.order16.item():.12E} 0 0 1.000000000000E+00 0.000000000000E+00 0 ""
    DIAM {self.r} 1 0 0 1 ""
"""
        return zmx_str

    def surf_dict(self):
        """Return surface parameters."""
        surf_dict = {
            "type": self.__class__.__name__,
            "r": self.r,
            "is_square": self.is_square,
            "param_model": self.param_model,
            "order2": round(self.order2.item(), 4),
            "order4": round(self.order4.item(), 4),
            "order6": round(self.order6.item(), 4),
            "order8": round(self.order8.item(), 4),
            "order10": round(self.order10.item(), 4),
            "order12": round(self.order12.item(), 4),
            "order14": round(self.order14.item(), 4),
            "order16": round(self.order16.item(), 4),
            "norm_radii": round(self.norm_radii, 4),
            "d": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
        }
        return surf_dict
