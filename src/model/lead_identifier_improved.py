"""Improved LeadIdentifier with median-based baseline removal.

PMcardio insight: using nanmedian instead of nanmean for baseline estimation
is more robust to QRS peak outliers that skew the mean.
"""

from typing import Any

import numpy as np
import torch

from src.model.lead_identifier import LeadIdentifier


class LeadIdentifierImproved(LeadIdentifier):
    """LeadIdentifier with nanmedian baseline removal (Step 1).

    The only change from the parent class is in normalize():
    ``lines - lines.nanmedian(dim=1).values`` instead of ``lines - lines.nanmean(dim=1)``.
    This prevents large QRS peaks from biasing the baseline estimate.
    """

    def normalize(self, lines: torch.Tensor, avg_pixel_per_mm: float, mv_per_mm: float) -> torch.Tensor:
        """Changes the units of the ECG signals from pixels to µV, using median baseline."""
        print(
            f"[DEBUG LeadIdentifierImproved] normalize input: shape={lines.shape}, "
            f"avg_pixel_per_mm={avg_pixel_per_mm:.4f}, mv_per_mm={mv_per_mm}"
        )
        scale = (mv_per_mm / avg_pixel_per_mm) * 1000
        print(f"[DEBUG LeadIdentifierImproved]   scale factor = {scale:.4f}")

        # ---- KEY CHANGE: nanmedian instead of nanmean ----
        baseline = lines.nanmedian(dim=1, keepdim=True).values
        lines = lines - baseline

        lines = lines * scale
        print(
            f"[DEBUG LeadIdentifierImproved]   after scale: value range = "
            f"[{torch.nan_to_num(lines, nan=float('inf')).min():.1f}, "
            f"{torch.nan_to_num(lines, nan=float('-inf')).max():.1f}] uV"
        )

        # Crop to columns where at least `required_valid_samples` leads are valid
        non_nan_samples_per_column = torch.sum(~torch.isnan(lines), dim=0).numpy()
        first_valid_index: int = int(np.argmax(non_nan_samples_per_column >= self.required_valid_samples))
        last_valid_index: int = int(np.argmax(non_nan_samples_per_column[::-1] >= self.required_valid_samples))
        last_valid_index = lines.shape[1] - last_valid_index - 1
        print(
            f"[DEBUG LeadIdentifierImproved]   valid column range: [{first_valid_index}, {last_valid_index}] "
            f"(required_valid_samples={self.required_valid_samples})"
        )
        if first_valid_index <= last_valid_index:
            lines = lines[:, first_valid_index : last_valid_index + 1]
        print(f"[DEBUG LeadIdentifierImproved]   after crop: shape={lines.shape}")

        # Debug per-line stats before interpolation
        for i in range(lines.shape[0]):
            nan_count = torch.isnan(lines[i]).sum().item()
            valid_count = (~torch.isnan(lines[i])).sum().item()
            if valid_count > 0:
                vmin = lines[i][~torch.isnan(lines[i])].min().item()
                vmax = lines[i][~torch.isnan(lines[i])].max().item()
            else:
                vmin, vmax = float("nan"), float("nan")
            print(
                f"[DEBUG LeadIdentifierImproved]   line {i} before interp: "
                f"valid={valid_count}, nan={nan_count}, range=[{vmin:.1f}, {vmax:.1f}]"
            )

        lines = self._interpolate_lines(lines, self.target_num_samples)
        print(f"[DEBUG LeadIdentifierImproved]   after interpolation to {self.target_num_samples}: shape={lines.shape}")

        for i in range(lines.shape[0]):
            nan_count = torch.isnan(lines[i]).sum().item()
            if nan_count > 0:
                print(f"[DEBUG LeadIdentifierImproved]   WARNING: line {i} has {nan_count} NaN after interp!")
            valid = ~torch.isnan(lines[i])
            vmin = lines[i][valid].min().item() if valid.any() else float("nan")
            vmax = lines[i][valid].max().item() if valid.any() else float("nan")
            print(f"[DEBUG LeadIdentifierImproved]   line {i} after interp: range=[{vmin:.1f}, {vmax:.1f}]")

        return lines
