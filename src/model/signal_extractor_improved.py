"""Improved SignalExtractor with PMcardio-inspired post-processing.

Improvements over the base SignalExtractor:
- Binarization + morphology + skeletonization for line extraction (replaces weighted centroid)
- Step 2: Dynamic row detection via Y-axis projection peak finding
- Step 3: Endpoint slope-aware cost matrix + linear extrapolation for gap filling
- Step 4: Beam-search horizontal path tracing + baseline-aware overlap resolution
"""

from collections import defaultdict, deque
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torchvision
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import binary_closing, binary_opening
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from skimage.measure import label

from src.model.signal_extractor import SignalExtractor


class SignalExtractorImproved(SignalExtractor):
    """SignalExtractor with binarization + skeletonization, dynamic row detection,
    slope-aware matching, and beam-search tracing."""

    def __init__(
        self,
        *,
        threshold_sum: float = 10.0,
        threshold_line_in_mask: float = 0.95,
        label_thresh: float = 0.1,
        max_iterations: int = 4,
        split_num_stripes: int = 4,
        candidate_span: int = 10,
        debug: int = 0,
        lam: float = 0.5,
        min_line_width: int = 30,
        # Binarization params
        binarize: bool = True,
        binarize_thresh: float = 0.3,
        morph_open_size: int = 2,
        morph_close_size: int = 3,
        # Step 2
        dynamic_rows: bool = True,
        row_sigma: float = 15.0,
        row_min_distance: int = 50,
        row_prominence: float = 0.05,
        # Step 3
        slope_window: int = 20,
        slope_penalty: float = 15.0,
        max_extrapolation_gap: int = 40,
        # Step 4
        beam_width: int = 3,
    ) -> None:
        super().__init__(
            threshold_sum=threshold_sum,
            threshold_line_in_mask=threshold_line_in_mask,
            label_thresh=label_thresh,
            max_iterations=max_iterations,
            split_num_stripes=split_num_stripes,
            candidate_span=candidate_span,
            debug=debug,
            lam=lam,
            min_line_width=min_line_width,
        )
        # Binarization
        self.binarize = binarize
        self.binarize_thresh = binarize_thresh
        self.morph_open_size = morph_open_size
        self.morph_close_size = morph_close_size
        # Step 2
        self.dynamic_rows = dynamic_rows
        self.row_sigma = row_sigma
        self.row_min_distance = row_min_distance
        self.row_prominence = row_prominence
        # Step 3
        self.slope_window = slope_window
        self.slope_penalty = slope_penalty
        self.max_extrapolation_gap = max_extrapolation_gap
        # Step 4
        self.beam_width = beam_width

    # =========================================================================
    # Core override: binarization + morphology + skeletonization
    # =========================================================================

    def _binarize_and_clean(self, fmap: torch.Tensor) -> npt.NDArray[np.bool_]:
        """Hard-threshold the probability map + morphological cleanup.

        Returns a clean binary mask (True = signal pixel).
        """
        binary = (fmap.numpy() > self.binarize_thresh).astype(np.uint8)
        print(
            f"[DEBUG SignalExtractorImproved] binarize: thresh={self.binarize_thresh}, "
            f"signal_pixels={binary.sum()}/{binary.size} ({binary.sum()/binary.size*100:.2f}%)"
        )

        # Closing first: bridge small horizontal gaps in signal lines
        if self.morph_close_size > 0:
            struct_close = np.ones((1, self.morph_close_size))  # horizontal closing
            binary = binary_closing(binary, structure=struct_close).astype(np.uint8)

        # Opening: remove small noise specks
        if self.morph_open_size > 0:
            struct_open = np.ones((self.morph_open_size, self.morph_open_size))
            binary = binary_opening(binary, structure=struct_open).astype(np.uint8)

        print(
            f"[DEBUG SignalExtractorImproved]   after morph cleanup: {binary.sum()} pixels"
        )
        return binary.astype(np.bool_)

    def _extract_line_from_region(self, fmap: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Override: use median-y of mask pixels per column instead of probability-weighted centroid.

        The mask comes from the parent's soft-threshold connected component labeling,
        so it has the same intact regions. We just change HOW we pick the y-coordinate:
        - Parent: weighted centroid  Σ(prob * y) / Σ(prob) — biased by thick/noisy regions
        - This:   median y of all mask pixels — robust to signal width variation
        """
        if not self.binarize:
            return super()._extract_line_from_region(fmap, mask)

        W = mask.shape[1]
        line = torch.full((W,), float("nan"))

        mask_np = mask.numpy().astype(bool)
        for col in range(W):
            ys = np.where(mask_np[:, col])[0]
            if len(ys) > 0:
                line[col] = float(np.median(ys))

        return line

    def _label_regions(self, fmap: torch.Tensor) -> npt.NDArray[np.int64]:
        """Always use the parent's soft-threshold connected component labeling.

        Binarization only affects _extract_line_from_region (median-y vs centroid).
        Keeping the parent's labeling preserves intact connected components.
        """
        return super()._label_regions(fmap)

    # =========================================================================
    # Step 2: Dynamic row detection
    # =========================================================================

    def _detect_row_positions(self, fmap: torch.Tensor) -> list[tuple[int, int]]:
        """Detect ECG lead rows by projecting the probability map onto the Y axis."""
        H, W = fmap.shape
        y_projection = fmap.sum(dim=1).numpy().astype(float)
        y_smooth = gaussian_filter1d(y_projection, sigma=self.row_sigma)
        y_max = y_smooth.max()
        if y_max < 1e-6:
            return self._uniform_stripes(H)
        y_norm = y_smooth / y_max

        peaks, props = find_peaks(
            y_norm,
            distance=self.row_min_distance,
            prominence=self.row_prominence,
        )
        if len(peaks) < 2:
            print(
                f"[DEBUG SignalExtractorImproved] dynamic row detection found "
                f"{len(peaks)} peaks, falling back to uniform"
            )
            return self._uniform_stripes(H)

        print(f"[DEBUG SignalExtractorImproved] detected {len(peaks)} row centres: {peaks.tolist()}")

        boundaries = [0]
        for i in range(len(peaks) - 1):
            boundaries.append((peaks[i] + peaks[i + 1]) // 2)
        boundaries.append(H)

        stripes = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
        return stripes

    def _uniform_stripes(self, H: int) -> list[tuple[int, int]]:
        n = self.split_num_stripes
        return [(i * H // n, (i + 1) * H // n) for i in range(n)]

    def _extract_candidate_lines(
        self, fmap: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Override: use dynamic row detection when enabled (Step 2)."""
        lab: npt.NDArray[Any] = self._label_regions(fmap)
        zero_mask = lab == 0
        big_offset = lab.max() + 1000

        if self.dynamic_rows:
            stripes = self._detect_row_positions(fmap)
        else:
            H = lab.shape[0]
            stripes = self._uniform_stripes(H)

        for i, (y0, y1) in enumerate(stripes):
            relab: npt.NDArray[np.int64] = label(lab[y0:y1, :] > 0, connectivity=1)  # type: ignore
            lab[y0:y1, :] += big_offset * i + relab
        lab[zero_mask] = 0

        good: list[torch.Tensor] = []
        rejected: list[torch.Tensor] = []
        rej_maps: list[torch.Tensor] = []
        for lid in np.unique(lab):
            if lid == 0:
                continue
            mask = torch.tensor(lab == lid)
            line = self._extract_line_from_region(fmap, mask)
            if self._classify_line(line, mask):
                good.append(line)
            else:
                rejected.append(line)
                rej_maps.append(fmap * mask)

        for line in (*good, *rejected):
            line[line < 5] = float("nan")

        return good, rejected, rej_maps

    # =========================================================================
    # Step 3: Slope-aware cost matrix + endpoint extrapolation
    # =========================================================================

    def _compute_endpoint_slopes(self, lines: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        N = lines.shape[0]
        left_slopes = torch.zeros(N)
        right_slopes = torch.zeros(N)
        window = self.slope_window

        for i in range(N):
            valid = ~torch.isnan(lines[i])
            indices = valid.nonzero(as_tuple=True)[0]
            if len(indices) < 2:
                continue

            left_idx = indices[:window]
            if len(left_idx) >= 2:
                x = left_idx.float()
                y = lines[i, left_idx]
                xm, ym = x.mean(), y.mean()
                var_x = ((x - xm) ** 2).sum()
                if var_x > 1e-8:
                    left_slopes[i] = ((x - xm) * (y - ym)).sum() / var_x

            right_idx = indices[-window:]
            if len(right_idx) >= 2:
                x = right_idx.float()
                y = lines[i, right_idx]
                xm, ym = x.mean(), y.mean()
                var_x = ((x - xm) ** 2).sum()
                if var_x > 1e-8:
                    right_slopes[i] = ((x - xm) * (y - ym)).sum() / var_x

        return left_slopes, right_slopes

    def compute_cost_matrix(
        self, min_coords: torch.Tensor, max_coords: torch.Tensor, W: int, heights: torch.Tensor
    ) -> tuple[npt.NDArray[Any], torch.Tensor]:
        """Override: add slope-consistency penalty (Step 3)."""
        lam = self.lam
        N = min_coords.shape[0]
        min_exp = min_coords.unsqueeze(1).expand(N, N, 2)
        max_exp = max_coords.unsqueeze(0).expand(N, N, 2)

        delta_x = min_exp[..., 0] - max_exp[..., 0]
        wrapped_x = torch.minimum(delta_x.abs(), W - delta_x.abs()) * lam
        wrapped_mask = (W - delta_x.abs()) < delta_x.abs()

        delta_x = torch.where(delta_x < 0, delta_x / lam, delta_x)
        wrapped_x = torch.where(wrapped_mask, wrapped_x, delta_x.abs())

        delta_y = min_exp[..., 1] - max_exp[..., 1]
        delta_y = torch.where(wrapped_mask, delta_y * lam, delta_y)

        distances = wrapped_x.abs() + delta_y.abs()

        heights_norm = (heights - heights.min()) / heights.max().clamp(min=1e-8)
        heights_diff = torch.abs(heights_norm.unsqueeze(1) - heights_norm.unsqueeze(0))

        slope_penalty_matrix = torch.zeros(N, N)
        if hasattr(self, "_left_slopes") and hasattr(self, "_right_slopes"):
            for i in range(N):
                for j in range(N):
                    slope_diff = abs(float(self._left_slopes[i]) - float(self._right_slopes[j]))
                    slope_penalty_matrix[i, j] = slope_diff

        cost_matrix: npt.NDArray[Any] = (
            distances * (1 + heights_diff * 30 + slope_penalty_matrix * self.slope_penalty)
        ).numpy()
        return cost_matrix, wrapped_mask

    def extract_graph_params(self, lines: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        self._left_slopes, self._right_slopes = self._compute_endpoint_slopes(lines)
        return super().extract_graph_params(lines)

    def _extrapolate_endpoints(self, lines: torch.Tensor) -> torch.Tensor:
        lines = lines.clone()
        left_slopes, right_slopes = self._compute_endpoint_slopes(lines)
        max_gap = self.max_extrapolation_gap

        for i in range(lines.shape[0]):
            valid = ~torch.isnan(lines[i])
            indices = valid.nonzero(as_tuple=True)[0]
            if len(indices) < 2:
                continue
            first, last = int(indices[0].item()), int(indices[-1].item())

            if first > 0 and first <= max_gap:
                slope = float(left_slopes[i])
                y0 = float(lines[i, first])
                for x in range(first - 1, -1, -1):
                    lines[i, x] = y0 + slope * (x - first)

            W = lines.shape[1]
            if last < W - 1 and (W - 1 - last) <= max_gap:
                slope = float(right_slopes[i])
                y0 = float(lines[i, last])
                for x in range(last + 1, W):
                    lines[i, x] = y0 + slope * (x - last)

        return lines

    # =========================================================================
    # Step 4: Beam-search tracing + baseline-aware overlap resolution
    # =========================================================================

    def _trace_horizontal_path(self, img_arr: npt.NDArray[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        img = torch.tensor(img_arr, dtype=torch.float32)
        blurry = img.clone()
        blurry += (torch.linspace(-1, 1, blurry.shape[0]) ** 6).unsqueeze(1)
        H, W = img.shape
        K = min(self.beam_width, H)

        if K <= 1 or W < 2:
            return super()._trace_horizontal_path(img_arr)

        beams: list[tuple[float, list[int]]] = [(0.0, [H // 2])]

        for x in range(1, W):
            new_beams: list[tuple[float, list[int]]] = []
            for score, path in beams:
                prev = path[-1]
                candidates = torch.arange(
                    max(0, prev - self.candidate_span),
                    min(H, prev + self.candidate_span + 1),
                )
                vals = self._get_pixel_vals(blurry, candidates, x)
                top_k = min(K, len(candidates))
                _, best_indices = torch.topk(vals, top_k, largest=False)
                for idx in best_indices:
                    y = int(candidates[idx].item())
                    new_score = score + float(vals[idx])
                    new_beams.append((new_score, path + [y]))

            new_beams.sort(key=lambda b: b[0])
            beams = new_beams[:K]

        best_path = beams[0][1]

        result_img = img.clone()
        for i in range(len(best_path) - 1):
            lo, hi = sorted((best_path[i], best_path[i + 1]))
            result_img[max(0, lo - 1) : min(H, hi + 2), i] = 0

        return torch.tensor(best_path), result_img

    def merge_components(
        self, lines: torch.Tensor, components: list[list[int]]
    ) -> tuple[list[torch.Tensor], list[float]]:
        merged: list[torch.Tensor] = []
        overlaps: list[float] = []
        for group in components:
            group_lines = lines[torch.tensor(group)]
            valid_mask = ~torch.isnan(group_lines)
            merged_line = torch.full((group_lines.shape[1],), float("nan"))
            overlap = valid_mask.sum(0)
            overlaps.append(float(overlap[overlap > 0].float().mean().item()))

            all_valid = group_lines[valid_mask]
            baseline = float(all_valid.mean().item()) if len(all_valid) > 0 else 0.0

            for col in range(group_lines.shape[1]):
                valid_values = group_lines[:, col][~torch.isnan(group_lines[:, col])]
                if len(valid_values) == 0:
                    continue
                if len(valid_values) == 1:
                    merged_line[col] = valid_values[0]
                else:
                    dists = torch.abs(valid_values - baseline)
                    merged_line[col] = valid_values[torch.argmin(dists)]

            merged.append(merged_line)
        return merged, overlaps

    # =========================================================================
    # Override main match_and_merge to include extrapolation (Step 3)
    # =========================================================================

    def match_and_merge_lines(self, lines: torch.Tensor) -> tuple[list[torch.Tensor], list[float]]:
        lines = self.preprocess_lines(lines)
        if self.debug:
            self.plot_lines(lines, "Preprocessed Lines (Improved)")

        lines = self._extrapolate_endpoints(lines)

        min_coords, max_coords, heights, W = self.extract_graph_params(lines)
        cost_matrix, wrapped_mask = self.compute_cost_matrix(min_coords, max_coords, W, heights)

        row_ind, col_ind = self.match_lines(cost_matrix)

        valid_mask = ~wrapped_mask[row_ind, col_ind]
        row_ind, col_ind = row_ind[valid_mask], col_ind[valid_mask]

        graph = self.build_match_graph(row_ind, col_ind)
        components = self.get_connected_components(graph)

        merged_lines, overlaps = self.merge_components(lines, components)
        filtered_lines = [line for line in merged_lines if torch.sum(~torch.isnan(line)) >= W // 5]

        if self.debug:
            self.plot_graph(min_coords, max_coords, row_ind, col_ind)

        return filtered_lines, overlaps
