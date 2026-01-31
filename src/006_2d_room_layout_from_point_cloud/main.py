import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from plyfile import PlyData
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.ndimage import rotate
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from tqdm import tqdm

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    'grid_size': 256,
    'batch_size': 16,
    'train_steps': 2000,
    'lr': 0.0005,
    'ply_file': "manhattan_apartment.ply",
    # 'ply_file': "40958733_3dod_mesh.ply",
    'weights_file': "room_detector.pth",
}


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super(FocalLoss, self).__init__()
        # alpha: weighting factor for the rare class (corners).
        #        Higher = focus more on positive labels.
        # gamma: focusing parameter.
        #        Higher = punish "hard" mistakes (false negatives) more.
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # 1. Standard BCE (Per-pixel loss)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # 2. Convert to probabilities (pt) to calculate focusing factor
        pt = torch.exp(-bce_loss)

        # 3. Apply Focal Loss Formula
        # Loss = -alpha * (1 - pt)^gamma * log(pt)
        f_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        return f_loss.mean()


# ==========================================
# 2. SYNTHETIC DATA GENERATOR (Sim2Real)
# ==========================================
class SyntheticRoomGenerator(IterableDataset):
    def __init__(self, grid_size=512):
        self.grid = grid_size
        self.padding = 20
        self.struct = generate_binary_structure(2, 2)

    def _get_scanner_state(self, floor_mask):
        """Generates the parameters for a 'live' partial scan."""
        y_floor, x_floor = np.where(floor_mask)
        if len(y_floor) > 0:
            idx = random.randint(0, len(y_floor) - 1)
            scanner_pos = np.array([y_floor[idx], x_floor[idx]])
        else:
            scanner_pos = np.array([self.grid // 2, self.grid // 2])

        return {
            'scanner_pos': scanner_pos,
            'is_partial_fov': random.random() > 0.3,
            'fov_angle': random.uniform(np.pi / 2, 1.8 * np.pi),
            'start_angle': random.uniform(0, 2 * np.pi),
            'max_range': random.uniform(self.grid // 2.5, self.grid * 1.1)
        }

    def _apply_visibility_culling(self, mask, state):
        """Filters a mask based on scanner visibility state."""
        y_idx, x_idx = np.where(mask > 0)
        if len(y_idx) == 0:
            return np.zeros_like(mask)

        dy = y_idx - state['scanner_pos'][0]
        dx = x_idx - state['scanner_pos'][1]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        angles = np.arctan2(dy, dx)

        valid = dist < state['max_range']
        if state['is_partial_fov']:
            norm_angles = (angles - state['start_angle']) % (2 * np.pi)
            valid &= (norm_angles < state['fov_angle'])

        culled_mask = np.zeros_like(mask, dtype=float)
        culled_mask[y_idx[valid], x_idx[valid]] = mask[y_idx[valid], x_idx[valid]]
        return culled_mask

    def generate_sample(self):
        # 1. Base Floor Plan
        floor_mask = self._generate_random_floor_mask()

        # 2. Scanner State (Moved Up)
        state = self._get_scanner_state(floor_mask)

        # 3. Generate Full Targets
        dilated = binary_dilation(floor_mask, structure=self.struct)
        full_edge_mask = (dilated ^ floor_mask).astype(float)
        full_corner_mask = self._detect_corners(floor_mask, radius=3)

        # 4. Apply Culling to Targets (GT matches Input)
        edge_mask = self._apply_visibility_culling(full_edge_mask, state)
        corner_mask = self._apply_visibility_culling(full_corner_mask, state)

        # 5. Simulate Lidar using the same state
        input_map = self._simulate_lidar(floor_mask, edge_mask, state)

        # 6. Global Rotation Augmentation
        angle = random.uniform(-180, 180)
        input_map = rotate(input_map, angle, reshape=False, order=1)
        edge_mask = rotate(edge_mask, angle, reshape=False, order=0)
        corner_mask = rotate(corner_mask, angle, reshape=False, order=0)

        # 7. Final Output Tensors
        input_map = np.clip(input_map, 0, 1)[np.newaxis, :, :]
        targets = np.stack([edge_mask, corner_mask], axis=0)

        return torch.from_numpy(input_map).float(), torch.from_numpy(targets).float()

    def _generate_random_floor_mask(self):
        while True:
            mask = np.zeros((self.grid, self.grid), dtype=bool)
            num_rects = random.choices([2, 3, 4, 5], weights=[0.3, 0.4, 0.2, 0.1])[0]
            prev_rect = None
            valid_room = True

            for i in range(num_rects):
                w, h = random.randint(40, 180), random.randint(40, 180)
                if i == 0:
                    x = random.randint(self.padding, self.grid - self.padding - w)
                    y = random.randint(self.padding, self.grid - self.padding - h)
                else:
                    px, py, pw, ph = prev_rect
                    x = random.randint(px, px + pw) - w // 2
                    y = random.randint(py, py + ph) - h // 2

                if (x < self.padding or x + w > self.grid - self.padding or
                        y < self.padding or y + h > self.grid - self.padding):
                    valid_room = False;
                    break

                mask[y:y + h, x:x + w] = True
                prev_rect = (x, y, w, h)

            if valid_room: return mask

    def _detect_corners(self, mask, radius=3):
        corners = np.zeros_like(mask, dtype=float)
        img = mask.astype(int)
        tl, tr, bl, br = img[:-1, :-1], img[:-1, 1:], img[1:, :-1], img[1:, 1:]
        sums = tl + tr + bl + br
        is_corner = (sums == 1) | (sums == 3)
        y_idxs, x_idxs = np.where(is_corner)

        yy, xx = np.ogrid[:self.grid, :self.grid]
        for y, x in zip(y_idxs, x_idxs):
            dist_sq = (xx - x) ** 2 + (yy - y) ** 2
            corners[dist_sq <= radius ** 2] = 1.0
        return corners

    def _simulate_lidar(self, floor_mask, visible_edge_mask, state):
        input_map = np.zeros((self.grid, self.grid), dtype=float)
        scan_density = random.uniform(0.5, 3.0)

        # --- A. WALLS (Using Culled Edge Mask) ---
        y_walls, x_walls = np.where(visible_edge_mask > 0)
        if len(y_walls) > 0:
            wall_presence = np.ones(len(y_walls))
            num_gaps = random.randint(2, 5)
            for _ in range(num_gaps):
                idx = random.randint(0, len(y_walls))
                gap = random.randint(20, 80)
                wall_presence[idx: idx + gap] *= random.uniform(0.1, 0.5)

            wall_repeats = (random.randint(15, 30) * scan_density * wall_presence).astype(int)
            active = wall_repeats > 0
            ys = np.repeat(y_walls[active], wall_repeats[active]).astype(float)
            xs = np.repeat(x_walls[active], wall_repeats[active]).astype(float)
            ys += np.random.normal(0, 0.7, len(ys))
            xs += np.random.normal(0, 0.7, len(xs))

            valid = (xs >= 0) & (xs < self.grid) & (ys >= 0) & (ys < self.grid)
            np.add.at(input_map, (ys[valid].astype(int), xs[valid].astype(int)), 1.0)

        # --- B. FLOOR NOISE (Culled) ---
        y_f_all, x_f_all = np.where(floor_mask)
        dist_f = np.sqrt((y_f_all - state['scanner_pos'][0]) ** 2 + (x_f_all - state['scanner_pos'][1]) ** 2)
        vis_f = dist_f < state['max_range']
        if state['is_partial_fov']:
            ang_f = (np.arctan2(y_f_all - state['scanner_pos'][0], x_f_all - state['scanner_pos'][1]) - state[
                'start_angle']) % (2 * np.pi)
            vis_f &= (ang_f < state['fov_angle'])

        y_f, x_f = y_f_all[vis_f], x_f_all[vis_f]
        if len(y_f) > 0:
            # Increase coverage and density
            coverage = random.uniform(0.3, 0.6)
            n_pix = int(len(y_f) * coverage)
            idx = np.random.choice(len(y_f), n_pix, replace=False)

            # Increase the number of points per floor pixel (from 2 to 5-10)
            repeats = random.randint(5, 10)
            ys_n = np.repeat(y_f[idx], repeats).astype(float)
            xs_n = np.repeat(x_f[idx], repeats).astype(float)

            # Add varying jitter to create "blobs" rather than single points
            ys_n += np.random.normal(0, 2.0, len(ys_n))
            xs_n += np.random.normal(0, 2.0, len(xs_n))

            v = (xs_n >= 0) & (xs_n < self.grid) & (ys_n >= 0) & (ys_n < self.grid)

            # Use a higher value before the log1p normalization
            # Adding a higher weight here makes these pixels 'brighter' in the final input
            noise_value = random.uniform(0.1, 1.0)

            np.add.at(input_map, (ys_n[v].astype(int), xs_n[v].astype(int)), noise_value)

        # --- C. FLARES & GHOSTS ---
        if random.random() > 0.2:  # Flare from scanner
            angle = random.uniform(0, 2 * np.pi)
            t = np.linspace(0, random.randint(50, 200), 100)
            fx = (state['scanner_pos'][1] + t * np.cos(angle)).astype(int)
            fy = (state['scanner_pos'][0] + t * np.sin(angle)).astype(int)
            v = (fx >= 0) & (fx < self.grid) & (fy >= 0) & (fy < self.grid)
            input_map[fy[v], fx[v]] += 0.5

        input_map = np.clip(input_map, 0, 15)
        input_map = np.log1p(input_map)
        input_map /= input_map.max() if input_map.max() > 0 else 1.0
        return input_map

    def __iter__(self):
        while True: yield self.generate_sample()


# ==========================================
# 3. NEURAL NETWORK
# ==========================================
class SpatialAttention(nn.Module):
    """
    Highlights 'interesting' areas by looking at channel-wise max and average.
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(combined))


class EdgeCornerDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared Encoder
        self.enc1 = self.conv_block(1, 16)
        self.enc2 = self.conv_block(16, 32)
        self.enc3 = self.conv_block(32, 64)

        # Shared Decoder
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = self.conv_block(64 + 32, 32)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.conv_block(32 + 16, 16)

        # Attention Gate: Helps focus the heads on specific spatial features
        self.attention = SpatialAttention()

        # Head 1: Wall Detection (Standard 3x3 kernels)
        self.edge_head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )

        # Head 2: Corner Detection (Uses Dilation to see "L" or "T" junctions)
        # Dilated convolutions expand the area the network sees without losing resolution.

        self.corner_head = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(F.max_pool2d(x1, 2))
        x3 = self.enc3(F.max_pool2d(x2, 2))

        # Decoder
        u2 = self.up2(x3)
        u2 = torch.cat([u2, x2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, x1], dim=1)
        d1 = self.dec1(u1)

        # Apply Attention to the decoded features before splitting
        att_map = self.attention(d1)
        d1_refined = d1 * att_map

        # Split Output
        edge_out = self.edge_head(d1_refined)
        corner_out = self.corner_head(d1_refined)

        return torch.cat([edge_out, corner_out], dim=1)


# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train_network(model, steps=500):
    print(f"--- Starting Training ({steps} steps) ---")
    dataset = SyntheticRoomGenerator(grid_size=CONFIG['grid_size'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])

    criterion_wall = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
    criterion_corner = FocalLoss(alpha=0.9, gamma=3.0)

    plt.ion()
    fig, axes = plt.subplots(2, 3, figsize=(20, 5))

    dummy = np.zeros((CONFIG['grid_size'], CONFIG['grid_size']))
    im_input = axes[0][0].imshow(dummy, cmap='inferno')
    im_targ_edg = axes[0][1].imshow(dummy, cmap='gray')  # Target Edges
    im_targ_cor = axes[0][2].imshow(dummy, cmap='magma')  # Target Corners
    im_pred_edg = axes[1][1].imshow(dummy, cmap='magma')  # Pred Corners
    im_pred_cor = axes[1][2].imshow(dummy, cmap='magma')  # Pred Corners

    axes[0][0].set_title("Input (Lidar)")
    axes[0][1].set_title("Target Edges")
    axes[0][2].set_title("Target Corner")
    axes[1][1].set_title("Pred Edges")
    axes[1][2].set_title("Pred Corner")

    fig.suptitle("Training Progress", fontsize=16)
    for ax in axes.flatten(): ax.axis('off')
    plt.tight_layout()
    plt.show()

    model.train()
    data_iter = iter(dataloader)
    pbar = tqdm(range(steps))
    corner_multiplier = 10.0

    for i in pbar:
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            inputs, targets = next(data_iter)

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        pred_edge = outputs[:, 0:1, :, :]
        pred_corner = outputs[:, 1:2, :, :]

        target_edge = targets[:, 0:1, :, :]
        target_corner = targets[:, 1:2, :, :]

        loss_e = criterion_wall(pred_edge, target_edge)
        loss_c = criterion_corner(pred_corner, target_corner)
        total_loss = loss_e + (corner_multiplier * loss_c)

        total_loss.backward()
        optimizer.step()

        # 2. Update all 4 visualization columns
        if i % 4 == 0:
            img_in = inputs[0, 0].cpu().numpy()
            tar_e = target_edge[0, 0].cpu().numpy()
            tar_c = target_corner[0, 0].cpu().numpy()
            prd_e = torch.sigmoid(pred_edge[0, 0]).detach().cpu().numpy()
            prd_c = torch.sigmoid(pred_corner[0, 0]).detach().cpu().numpy()

            im_input.set_data(img_in)
            im_targ_edg.set_data(tar_e)
            im_targ_cor.set_data(tar_c)
            im_pred_edg.set_data(prd_e)
            im_pred_cor.set_data(prd_c)

            # Ensure intensity ranges are correct for visibility
            im_input.set_clim(vmin=img_in.min(), vmax=img_in.max())
            im_targ_edg.set_clim(vmin=tar_e.min(), vmax=tar_e.max())
            im_targ_cor.set_clim(vmin=tar_c.min(), vmax=tar_c.max())
            im_pred_edg.set_clim(vmin=prd_e.min(), vmax=prd_e.max())
            im_pred_cor.set_clim(vmin=prd_c.min(), vmax=prd_c.max())

            fig.suptitle(f"Step {i + 1}/{steps}", fontsize=16)

            fig.canvas.draw()
            fig.canvas.flush_events()

        pbar.set_description(f"Loss: {total_loss.item():.4f} (E:{loss_e.item():.2f} C:{loss_c.item():.2f})")

    plt.ioff()
    plt.close(fig)
    print("--- Training Complete ---")
    torch.save(model.state_dict(), CONFIG['weights_file'])


# ==========================================
# 5. REAL DATA PROCESSING
# ==========================================
def estimate_up_vector(points):
    """
    Uses PCA to find the 'Up' vector.
    Assumption: The floor is the dominant plane, so the direction of
    LEAST variance (smallest eigenvalue) is the normal to the floor.
    """
    # 1. Center the data (subset for speed)
    # taking a random subset of 10k points is enough and faster
    if len(points) > 10000:
        idx = np.random.choice(len(points), 10000, replace=False)
        subset = points[idx]
    else:
        subset = points

    mean = np.mean(subset, axis=0)
    centered = subset - mean

    # 2. Compute Covariance Matrix and PCA
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # 3. The 'Up' vector is the eigenvector with the SMALLEST eigenvalue
    # (The direction where the data is flattest)
    up_vector = eigenvectors[:, 0]

    return up_vector


def load_and_project_ply(filepath, grid_size=256, auto_orient=True):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PLY file not found: {filepath}")

    # 1. Load Data
    plydata = PlyData.read(filepath)
    vertex = plydata['vertex']
    points = np.column_stack([vertex['x'], vertex['y'], vertex['z']])

    # 2. Automatic Up-Vector Alignment
    if auto_orient:
        estimated_up = estimate_up_vector(points)
        print(f"Estimated Up-Vector: {estimated_up}")

        target_up = np.array([0.0, 0.0, 1.0])

        # Check alignment direction (points usually go UP, so dot product should be positive)
        # If negative, our normal is pointing down, so flip it.
        if np.dot(estimated_up, target_up) < 0:
            # This heuristic assumes the original scan isn't upside down relative to world Z
            # If strictly PCA, sign is arbitrary. Visual verification might be needed.
            pass

        rot, _ = R.align_vectors([target_up], [estimated_up])
        points = rot.apply(points)

    # 4. Project (Drop Z from SLICED points)
    x = points[:, 0]
    y = points[:, 1]

    # Normalize
    padding = 0.5
    min_x, max_x = x.min() - padding, x.max() + padding
    min_y, max_y = y.min() - padding, y.max() + padding

    norm_x = (x - min_x) / (max_x - min_x)
    norm_y = (y - min_y) / (max_y - min_y)

    img_x = (norm_x * (grid_size - 1)).astype(int)
    img_y = (norm_y * (grid_size - 1)).astype(int)

    # Rasterize
    H, _, _ = np.histogram2d(img_y, img_x, bins=grid_size, range=[[0, grid_size], [0, grid_size]])

    grid = np.log1p(H)
    if grid.max() > 0: grid /= grid.max()

    return torch.from_numpy(grid).float().unsqueeze(0).unsqueeze(0)


# ==========================================
# 6. MAIN EXECUTION
# ==========================================
def main():
    model = EdgeCornerDetector().to(device)

    # ... (Loading Logic Same as Before) ...
    if not os.path.exists(CONFIG['weights_file']):
        print("Training...")
        train_network(model, steps=CONFIG['train_steps'])
    else:
        print("Loading weights...")
        model.load_state_dict(torch.load(CONFIG['weights_file'], weights_only=True))

    print(f"Processing {CONFIG['ply_file']}...")
    real_input = load_and_project_ply(CONFIG['ply_file'], CONFIG['grid_size'], auto_orient=True)
    real_input = real_input.to(device)

    # 3. INFERENCE
    model.eval()
    with torch.no_grad():
        # IMPORTANT: Since we trained with Logits, we must apply Sigmoid here to get 0.0-1.0
        logits = model(real_input)
        output = torch.sigmoid(logits)

    # 4. VISUALIZE (With Thresholding)
    input_img = real_input[0, 0].cpu().numpy()
    pred_edges = output[0, 0].cpu().numpy()
    pred_corners = output[0, 1].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].imshow(input_img, cmap='inferno')
    axes[0].set_title("1. Input")

    axes[1].imshow(pred_edges, cmap='gray')
    axes[1].set_title("2. Predicted Walls")

    axes[2].imshow(pred_corners, cmap='magma')
    axes[2].set_title("3. Predicted Corners")

    for ax in axes: ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
