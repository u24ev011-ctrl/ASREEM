"""
EcoFuse-µNet Kaggle Training Script
Paste this into Kaggle notebook cells. Requires: ecofuse_model.py content above it.
Dataset: imtkaggleteam/microplastic-dataset-for-computer-vision
"""
import os, cv2, math, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ── Dataset ──────────────────────────────────────────────────────────────────
class EcoFuseDataset(Dataset):
    def __init__(self, image_dir, csv_path, grid_sizes=(80, 40), num_classes=3,
                 augment=False, drop_optical_prob=0.1, drop_ultrasonic_prob=0.15):
        self.image_dir = image_dir
        self.grid_sizes = grid_sizes
        self.num_classes = num_classes
        self.augment = augment
        self.drop_opt_p = drop_optical_prob
        self.drop_ult_p = drop_ultrasonic_prob

        self.df = pd.read_csv(csv_path)
        self.image_files = self.df['filename'].unique().tolist()
        self.boxes_dict = {}
        for fn, grp in self.df.groupby('filename'):
            boxes = []
            for _, r in grp.iterrows():
                ow, oh = r['width'], r['height']
                cx = ((r['xmin']+r['xmax'])/2.0)/ow
                cy = ((r['ymin']+r['ymax'])/2.0)/oh
                bw = (r['xmax']-r['xmin'])/ow
                bh = (r['ymax']-r['ymin'])/oh
                boxes.append([cx, cy, bw, bh])
            self.boxes_dict[fn] = boxes

    def __len__(self):
        return len(self.image_files)

    def _synth_ultrasonic(self, boxes, length=256):
        """Physics-informed synthetic ultrasonic envelope generation."""
        env = np.random.normal(0.05, 0.02, length).astype(np.float32)
        for cx, cy, w, h in boxes:
            tof = int(cy * length)
            tof = max(0, min(tof, length-1))
            amp = float(np.clip(0.3 + (w*h)*5, 0, 1))
            spread = max(1.0, w*20)
            t = np.arange(length, dtype=np.float32)
            env += amp * np.exp(-0.5*((t - tof)/spread)**2)
        # Attenuation
        env *= np.linspace(1.0, 0.4, length, dtype=np.float32)
        env = np.clip(env, 0, 1)
        # Augmentation
        if self.augment:
            env += np.random.normal(0, 0.03, length).astype(np.float32)
            if np.random.rand() < 0.1:  # pulse dropout
                drop_start = np.random.randint(0, length-20)
                env[drop_start:drop_start+20] = 0.05
            env = np.clip(env, 0, 1)
        slope = np.gradient(env).astype(np.float32)
        tdiff = np.random.normal(0, 0.01, length).astype(np.float32)
        return np.stack([env, slope, tdiff], axis=0)

    def _encode_targets(self, boxes, gs):
        """Encode GT boxes into grid-based targets for one scale."""
        nc = self.num_classes
        obj = np.zeros((1, gs, gs), dtype=np.float32)
        box = np.zeros((4, gs, gs), dtype=np.float32)
        cls = np.zeros((nc, gs, gs), dtype=np.float32)
        for cx, cy, w, h in boxes:
            gx = min(int(cx * gs), gs-1)
            gy = min(int(cy * gs), gs-1)
            # Gaussian splat for objectness (radius=2)
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    ni, nj = gy+di, gx+dj
                    if 0 <= ni < gs and 0 <= nj < gs:
                        d2 = di*di + dj*dj
                        obj[0, ni, nj] = max(obj[0, ni, nj], math.exp(-d2/2.0))
            box[:, gy, gx] = [cx, cy, w, h]
            cls[0, gy, gx] = 1.0  # all microplastic -> class 0
        return obj, box, cls

    def __getitem__(self, idx):
        fn = self.image_files[idx]
        img = cv2.imread(os.path.join(self.image_dir, fn))
        boxes = self.boxes_dict.get(fn, [])

        if img is None:
            return (torch.zeros(2,160,160), torch.zeros(3,256),
                    torch.zeros(8,80,80), torch.zeros(8,40,40))
        img = cv2.resize(img, (160,160))

        # Optical: grayscale + Sobel
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(sx**2 + sy**2)
        mx = edge.max()
        edge = edge/mx if mx > 0 else edge

        # Augmentation: synthetic turbidity
        if self.augment and np.random.rand() < 0.3:
            beta = np.random.uniform(0.5, 3.0)
            d = np.random.uniform(0.3, 1.0)
            A = np.random.uniform(0.2, 0.6)
            gray = gray * np.exp(-beta*d) + A*(1-np.exp(-beta*d))
            gray += np.random.normal(0, 0.02, gray.shape).astype(np.float32)
            gray = np.clip(gray, 0, 1)

        opt = np.stack([gray, edge], axis=0)

        # Modality dropout
        if self.augment:
            if np.random.rand() < self.drop_opt_p:
                opt = np.random.normal(0.5, 0.1, opt.shape).astype(np.float32)
        ult = self._synth_ultrasonic(boxes)
        if self.augment:
            if np.random.rand() < self.drop_ult_p:
                ult = np.random.normal(0.05, 0.02, ult.shape).astype(np.float32)

        # Targets for both scales
        obj1, box1, cls1 = self._encode_targets(boxes, 80)
        obj2, box2, cls2 = self._encode_targets(boxes, 40)
        tgt1 = np.concatenate([obj1, box1, cls1], axis=0)  # [8, 80, 80]
        tgt2 = np.concatenate([obj2, box2, cls2], axis=0)  # [8, 40, 40]

        return (torch.from_numpy(opt), torch.from_numpy(ult),
                torch.from_numpy(tgt1), torch.from_numpy(tgt2))


# ── Loss Functions ───────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1-pt)**self.gamma * bce).mean()

def ciou_loss(pred_box, tgt_box, mask):
    """CIoU loss at positive locations."""
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_box.device)
    pb = torch.sigmoid(pred_box[:, :, mask])  # [B, 4, N]
    tb = tgt_box[:, :, mask]
    # Convert cx,cy,w,h to x1,y1,x2,y2
    px1 = pb[:,0]-pb[:,2]/2; py1 = pb[:,1]-pb[:,3]/2
    px2 = pb[:,0]+pb[:,2]/2; py2 = pb[:,1]+pb[:,3]/2
    tx1 = tb[:,0]-tb[:,2]/2; ty1 = tb[:,1]-tb[:,3]/2
    tx2 = tb[:,0]+tb[:,2]/2; ty2 = tb[:,1]+tb[:,3]/2
    inter_w = (torch.min(px2,tx2)-torch.max(px1,tx1)).clamp(min=0)
    inter_h = (torch.min(py2,ty2)-torch.max(py1,ty1)).clamp(min=0)
    inter = inter_w * inter_h
    area_p = (px2-px1)*(py2-py1)
    area_t = (tx2-tx1)*(ty2-ty1)
    union = area_p + area_t - inter + 1e-7
    iou = inter / union
    return (1 - iou).mean()

class EcoFuseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss()
        self.cls_loss = FocalLoss(alpha=0.5)

    def forward(self, pred, target):
        """pred, target: [B, 8, H, W]. Channels: obj(1), box(4), cls(3)."""
        obj_pred = pred[:, 0:1]
        box_pred = pred[:, 1:5]
        cls_pred = pred[:, 5:8]

        obj_tgt = target[:, 0:1]
        box_tgt = target[:, 1:5]
        cls_tgt = target[:, 5:8]

        # Objectness: focal loss
        l_obj = self.focal(obj_pred, obj_tgt)

        # Box: CIoU at positive cells
        pos_mask = (obj_tgt[:, 0] > 0.5)  # [B, H, W]
        l_box = ciou_loss(box_pred, box_tgt, pos_mask)

        # Classification at positive cells
        l_cls = self.cls_loss(cls_pred * pos_mask.unsqueeze(1).float(),
                              cls_tgt * pos_mask.unsqueeze(1).float())

        return l_obj + 2.0 * l_box + 0.5 * l_cls, l_obj, l_box, l_cls


# ── Training ─────────────────────────────────────────────────────────────────
def train_ecofuse(data_dir, epochs=100, batch_size=16, lr=1e-3):
    csv_path = os.path.join(data_dir, "train", "_annotations.csv")
    train_ds = EcoFuseDataset(os.path.join(data_dir,"train"), csv_path, augment=True)

    val_csv = os.path.join(data_dir, "valid", "_annotations.csv")
    val_ds = None
    if os.path.exists(val_csv):
        val_ds = EcoFuseDataset(os.path.join(data_dir,"valid"), val_csv, augment=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Train: {len(train_ds)} samples")

    model = EcoFuseUNet(num_classes=3).to(device)
    print_model_info(model)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = EcoFuseLoss().to(device)

    history = {'loss':[], 'obj':[], 'box':[], 'cls':[]}

    for epoch in range(epochs):
        model.train()
        ep_loss, ep_obj, ep_box, ep_cls = 0,0,0,0

        for opt, ult, tgt1, tgt2 in train_dl:
            opt, ult = opt.to(device), ult.to(device)
            tgt1, tgt2 = tgt1.to(device), tgt2.to(device)

            det1, det2, r_o, r_u = model(opt, ult)

            loss1, o1, b1, c1 = criterion(det1, tgt1)
            loss2, o2, b2, c2 = criterion(det2, tgt2)
            loss = loss1 + 0.5 * loss2  # weight tiny-scale more

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            ep_loss += loss.item()
            ep_obj += (o1+o2).item()
            ep_box += (b1+b2).item()
            ep_cls += (c1+c2).item()

        scheduler.step()
        n = len(train_dl)
        history['loss'].append(ep_loss/n)
        history['obj'].append(ep_obj/n)
        history['box'].append(ep_box/n)
        history['cls'].append(ep_cls/n)

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Ep {epoch+1:3d}/{epochs} | Loss: {ep_loss/n:.4f} | "
                  f"Obj: {ep_obj/n:.4f} | Box: {ep_box/n:.4f} | Cls: {ep_cls/n:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save model
    save_path = "/kaggle/working/ecofuse_unet_best.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history['loss']); axes[0].set_title('Total Loss')
    axes[1].plot(history['obj']); axes[1].set_title('Objectness Loss')
    axes[2].plot(history['box']); axes[2].set_title('Box Loss')
    for ax in axes: ax.grid(True); ax.set_xlabel('Epoch')
    plt.tight_layout(); plt.show()

    return model, history


# ── Evaluation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, data_dir, device='cuda', conf_thresh=0.3, iou_thresh=0.3):
    val_csv = os.path.join(data_dir, "valid", "_annotations.csv")
    if not os.path.exists(val_csv):
        print("No validation set found."); return
    val_ds = EcoFuseDataset(os.path.join(data_dir,"valid"), val_csv, augment=False)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False)
    model.eval()

    TP, FP, FN = 0, 0, 0
    for opt, ult, tgt1, tgt2 in val_dl:
        opt, ult, tgt1 = opt.to(device), ult.to(device), tgt1.to(device)
        det1, det2, r_o, r_u = model(opt, ult)
        obj_map = torch.sigmoid(det1[0, 0]).cpu().numpy()  # 80x80

        # Extract predictions
        preds = []
        for r in range(80):
            for c in range(80):
                if obj_map[r, c] > conf_thresh:
                    bx = torch.sigmoid(det1[0, 1:5, r, c]).cpu().numpy()
                    preds.append(bx)

        # Extract GT
        gt_obj = tgt1[0, 0].cpu().numpy()
        gts = []
        for r in range(80):
            for c in range(80):
                if gt_obj[r, c] > 0.5:
                    gts.append(tgt1[0, 1:5, r, c].cpu().numpy())

        # Match
        matched = set()
        for p in preds:
            found = False
            for gi, g in enumerate(gts):
                if gi in matched: continue
                # Simple IoU
                ix1=max(p[0]-p[2]/2,g[0]-g[2]/2); iy1=max(p[1]-p[3]/2,g[1]-g[3]/2)
                ix2=min(p[0]+p[2]/2,g[0]+g[2]/2); iy2=min(p[1]+p[3]/2,g[1]+g[3]/2)
                inter=max(0,ix2-ix1)*max(0,iy2-iy1)
                union=p[2]*p[3]+g[2]*g[3]-inter+1e-7
                if inter/union > iou_thresh:
                    TP += 1; matched.add(gi); found = True; break
            if not found: FP += 1
        FN += len(gts) - len(matched)

    P = TP/(TP+FP+1e-7); R = TP/(TP+FN+1e-7); F1 = 2*P*R/(P+R+1e-7)
    print(f"\n{'='*50}")
    print(f"EcoFuse-µNet Validation Results")
    print(f"{'='*50}")
    print(f"  TP={TP}, FP={FP}, FN={FN}")
    print(f"  Precision: {P*100:.2f}%")
    print(f"  Recall:    {R*100:.2f}%")
    print(f"  F1-Score:  {F1*100:.2f}%")
    return {'P': P, 'R': R, 'F1': F1, 'TP': TP, 'FP': FP, 'FN': FN}


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import kagglehub
    dataset_path = kagglehub.dataset_download("imtkaggleteam/microplastic-dataset-for-computer-vision")
    print(f"Dataset: {dataset_path}")

    model, history = train_ecofuse(dataset_path, epochs=100, batch_size=16, lr=1e-3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluate(model, dataset_path, device=device)
