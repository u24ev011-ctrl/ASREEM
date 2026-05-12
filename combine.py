"""
╔══════════════════════════════════════════════════════════════════╗
║  EcoFuse-µNet: Complete Kaggle Notebook                         ║
║  Paste this ENTIRE file into ONE Kaggle cell and run.           ║
║  Enable GPU: Settings > Accelerator > GPU T4 x2                ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ── Cell 1: Imports ──────────────────────────────────────────────────────────
import os, cv2, math, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import kagglehub

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ═══════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════

def hard_sigmoid(x):
    return torch.clamp((x + 3.0) / 6.0, 0.0, 1.0)


class GhostDSBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        half = max(out_c // 2, 8)
        exp_c = half * 2
        self.pw_expand = nn.Sequential(
            nn.Conv2d(in_c, half, 1, bias=False), nn.BatchNorm2d(half), nn.ReLU6(True))
        self.dw_ghost = nn.Sequential(
            nn.Conv2d(half, half, 3, padding=1, groups=half, bias=False),
            nn.BatchNorm2d(half), nn.ReLU6(True))
        self.dw_spatial = nn.Sequential(
            nn.Conv2d(exp_c, exp_c, 3, stride=stride, padding=1, groups=exp_c, bias=False),
            nn.BatchNorm2d(exp_c), nn.ReLU6(True))
        self.pw_project = nn.Sequential(
            nn.Conv2d(exp_c, out_c, 1, bias=False), nn.BatchNorm2d(out_c))
        self.use_res = (in_c == out_c) and (stride == 1)

    def forward(self, x):
        e = self.pw_expand(x)
        g = self.dw_ghost(e)
        out = self.pw_project(self.dw_spatial(torch.cat([e, g], 1)))
        return F.relu6(out + x if self.use_res else out)


def _make_stage(in_c, out_c, n, s=1):
    return nn.Sequential(GhostDSBlock(in_c, out_c, s),
                         *[GhostDSBlock(out_c, out_c) for _ in range(n-1)])


class OpticalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU6(True))
        self.o1 = _make_stage(16, 16, 2)
        self.o2 = _make_stage(16, 32, 3, s=2)
        self.o3 = _make_stage(32, 64, 4, s=2)
        self.o4 = _make_stage(64, 96, 3, s=2)
        self.o5 = _make_stage(96, 128, 2, s=2)

    def forward(self, x):
        x = self.stem(x)
        o1 = self.o1(x); o2 = self.o2(o1); o3 = self.o3(o2)
        o4 = self.o4(o3); o5 = self.o5(o4)
        return o1, o2, o3, o4, o5


class DSTCNBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, dilation=1):
        super().__init__()
        self.dw = nn.Conv1d(in_c, in_c, 3, stride=stride, padding=dilation,
                            dilation=dilation, groups=in_c, bias=False)
        self.bn1 = nn.BatchNorm1d(in_c)
        self.pw = nn.Conv1d(in_c, out_c, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.use_res = (in_c == out_c) and (stride == 1) and (dilation == 1)

    def forward(self, x):
        out = F.relu6(self.bn2(self.pw(F.relu6(self.bn1(self.dw(x))))))
        return out + x if self.use_res else out


class UltrasonicEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(3, 16, 7, padding=3, bias=False), nn.BatchNorm1d(16), nn.ReLU6(True))
        self.u1 = DSTCNBlock(16, 32, stride=2)
        self.u2 = DSTCNBlock(32, 48, stride=2)
        self.u3 = DSTCNBlock(48, 64, dilation=2)
        self.tpool = nn.AdaptiveAvgPool1d(32)
        self.gpool = nn.AdaptiveAvgPool1d(1)
        self.gproj = nn.Linear(64, 96)

    def forward(self, x):
        x = self.u3(self.u2(self.u1(self.stem(x))))
        u_t = self.tpool(x)
        u_g = self.gproj(self.gpool(u_t).squeeze(-1))
        return u_t, u_g


class CompactFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lat_o2 = nn.Conv2d(32, 48, 1, bias=False)
        self.lat_o3 = nn.Conv2d(64, 48, 1, bias=False)
        self.smooth_p2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, groups=48, bias=False),
            nn.BatchNorm2d(48), nn.ReLU6(True))
        self.lat_o1 = nn.Conv2d(16, 32, 1, bias=False)
        self.lat_p2 = nn.Conv2d(48, 32, 1, bias=False)
        self.smooth_p1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32), nn.ReLU6(True))

    def forward(self, o1, o2, o3, o5):
        p2 = self.smooth_p2(self.lat_o2(o2) +
             F.interpolate(self.lat_o3(o3), size=o2.shape[2:], mode='nearest'))
        p1 = self.smooth_p1(self.lat_o1(o1) +
             F.interpolate(self.lat_p2(p2), size=o1.shape[2:], mode='nearest'))
        pg = F.adaptive_avg_pool2d(o5, 1).squeeze(-1).squeeze(-1)
        return p1, p2, pg


class RIMEFuse(nn.Module):
    def __init__(self, C, H, W):
        super().__init__()
        self.C, self.H, self.W = C, H, W
        self.mlp_o = nn.Sequential(nn.Linear(C, 16), nn.ReLU6(), nn.Linear(16, 1))
        self.mlp_u = nn.Sequential(nn.Linear(96, 16), nn.ReLU6(), nn.Linear(16, 1))
        self.proj_a = nn.Conv1d(64, 1, 1, bias=False)
        self.proj_b = nn.Linear(96, C, bias=False)
        self.dw_gate = nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False)
        self.pw_gate = nn.Conv2d(C, C, 1, bias=False)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.lam = nn.Parameter(torch.tensor(0.5))
        self.mu = nn.Parameter(torch.tensor(0.3))

    def forward(self, P, U_t, U_g, delta_t=None):
        B = P.size(0)
        r_o = hard_sigmoid(self.mlp_o(F.adaptive_avg_pool2d(P, 1).view(B, -1)))
        r_u = hard_sigmoid(self.mlp_u(U_g))
        if delta_t is not None:
            r_u = r_u * torch.clamp(1.0 - delta_t.view(B,1)/10.0, 0, 1)

        a = F.interpolate(self.proj_a(U_t), size=self.H, mode='linear',
                          align_corners=False).view(B, 1, self.H, 1)
        b = self.proj_b(U_g).view(B, self.C, 1, 1)
        E = (a * b).expand(-1, -1, -1, self.W)

        C_o = hard_sigmoid(self.dw_gate(P))
        C_u = hard_sigmoid(self.pw_gate(E))
        r_u_b = r_u.view(B, 1, 1, 1)

        F_l = P*(1+self.lam*r_u_b*C_u) + E*(r_u_b*C_o) - P*self.mu*(1-C_u)*r_u_b
        return F_l, r_o.squeeze(-1), r_u.squeeze(-1)


class DetHead(nn.Module):
    def __init__(self, in_c, nc=3):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c), nn.ReLU6(True))
        self.pw = nn.Conv2d(in_c, 5+nc, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class EcoFuseUNet(nn.Module):
    def __init__(self, nc=3):
        super().__init__()
        self.opt = OpticalEncoder()
        self.ult = UltrasonicEncoder()
        self.fpn = CompactFPN()
        self.rime1 = RIMEFuse(32, 80, 80)
        self.rime2 = RIMEFuse(48, 40, 40)
        self.head1 = DetHead(32, nc)
        self.head2 = DetHead(48, nc)

    def forward(self, optical, ultrasonic, delta_t=None):
        o1, o2, o3, o4, o5 = self.opt(optical)
        u_t, u_g = self.ult(ultrasonic)
        p1, p2, pg = self.fpn(o1, o2, o3, o5)
        f1, ro1, ru1 = self.rime1(p1, u_t, u_g, delta_t)
        f2, ro2, ru2 = self.rime2(p2, u_t, u_g, delta_t)
        return self.head1(f1), self.head2(f2), (ro1+ro2)/2, (ru1+ru2)/2


def model_info(m):
    t = sum(p.numel() for p in m.parameters())
    parts = {'OpticalEnc': m.opt, 'UltrasonicEnc': m.ult, 'FPN': m.fpn,
             'RIME_P1': m.rime1, 'RIME_P2': m.rime2,
             'Head_P1': m.head1, 'Head_P2': m.head2}
    print("="*55)
    print("EcoFuse-µNet Architecture")
    print("="*55)
    for n, mod in parts.items():
        print(f"  {n:.<35s} {sum(p.numel() for p in mod.parameters()):>8,d}")
    print(f"  {'TOTAL':.<35s} {t:>8,d}")
    print(f"  INT8 size: {t/1e6:.2f} MB")
    print("="*55)


# ═══════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════

class EcoFuseDataset(Dataset):
    def __init__(self, img_dir, csv_path, augment=False):
        self.img_dir, self.augment = img_dir, augment
        df = pd.read_csv(csv_path)
        self.fnames = df['filename'].unique().tolist()
        self.boxes = {}
        for fn, grp in df.groupby('filename'):
            bx = []
            for _, r in grp.iterrows():
                ow, oh = r['width'], r['height']
                bx.append([(r['xmin']+r['xmax'])/2/ow, (r['ymin']+r['ymax'])/2/oh,
                           (r['xmax']-r['xmin'])/ow, (r['ymax']-r['ymin'])/oh])
            self.boxes[fn] = bx

    def __len__(self):
        return len(self.fnames)

    def _synth_ult(self, boxes):
        env = np.random.normal(0.05, 0.02, 256).astype(np.float32)
        for cx, cy, w, h in boxes:
            t = np.arange(256, dtype=np.float32)
            tof = int(cy * 256)
            amp = float(np.clip(0.3 + w*h*5, 0, 1))
            env += amp * np.exp(-0.5*((t-tof)/max(1, w*20))**2)
        env *= np.linspace(1, 0.4, 256, dtype=np.float32)
        env = np.clip(env, 0, 1)
        if self.augment:
            env += np.random.normal(0, 0.03, 256).astype(np.float32)
            env = np.clip(env, 0, 1)
        slope = np.gradient(env).astype(np.float32)
        tdiff = np.random.normal(0, 0.01, 256).astype(np.float32)
        return np.stack([env, slope, tdiff])

    def _encode_tgt(self, boxes, gs):
        tgt = np.zeros((8, gs, gs), dtype=np.float32)
        for cx, cy, w, h in boxes:
            gx, gy = min(int(cx*gs), gs-1), min(int(cy*gs), gs-1)
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    ni, nj = gy+di, gx+dj
                    if 0 <= ni < gs and 0 <= nj < gs:
                        tgt[0, ni, nj] = max(tgt[0, ni, nj], math.exp(-(di*di+dj*dj)/2))
            tgt[1:5, gy, gx] = [cx, cy, w, h]
            tgt[5, gy, gx] = 1.0  # class 0
        return tgt

    def __getitem__(self, idx):
        fn = self.fnames[idx]
        img = cv2.imread(os.path.join(self.img_dir, fn))
        boxes = self.boxes.get(fn, [])
        if img is None:
            return torch.zeros(2,160,160), torch.zeros(3,256), torch.zeros(8,80,80), torch.zeros(8,40,40)

        img = cv2.resize(img, (160, 160))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(sx**2+sy**2)
        mx = edge.max()
        if mx > 0: edge /= mx

        if self.augment and np.random.rand() < 0.3:
            b, d, A = np.random.uniform(0.5,3), np.random.uniform(0.3,1), np.random.uniform(0.2,0.6)
            gray = gray*np.exp(-b*d) + A*(1-np.exp(-b*d))
            gray = np.clip(gray + np.random.normal(0, 0.02, gray.shape).astype(np.float32), 0, 1)

        opt = np.stack([gray, edge]).astype(np.float32)
        ult = self._synth_ult(boxes)

        if self.augment:
            if np.random.rand() < 0.1: opt = np.random.normal(0.5, 0.1, opt.shape).astype(np.float32)
            if np.random.rand() < 0.15: ult = np.random.normal(0.05, 0.02, ult.shape).astype(np.float32)

        tgt1 = self._encode_tgt(boxes, 80)
        tgt2 = self._encode_tgt(boxes, 40)
        return torch.from_numpy(opt), torch.from_numpy(ult), torch.from_numpy(tgt1), torch.from_numpy(tgt2)


# ═══════════════════════════════════════════════════════════════════
# LOSS
# ═══════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.a, self.g = alpha, gamma
    def forward(self, p, t):
        bce = F.binary_cross_entropy_with_logits(p, t, reduction='none')
        return (self.a * (1-torch.exp(-bce))**self.g * bce).mean()

def ciou_loss(pred, tgt, mask):
    if mask.sum() == 0:
        return pred.sum() * 0
    p = torch.sigmoid(pred[:, :, mask])
    t = tgt[:, :, mask]
    px1, py1 = p[:,0]-p[:,2]/2, p[:,1]-p[:,3]/2
    px2, py2 = p[:,0]+p[:,2]/2, p[:,1]+p[:,3]/2
    tx1, ty1 = t[:,0]-t[:,2]/2, t[:,1]-t[:,3]/2
    tx2, ty2 = t[:,0]+t[:,2]/2, t[:,1]+t[:,3]/2
    iw = (torch.min(px2,tx2)-torch.max(px1,tx1)).clamp(0)
    ih = (torch.min(py2,ty2)-torch.max(py1,ty1)).clamp(0)
    inter = iw * ih
    union = (px2-px1)*(py2-py1)+(tx2-tx1)*(ty2-ty1)-inter+1e-7
    return (1 - inter/union).mean()

def compute_loss(pred, tgt):
    focal = FocalLoss()
    l_obj = focal(pred[:,0:1], tgt[:,0:1])
    mask = tgt[:,0] > 0.5
    l_box = ciou_loss(pred[:,1:5], tgt[:,1:5], mask)
    l_cls = focal(pred[:,5:8]*mask.unsqueeze(1).float(), tgt[:,5:8]*mask.unsqueeze(1).float())
    return l_obj + 2*l_box + 0.5*l_cls, l_obj, l_box, l_cls


# ═══════════════════════════════════════════════════════════════════
# TRAIN & EVALUATE
# ═══════════════════════════════════════════════════════════════════

def run_training(data_dir, epochs=100, bs=16, lr=1e-3):
    train_ds = EcoFuseDataset(f"{data_dir}/train", f"{data_dir}/train/_annotations.csv", augment=True)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EcoFuseUNet().to(dev)
    model_info(model)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    print(f"\n🚀 Training on {dev} | {len(train_ds)} samples | {epochs} epochs\n")
    hist = []

    for ep in range(epochs):
        model.train()
        ep_loss = 0
        for o, u, t1, t2 in train_dl:
            o, u, t1, t2 = o.to(dev), u.to(dev), t1.to(dev), t2.to(dev)
            d1, d2, ro, ru = model(o, u)
            loss1, _, _, _ = compute_loss(d1, t1)
            loss2, _, _, _ = compute_loss(d2, t2)
            loss = loss1 + 0.5*loss2

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += loss.item()

        sched.step()
        avg = ep_loss/len(train_dl)
        hist.append(avg)
        if (ep+1) % 10 == 0 or ep == 0:
            print(f"  Epoch {ep+1:3d}/{epochs} | Loss: {avg:.4f} | LR: {sched.get_last_lr()[0]:.6f}")

    # Save
    torch.save(model.state_dict(), "/kaggle/working/ecofuse_best.pth")
    print("\n✅ Model saved to /kaggle/working/ecofuse_best.pth")

    plt.figure(figsize=(10,4))
    plt.plot(hist); plt.title("Training Loss"); plt.xlabel("Epoch"); plt.grid(True); plt.show()

    # Evaluate
    if os.path.exists(f"{data_dir}/valid/_annotations.csv"):
        val_ds = EcoFuseDataset(f"{data_dir}/valid", f"{data_dir}/valid/_annotations.csv")
        val_dl = DataLoader(val_ds, batch_size=1)
        model.eval()
        TP, FP, FN = 0, 0, 0
        with torch.no_grad():
            for o, u, t1, t2 in val_dl:
                o, u, t1 = o.to(dev), u.to(dev), t1.to(dev)
                d1, d2, _, _ = model(o, u)
                obj = torch.sigmoid(d1[0,0]).cpu().numpy()
                gt = t1[0,0].cpu().numpy()
                preds = (obj > 0.3).sum()
                gts = (gt > 0.5).sum()
                matched = min(preds, gts)
                TP += int(matched)
                FP += int(max(0, preds - gts))
                FN += int(max(0, gts - preds))

        P = TP/(TP+FP+1e-7); R = TP/(TP+FN+1e-7); F1 = 2*P*R/(P+R+1e-7)
        print(f"\n{'='*50}")
        print(f"📊 EcoFuse-µNet Results")
        print(f"{'='*50}")
        print(f"  Precision: {P*100:.2f}%  |  Recall: {R*100:.2f}%  |  F1: {F1*100:.2f}%")
        print(f"  TP={TP}  FP={FP}  FN={FN}")

    return model


# ═══════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════

# Download dataset
data_path = kagglehub.dataset_download("imtkaggleteam/microplastic-dataset-for-computer-vision")
print(f"📦 Dataset at: {data_path}")
print(f"📂 Contents: {os.listdir(data_path)}")

# Train
model = run_training(data_path, epochs=100, bs=16, lr=1e-3)
