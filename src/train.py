import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import ForgeryDataset
from .losses import BCEDiceLoss


def _run_epoch(model, loader, loss_fn, device, opt=None, scaler=None):
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
    for b in tqdm(loader, leave=False):
        x, y = b["image"].to(device), b["mask"].to(device)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                loss = loss_fn(model(x), y)
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        total += loss.item() * len(x)
        n += len(x)
    return total / n


def fit(model, train_df, val_df, epochs=10, batch_size=8, lr=3e-4, size=512,
        save_path="/content/checkpoints/unet.pth", num_workers=2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    tr = DataLoader(ForgeryDataset(train_df, size, augment=True), batch_size=batch_size,
                    shuffle=True, num_workers=num_workers, pin_memory=True)
    va = DataLoader(ForgeryDataset(val_df, size), batch_size=batch_size,
                    shuffle=False, num_workers=num_workers, pin_memory=True)
    loss_fn = BCEDiceLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best, history = float("inf"), []
    for ep in range(1, epochs + 1):
        tl = _run_epoch(model, tr, loss_fn, device, opt, scaler)
        vl = _run_epoch(model, va, loss_fn, device)
        history.append({"epoch": ep, "train_loss": tl, "val_loss": vl})
        print(f"epoch {ep}/{epochs} | train {tl:.4f} | val {vl:.4f}")
        if vl < best:
            best = vl
            torch.save(model.state_dict(), save_path)
    return history