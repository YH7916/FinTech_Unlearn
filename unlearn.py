import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset import UnLearningData, UnLearningDataWithLabel
import numpy as np
from utils import *


def fit_one_unlearning_cycle(epochs, model, train_loader, val_loader, lr, device):
    history = []

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = training_step(model, batch, device)
            loss.backward()
            train_losses.append(loss.detach().cpu())

            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))

        result = evaluate(model, val_loader, device)
        result["train_loss"] = torch.stack(train_losses).mean()
        result["lrs"] = lrs
        epoch_end(model, epoch, result)
        history.append(result)
    return history


def UnlearnerLoss(output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature):
    """Bad-teacher (blindspot) distillation loss.

    Reference: "Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks
    Using an Incompetent Teacher" (AAAI 2023).

    Each sample carries a binary label from ``UnLearningData``:
        labels == 1  -> forget sample  -> imitate the *incompetent* teacher
        labels == 0  -> retain sample  -> imitate the *fully-trained* teacher

    The student is pushed (via temperature-scaled KL divergence) towards the
    per-sample selected teacher, so it keeps its knowledge on retain data while
    collapsing to a near-random response on forget data.
    """
    labels = torch.unsqueeze(labels, dim=1)

    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

    # For each sample pick the teacher distribution to mimic.
    overall_teacher_out = labels * u_teacher_out + (1 - labels) * f_teacher_out

    student_out = F.log_softmax(output / KL_temperature, dim=1)
    return F.kl_div(student_out, overall_teacher_out, reduction="batchmean")


def blindspot_unlearner(
    model,
    unlearning_teacher,
    full_trained_teacher,
    retain_data,
    forget_data,
    epochs=10,
    optimizer="adam",
    lr=0.01,
    batch_size=256,
    num_workers=0,
    device="cuda",
    KL_temperature=1,
):
    # creating the unlearning dataset.
    unlearning_data = UnLearningData(forget_data=forget_data, retain_data=retain_data)
    unlearning_loader = DataLoader(
        unlearning_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda")
    )

    unlearning_teacher.eval()
    full_trained_teacher.eval()
    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        # if optimizer is not a valid string, then assuming it as a function to return optimizer
        optimizer = optimizer  # (model.parameters())

    for epoch in range(epochs):
        # =============================== todo ===============================
        model.train()
        running_loss, num_samples = 0.0, 0

        for x, y in unlearning_loader:
            x = x.to(device)
            y = y.float().to(device)

            # Both teachers are frozen: only produce reference distributions.
            with torch.no_grad():
                full_teacher_logits = full_trained_teacher(x)
                unlearn_teacher_logits = unlearning_teacher(x)

            output = model(x)
            optimizer.zero_grad()
            loss = UnlearnerLoss(
                output=output,
                labels=y,
                full_teacher_logits=full_teacher_logits,
                unlearn_teacher_logits=unlearn_teacher_logits,
                KL_temperature=KL_temperature,
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            num_samples += x.size(0)

        loss = running_loss / max(num_samples, 1)
        # =============================== end todo ===============================
        print("Epoch {} Unlearning Loss {}".format(epoch + 1, loss))


def blindspot_unlearner_retain_aware(
    model,
    unlearning_teacher,
    full_trained_teacher,
    retain_data,
    forget_data,
    epochs=10,
    lr=0.01,
    batch_size=256,
    num_workers=0,
    device="cuda",
    KL_temperature=1,
    retain_ce_weight=1.0,
):
    """Improved unlearner (our method): Retain-Aware Bad Teacher.

    Pure bad-teacher distillation only *imitates* the competent teacher on
    retain data; when the forget class is entangled with retain classes the
    distillation signal is noisy and retain accuracy drops. We add a supervised
    cross-entropy term on retain samples (using their ground-truth labels),
    weighted by ``retain_ce_weight``, which anchors the student's decision
    boundary on the data we want to keep and improves the forget/retain
    trade-off at negligible extra cost.

    Total loss = KL_distillation + retain_ce_weight * CE(retain, y_true)
    """
    unlearning_data = UnLearningDataWithLabel(forget_data=forget_data, retain_data=retain_data)
    unlearning_loader = DataLoader(
        unlearning_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda")
    )

    unlearning_teacher.eval()
    full_trained_teacher.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss, num_samples = 0.0, 0

        for x, y, clabel in unlearning_loader:
            x = x.to(device)
            y = y.float().to(device)
            clabel = clabel.to(device)

            with torch.no_grad():
                full_teacher_logits = full_trained_teacher(x)
                unlearn_teacher_logits = unlearning_teacher(x)

            output = model(x)

            kd_loss = UnlearnerLoss(
                output=output,
                labels=y,
                full_teacher_logits=full_teacher_logits,
                unlearn_teacher_logits=unlearn_teacher_logits,
                KL_temperature=KL_temperature,
            )

            # Supervised anchor on retain samples only (labels == 0).
            retain_mask = y == 0
            if retain_mask.any():
                ce_loss = F.cross_entropy(output[retain_mask], clabel[retain_mask])
            else:
                ce_loss = torch.zeros((), device=device)

            loss = kd_loss + retain_ce_weight * ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            num_samples += x.size(0)

        print(
            "Epoch {} Retain-Aware Unlearning Loss {}".format(epoch + 1, running_loss / max(num_samples, 1))
        )
