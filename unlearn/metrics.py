from torch.nn import functional as F
import torch
import numpy as np
from sklearn.svm import SVC


def JSDiv(p, q):
    m = (p + q) / 2
    return 0.5 * F.kl_div(torch.log(p), m) + 0.5 * F.kl_div(torch.log(q), m)


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


# --------------------------------------------------------------------------- #
# Metric 1: Membership Inference Attack (MIA)
# --------------------------------------------------------------------------- #
# Idea: an attacker only observes a model's output confidence. If a model was
# trained on a sample it tends to be *more* confident (lower prediction
# entropy) than on unseen data. We train a simple SVC attacker to separate
# "members" (retain-train data the model was trained on) from "non-members"
# (held-out test data), using per-sample entropy as the single feature, and
# then ask it about the *forget* samples. After successful unlearning the
# forget samples should look like non-members, so the fraction still flagged as
# members drops towards the non-member baseline.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_prob(data_loader, model, device):
    """Collect the softmax probability vectors produced by ``model``."""
    model.eval()
    probs = []
    for batch in data_loader:
        data = batch[0].to(device)
        output = model(data)
        probs.append(F.softmax(output, dim=-1).detach().cpu())
    return torch.cat(probs)


def get_membership_attack_data(retain_loader, forget_loader, test_loader, model, device):
    """Build (feature, label) sets for the membership-inference attacker.

    Members     (label 1): retain-train samples (seen during training).
    Non-members (label 0): held-out test samples (never seen).
    Query set             : forget-train samples (were members before unlearning).
    Feature               : per-sample prediction entropy.
    """
    retain_prob = collect_prob(retain_loader, model, device)
    forget_prob = collect_prob(forget_loader, model, device)
    test_prob = collect_prob(test_loader, model, device)

    X_r = torch.cat([entropy(retain_prob), entropy(test_prob)]).numpy().reshape(-1, 1)
    Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])

    X_f = entropy(forget_prob).numpy().reshape(-1, 1)
    Y_f = np.ones(len(forget_prob))
    return X_f, Y_f, X_r, Y_r


def get_membership_attack_prob(retain_loader, forget_loader, test_loader, model, device):
    """Fraction of forget samples the attacker still classifies as 'member'.

    Lower is better: a value close to the non-member baseline means the model
    no longer leaks that the forget data was used for training. A value close
    to 1 means the forget information is still fully recoverable.
    """
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(
        retain_loader, forget_loader, test_loader, model, device
    )
    clf = SVC(C=3, gamma="auto", kernel="rbf")
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    return float(results.mean())


# --------------------------------------------------------------------------- #
# Metric 2: Zero Retrain Forgetting (ZRF) score
# --------------------------------------------------------------------------- #
# Idea: a perfectly forgotten class should trigger the same (essentially
# random) reaction as a network that was never trained. ZRF measures how close
# the unlearned model's output distribution on the forget set is to that of a
# randomly-initialised "incompetent" teacher, via Jensen-Shannon divergence.
#   ZRF = 1 - mean_JS(student, incompetent_teacher)   in [0, 1]
# Near 1 -> behaves like an untrained net on forget data (good forgetting).
# Near 0 -> still responds confidently / distinctively (poor forgetting).
# --------------------------------------------------------------------------- #
def _js_divergence(p, q, eps=1e-12):
    """Batched Jensen-Shannon divergence (nats) between prob matrices [B, C]."""
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum(dim=1)
    kl_qm = (q * (q / m).log()).sum(dim=1)
    return 0.5 * kl_pm + 0.5 * kl_qm


@torch.no_grad()
def compute_zrf(model, unlearning_teacher, forget_loader, device):
    """Zero Retrain Forgetting score in [0, 1] (higher = better forgetting)."""
    model.eval()
    unlearning_teacher.eval()
    ln2 = torch.log(torch.tensor(2.0)).item()

    total, num_samples = 0.0, 0
    for batch in forget_loader:
        x = batch[0].to(device)
        student_out = F.softmax(model(x), dim=1)
        teacher_out = F.softmax(unlearning_teacher(x), dim=1)
        js = _js_divergence(student_out, teacher_out) / ln2  # normalise to [0, 1]
        total += js.sum().item()
        num_samples += x.size(0)

    return 1.0 - total / max(num_samples, 1)
