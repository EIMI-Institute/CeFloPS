# %%
import torch
torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
import os
import copy 
from multiprocessing import Process
import time 

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ is not defined, e.g. in a notebook execution
    SCRIPT_DIR = os.getcwd()

model_basepath = os.path.join(SCRIPT_DIR, "pkode_fits_grid")
os.makedirs(model_basepath, exist_ok=True)
print("Created or found:", os.path.abspath(model_basepath)) 
def input_feng_triexp(t, t0, A1, l1, A2, l2, A3, l3):
    t = np.asarray(t)
    t_shift = t - t0
    y = np.zeros_like(t)
    mask = t >= t0
    y[mask] = (A1 * np.exp(l1 * t_shift[mask]) +
               A2 * np.exp(l2 * t_shift[mask]) +
               A3 * np.exp(l3 * t_shift[mask]))
    return y

class KNet(nn.Module):
    def __init__(self, n_hidden=48):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, n_hidden), nn.Tanh(),
            nn.Linear(n_hidden, n_hidden), nn.Tanh(),
            nn.Linear(n_hidden, 2), nn.Softplus()
        )
    def forward(self, t):
        t = t.view(-1, 1)
        return self.model(t)

class PKODE(nn.Module):
    def __init__(self, k_net, K1, K2, K3, n_organs):
        super().__init__()
        self.k_net = k_net
        self.register_buffer("K1", torch.tensor(K1, dtype=torch.float32))
        self.register_buffer("K2", torch.tensor(K2, dtype=torch.float32))
        self.register_buffer("K3", torch.tensor(K3, dtype=torch.float32))
        self.n_organs = n_organs
    def forward(self, t, y):
        rates = self.k_net(t.view(1))
        kinj = rates[0,0]
        kelim = rates[0,1]
        dy = torch.zeros_like(y)
        dy[0] = -kinj * y[0]
        dy[1] = kinj * y[0] - kelim * y[1]
        dy[2] = kelim * y[1]
        for i in range(self.n_organs):
            region_idx = 3+2*i
            trap_idx = 3+2*i+1
            dy[1]    -= self.K1[i] * y[1]
            dy[region_idx] += self.K1[i] * y[1]
            dy[region_idx] -= (self.K2[i]+self.K3[i])*y[region_idx]
            dy[1]    += self.K2[i]*y[region_idx]
            dy[trap_idx] += self.K3[i]*y[region_idx]
        return dy

def train_pkode_model(
    organs, K1, K2, K3, n_organs, t_span, IF_target,
    model_basepath=model_basepath,
    device=None,
    n_epochs=1500,
    patience=10,
    lambda_tailstart=40,
    lambda_tailweight=3.0,
    lambda_injection_end=None,
    lambda_injectionweight=0,
    lambda_peak_weight=0,
    tagextrastr="",
    visualize=True
):
    """
    PKODE fitting with extended readable model naming and plot saving.
    """
    # ---- Directory and base name
    os.makedirs(model_basepath, exist_ok=True)

    # ---- Default IF if None ----
    if IF_target is None:
        t0, A1, l1, A2, l2, A3, l3 = 0.905, 767.3, -4.195, 27.43, -0.232, 29.13, -0.0165
        IF = input_feng_triexp(t_span, t0, A1, l1, A2, l2, A3, l3)
        IF = IF / np.trapz(IF, t_span)
    else:
        IF = np.asarray(IF_target)
        IF = IF / np.trapz(IF, t_span)
    # ---- Model/naming options
    injstr = f"injend{lambda_injection_end if lambda_injection_end is not None else 'full'}"
    injwstr = f"_injw{lambda_injectionweight:.2f}" if lambda_injectionweight else ""
    peakstr = f"_peakw{lambda_peak_weight:.2f}" if lambda_peak_weight else ""
    tailstr = f"_tailw{lambda_tailweight:.2f}_tailfrom{lambda_tailstart:.1f}" if lambda_tailweight else ""
    extrastr = tagextrastr
    model_name = f'pkode_{injstr}{injwstr}{peakstr}{tailstr}{extrastr}'
    model_path = os.path.join(model_basepath, f"{model_name}.pt")
    print("Model options:", model_name)

    # ---- Device and tensors
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    t_tensor = torch.tensor(t_span, dtype=torch.float32, device=device)
    IF_tensor = torch.tensor(IF, dtype=torch.float32, device=device)
    n_state = 3 + 2*n_organs
    y0 = torch.zeros(n_state, device=device)
    y0[0] = 1.0

    k_net = KNet().to(device)
    pkode = PKODE(k_net, K1, K2, K3, n_organs).to(device)
    optimizer = optim.Adam(pkode.parameters(), lr=5e-3)
    losses = []

    # --- Load vs train with everything in readable name
    if os.path.exists(model_path):
        print(f"[INFO] Loading existing weights from {model_path}.")
        pkode.load_state_dict(torch.load(model_path, map_location=device))
    else:
        best_loss = np.inf
        pat_counter = 0
        best_state = copy.deepcopy(pkode.state_dict())
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            ypred = odeint(pkode, y0, t_tensor)
            blood_pred = ypred[:,1]
            loss_main = torch.mean((blood_pred - IF_tensor)**2)
            loss_total = loss_main

            # --- TAIL loss in [lambda_tailstart, end] for late kinetics
            tail_mask = (t_tensor > lambda_tailstart)
            if lambda_tailweight > 0 and torch.sum(tail_mask)>0:
                loss_tail = torch.mean((blood_pred[tail_mask] - IF_tensor[tail_mask])**2)
                loss_total = loss_total + lambda_tailweight * loss_tail

            # --- INJECTION window loss in [0, lambda_injection_end] for early fit
            if lambda_injectionweight > 0 and lambda_injection_end is not None:
                inj_mask = (t_tensor < lambda_injection_end)
                if torch.sum(inj_mask)>0:
                    loss_inj = torch.mean((blood_pred[inj_mask] - IF_tensor[inj_mask])**2)
                    loss_total = loss_total + lambda_injectionweight * loss_inj

            # --- PEAK loss at IF peak time (forces peak amplitude fitting)
            if lambda_peak_weight > 0:
                peak_idx = np.argmax(IF)
                peak_pred = blood_pred[peak_idx]
                peak_true = IF_tensor[peak_idx]
                loss_peak = (peak_pred - peak_true)**2
                loss_total = loss_total + lambda_peak_weight * loss_peak

            # --- Gradients and early stopping ---
            loss_total.backward()
            optimizer.step()
            losses.append(loss_total.item())

            improved = loss_total.item() < best_loss - 1e-7
            if improved:
                best_loss = loss_total.item()
                pat_counter = 0
                best_state = copy.deepcopy(pkode.state_dict())
                torch.save(best_state, model_path)
                print(f"Epoch {epoch} Loss: {loss_total.item():.6f}  [best → saved]")
            else:
                pat_counter += 1
                print(f"Epoch {epoch} Loss: {loss_total.item():.6f}  [no improve {pat_counter}/{patience}]")
                if pat_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    pkode.load_state_dict(torch.load(model_path, map_location=device))

    # --- Output/plots ---
    resdir = os.path.join(model_basepath, model_name + "_plots")
    os.makedirs(resdir, exist_ok=True)

    with torch.no_grad():
        y_out = odeint(pkode, y0, t_tensor)
        blood = y_out[:,1].cpu().numpy()
        rates_out = k_net(t_tensor[:,None]).detach().cpu().numpy()
        region_sum = np.sum([y_out[:,3+2*i].cpu().numpy() for i in range(n_organs)], axis=0)
        trap_sum = np.sum([y_out[:,3+2*i+1].cpu().numpy() for i in range(n_organs)], axis=0)

    # --- Save plots ---
    plt.figure(figsize=(9,5))
    plt.plot(t_span, IF, 'r:', lw=2, label="Target IF")
    plt.plot(t_span, blood, 'b-', lw=2, label="NN ODE blood")
    plt.xlabel("Time (min)")
    plt.ylabel("Fraction / activity")
    plt.legend()
    plt.title("Blood fit: NN-input/elim rates, ODE vs target IF")
    plt.tight_layout()
    plt.savefig(os.path.join(resdir, "fit.png"))
    if visualize: plt.show(); plt.close()
    else: plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(t_span, rates_out[:,0], label='k_inj(t)')
    plt.plot(t_span, rates_out[:,1], label='k_elim(t)')
    plt.xlabel("Time (min)")
    plt.ylabel("Rate (1/min)")
    plt.legend()
    plt.title("NN learned k_inj(t) and k_elim(t)")
    plt.tight_layout()
    plt.savefig(os.path.join(resdir, "rates.png"))
    if visualize: plt.show(); plt.close()
    else: plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(t_span, region_sum, label='Sum of all regions')
    plt.plot(t_span, trap_sum, label='Sum of all traps')
    plt.xlabel('Time (min)')
    plt.ylabel('Total mass in regions/traps')
    plt.legend()
    plt.title('Region and trap sum (NN ODE model)')
    plt.tight_layout()
    plt.savefig(os.path.join(resdir, "regions_traps.png"))
    if visualize: plt.show(); plt.close()
    else: plt.close()

    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss history")
    plt.tight_layout()
    plt.savefig(os.path.join(resdir, "loss.png"))
    if visualize: plt.show(); plt.close()
    else: plt.close()

    plt.figure(figsize=(12,5))
    for i in range(min(40, n_organs)):
        plt.plot(t_span, y_out[:,3+2*i].cpu().numpy(), label=f"{organs[i]}")
        plt.plot(t_span, y_out[:,3+2*i+1].cpu().numpy(), linestyle='--', label=f"{organs[i]} trap")
    plt.xlabel('Time (min)')
    plt.legend()
    plt.title('First few region/trap curves (NN ODE)')
    plt.tight_layout()
    plt.savefig(os.path.join(resdir, "first_regions.png"))
    if visualize: plt.show(); plt.close()
    else: plt.close()
    print(f"[PID {os.getpid()}] Model saved to: {os.path.abspath(model_path)}")
    print(f"[PID {os.getpid()}] Plots saved in: {os.path.abspath(resdir)}")
    return dict(
        blood_pred=blood, IF_target=IF, t_span=t_span,
        rates_out=rates_out, region_sum=region_sum, trap_sum=trap_sum,
        y_out=y_out, losses=losses, model_path=model_path
    )




t0, A1, l1, A2, l2, A3, l3 = 0.905, 767.3, -4.195, 27.43, -0.232, 29.13, -0.0165
t_span = np.linspace(0, 60, 300)
IF = input_feng_triexp(t_span, t0, A1, l1, A2, l2, A3, l3)
IF = IF / np.trapz(IF, t_span)

organs = ['muscle', 'lung', 'liver', 'grey_matter', 'all', 'myocardium', 'spleen', 'guc_lesions',
            'cortex', 'whitematter', 'cerebellum', 'thyroid', 'pancreas', 'kidney']
K1 = np.array([0.026, 0.023, 0.660, 0.107, 0.553, 0.832, 1.593, 0.022,
    0.0896, 0.0337, 0.1318, 0.9663, 0.3561, 0.7023])
K2 = np.array([0.249, 0.205, 0.765, 0.165, 1.213, 2.651, 2.867, 0.296,
    0.2532, 0.1347, 0.6280, 4.6042, 1.7077, 1.3542])
K3 = np.array([0.016, 0.001, 0.002, 0.067, 0.039, 0.099, 0.006, 0.771,
    0.2213, 0.0482, 0.1870, 0.0748, 0.0787, 0.1778])
n_organs = len(organs)
model_basepath = "./pkode_fits_grid"

grid = [
    dict(lambda_injection_end=X, injw=Y, peakw=Z)
    for X in [5, 1/6, 1, None]
    for Y in [0, 2.0]
    for Z in [0, 4.0]
]
lambda_tailstart = 40
lambda_tailweight = 10.0

# limit to max_parallel if needed for VRAM; otherwise runs all (len(grid)) in parallel
max_parallel = 8
running = []
print(f"=== Spawning {len(grid)} model fits in parallel (max {max_parallel} at once) ===")

def run_job(combo):
    try:
        train_pkode_model(
            organs, K1, K2, K3, n_organs, t_span, IF,
            model_basepath=model_basepath,
            n_epochs=300 if combo['lambda_injection_end'] in [None, 5] else 500,
            patience=30,
            lambda_tailstart=lambda_tailstart,
            lambda_tailweight=lambda_tailweight,
            lambda_injection_end=combo['lambda_injection_end'],
            lambda_injectionweight=combo['injw'],
            lambda_peak_weight=combo['peakw'],
            visualize=False,  # never for jobs
        )
    except Exception as e:
        print(f"[ERROR] Process {os.getpid()} failed with exception: {e}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    

    job = 0
    for combo in grid:
        while len(running) >= max_parallel:
            still_alive = [p for p in running if p.is_alive()]
            if len(still_alive) < max_parallel:
                break
            print(f"[INFO] Waiting for available GPU slot ({len(still_alive)} running)...")
            time.sleep(30)
            running = still_alive
        job += 1
        print(f"=== Launching job {job} / {len(grid)}: end={combo['lambda_injection_end']}, injw={combo['injw']}, peakw={combo['peakw']} ===")
        p = Process(target=run_job, args=(combo,))
        p.start()
        running.append(p)

    # Wait for all jobs
    for p in running:
        p.join()
    print("=== All jobs done ===")

    print(f"[INFO] Model weights saved to: {os.path.abspath(model_basepath)}")
    print(f"[INFO] Plots saved in: {os.path.abspath(model_basepath)}")

 
