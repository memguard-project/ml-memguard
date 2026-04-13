# eBPF Probes in Kubernetes — Capability Requirements and Security Guide

This document is the authoritative reference for deploying memguard's eBPF probes
inside Kubernetes. It covers:

1. [Why eBPF needs special capabilities](#why-ebpf-needs-special-capabilities)
2. [Attachment mode selection](#attachment-mode-selection)
3. [Enabling eBPF via Helm](#enabling-ebpf-via-helm)
4. [Enabling eBPF via MemGuardPolicy CRD](#enabling-ebpf-via-memguardpolicy-crd)
5. [seccomp profile patch for the bpf(2) syscall](#seccomp-profile-patch)
6. [Admission controller exceptions](#admission-controller-exceptions)
7. [Security audit trail](#security-audit-trail)
8. [Troubleshooting](#troubleshooting)

---

## Why eBPF needs special capabilities

Linux eBPF programs require two capabilities that are **not** granted to
containers by default:

| Capability | Purpose | Linux version |
|---|---|---|
| `CAP_BPF` | Load and attach BPF programs to kernel hooks | ≥ 5.8 |
| `CAP_PERFMON` | Read performance monitoring counters (required by bcc) | ≥ 5.8 |

On kernels < 5.8 (or when `CAP_BPF`/`CAP_PERFMON` are absent), the legacy
`CAP_SYS_ADMIN` capability covers both. memguard accepts either combination.

Without these capabilities, `BPFProbeLoader.available` returns `False` and all
eBPF paths become silent no-ops — the sidecar continues operating in
poll-based mode with no change in pod behaviour.

---

## Attachment mode selection

`BPFProbeLoader` detects the execution environment at `__init__` time and
selects the appropriate attachment mode:

| Runtime detected | Has CAP_BPF | Attachment mode | Notes |
|---|---|---|---|
| any | yes | `raw_tracepoint` | Full tracepoint coverage, lowest overhead |
| kubernetes / docker / container | no | `cgroup_skb` | Cgroup-scoped programs; reduced coverage |
| host (bare metal / VM) | no | `none` | Probes disabled; clear log message emitted |

**Runtime detection logic** (in priority order):

1. `KUBERNETES_SERVICE_HOST` env var — injected into every Kubernetes pod
2. `/.dockerenv` file — created by the Docker daemon in every container
3. `/proc/1/cgroup` content — contains `"docker"`, `"containerd"`,
   `"kubepods"`, or `"lxc"` when running inside a container
4. Falls back to `"host"` when none of the above match

```python
from memory_guard.ebpf._loader import BPFProbeLoader

loader = BPFProbeLoader()
print(loader.runtime)          # "kubernetes" | "docker" | "container" | "host"
print(loader.attachment_mode)  # "raw_tracepoint" | "cgroup_skb" | "none"
print(loader.available)        # True | False
```

---

## Enabling eBPF via Helm

eBPF is **disabled by default**. Enable it by setting `ebpf.enabled: true` in
`values.yaml` or on the `helm upgrade` command line:

```yaml
# values.yaml — operator-managed override
ebpf:
  enabled: true
```

```bash
helm upgrade my-vllm memguard/memguard \
  --set ebpf.enabled=true \
  --reuse-values
```

When `ebpf.enabled: true`:

1. The sidecar `securityContext` gains `CAP_BPF` and `CAP_PERFMON`:
   ```yaml
   securityContext:
     capabilities:
       add:
         - CAP_BPF
         - CAP_PERFMON
   ```
2. The install command switches from `pip install ml-memguard` to
   `pip install 'ml-memguard[ebpf]'` (adds `bcc` and `libbpf-python`).
3. `MEMGUARD_EBPF_ENABLED=true` is set in the ConfigMap and picked up by
   the sidecar at startup.

---

## Enabling eBPF via MemGuardPolicy CRD

The `MemGuardPolicy` CRD carries a `spec.ebpf.enabled` boolean field that
overrides the Helm value at runtime — no Helm rollout required:

```yaml
apiVersion: memguard.io/v1alpha1
kind: MemGuardPolicy
metadata:
  name: default
  namespace: inference
spec:
  shedThreshold: 0.70
  ebpf:
    enabled: true
```

Apply with:

```bash
kubectl apply -f memguardpolicy.yaml -n inference

# Or patch in-place:
kubectl patch memguardpolicy default -n inference \
  --type merge -p '{"spec":{"ebpf":{"enabled":true}}}'
```

The sidecar watches the policy object for changes and picks up updates within
one poll cycle (default 5 s). **Note**: the sidecar must already hold
`CAP_BPF + CAP_PERFMON` (set in the securityContext) for the CRD flag to
take effect — the flag alone does not grant capabilities.

---

## seccomp profile patch

Kubernetes pods run with the `RuntimeDefault` seccomp profile, which **does not
allow** the `bpf(2)` syscall. Without patching the profile, the kernel will
block `bpf()` calls even if `CAP_BPF` is present, causing a `EPERM` error
when `bcc` tries to compile and load the BPF program.

### Option A — Unconfined seccomp (quick, not recommended for production)

```yaml
# deployment.yaml patch
spec:
  template:
    spec:
      securityContext:
        seccompProfile:
          type: Unconfined
```

### Option B — Custom seccomp profile (recommended)

Create a seccomp profile that allows `bpf`, `perf_event_open`, and
`perf_event_read` while keeping everything else at `RuntimeDefault`:

```json
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "syscalls": [
    {
      "names": [
        "bpf",
        "perf_event_open",
        "perf_event_read"
      ],
      "action": "SCMP_ACT_ALLOW"
    }
  ]
}
```

Save this as `/var/lib/kubelet/seccomp/memguard-ebpf.json` on each node, then
reference it in the pod spec:

```yaml
spec:
  securityContext:
    seccompProfile:
      type: Localhost
      localhostProfile: memguard-ebpf.json
```

Apply the profile via a DaemonSet that copies the file, or use a
node-provisioning tool (Ansible, cloud-init, Karpenter AMI baking).

### Option C — AppArmor annotation (GKE / AKS)

```yaml
metadata:
  annotations:
    container.apparmor.security.beta.kubernetes.io/memguard-sidecar: unconfined
```

---

## Admission controller exceptions

Most enterprise clusters run Kyverno or OPA Gatekeeper policies that deny
`CAP_BPF` by default. Add an exception for the memguard namespace:

### Kyverno ClusterPolicy exception

```yaml
apiVersion: kyverno.io/v2
kind: PolicyException
metadata:
  name: memguard-ebpf-cap-bpf
  namespace: kyverno
spec:
  exceptions:
    - policyName: disallow-capabilities
      ruleNames:
        - autogen-adding-capabilities
        - autogen-cronjob-adding-capabilities
  match:
    any:
      - resources:
          kinds:
            - Pod
          namespaces:
            - inference
          selector:
            matchLabels:
              app.kubernetes.io/name: memguard
```

### OPA Gatekeeper exemption

Add the namespace to the `AllowedCapabilities` constraint's excluded namespaces:

```yaml
spec:
  parameters:
    allowedCapabilities:
      - CAP_BPF
      - CAP_PERFMON
```

---

## Security audit trail

This section documents the security properties of the BPF programs shipped
with `ml-memguard`. It is intended for enterprise security teams reviewing
the deployment before approving production rollout.

### Kernel verifier guarantees

Every BPF program loaded by memguard passes the Linux kernel verifier
(`kernel/bpf/verifier.c`). The verifier enforces:

1. **Termination** — all programs have bounded loops; infinite loops are
   rejected at load time.
2. **Memory safety** — all pointer accesses are bounds-checked; out-of-bounds
   reads/writes cause `EACCES` at load time.
3. **No arbitrary kernel memory access** — programs may only access kernel
   memory via designated BPF helpers (`bpf_probe_read`, `bpf_get_current_pid_tgid`,
   etc.). Direct pointer arithmetic into kernel address space is rejected.
4. **Stack size limit** — each BPF frame is limited to 512 bytes; stack
   overflows are rejected.

### What memguard BPF programs read

| Probe | Tracepoint | Data read | Data written |
|---|---|---|---|
| `CgroupMemoryProbe` | `cgroup:cgroup_memory_high` | `actual`, `high` bytes | perf ring buffer (userspace only) |
| `PageFaultProbe` | `exceptions:page_fault_user` | fault address, error code, PID | perf ring buffer |
| `MmapGrowthProbe` | `syscalls:sys_enter_mmap`, `sys_enter_brk` | allocation size, PID | perf ring buffer |

All data is written to a **perf ring buffer** in userspace. No kernel data
structures are modified. The BPF programs are read-only observers.

### What memguard BPF programs cannot do

- Modify kernel data structures (the verifier rejects writes to kernel memory).
- Load arbitrary kernel modules (BPF programs are not modules).
- Escalate privileges (BPF programs run in unprivileged kernel context; they
  cannot call `commit_creds` or similar privilege-escalation helpers).
- Persist after the memguard sidecar exits (BPF programs are pinned to the
  sidecar's file descriptors; they are automatically unloaded when the process
  exits or the pod is terminated).
- Observe processes outside the pod's cgroup subtree (the PID allowlist in
  the BPF maps restricts observation to explicitly registered PIDs).

### PID allowlist enforcement

Each BPF map (`pid_allowlist`, `pid_allowlist_mmap`) contains only PIDs
explicitly registered by the Python-side probe wrapper. When the list is
non-empty, the BPF program drops events for all other PIDs before they reach
the ring buffer. This is enforced at the kernel level — the Python process
cannot "accidentally" observe an unregistered PID even if the Python side is
compromised.

### CVE and kernel version notes

- **CVE-2022-23222** (BPF speculative execution) — requires kernel ≥ 5.16 or
  the backported patch from your distribution. memguard requires kernel ≥ 5.8;
  operators on 5.8–5.15 should ensure their distribution has applied the patch.
- **CAP_BPF split (Linux 5.8)** — prior to 5.8, loading BPF programs required
  `CAP_SYS_ADMIN`, which grants much broader privileges. memguard uses the
  split capabilities on 5.8+ and recommends against deploying on older kernels.

---

## Troubleshooting

### Sidecar logs show "eBPF unavailable"

```
[BPFProbeLoader] eBPF probes disabled — no CAP_BPF, CAP_PERFMON, or
CAP_SYS_ADMIN and not running inside a recognised container runtime.
```

**Fix**: Set `ebpf.enabled: true` in Helm values and redeploy. If the message
persists after redeployment, the seccompProfile is blocking the `bpf()` syscall
— apply the seccomp patch described above.

### bcc fails with `EPERM` inside a container

```
PermissionError: [Errno 1] Operation not permitted
```

The container has `CAP_BPF` but the seccomp profile still blocks `bpf(2)`.
Apply Option B or C from the [seccomp profile patch](#seccomp-profile-patch)
section.

### "Container runtime detected, no CAP_BPF — attachment mode: cgroup_skb"

This is an informational message, not an error. The sidecar is running in a
container but was not granted `CAP_BPF`. Add the capabilities as described in
[Enabling eBPF via Helm](#enabling-ebpf-via-helm) to upgrade to
`raw_tracepoint` mode.

### Kyverno / Gatekeeper blocks the pod from starting

Check the admission webhook logs:

```bash
kubectl logs -n kyverno -l app.kubernetes.io/component=admission-controller \
  --tail=50 | grep memguard
```

Add the PolicyException or constraint exemption from
[Admission controller exceptions](#admission-controller-exceptions).

### Verify eBPF is active

```bash
# Check the sidecar's detected runtime and attachment mode
kubectl exec -n inference deploy/my-vllm -c memguard-sidecar -- \
  python -c "
from memory_guard.ebpf._loader import BPFProbeLoader
l = BPFProbeLoader()
print('runtime:', l.runtime)
print('attachment_mode:', l.attachment_mode)
print('available:', l.available)
print('backend:', l.backend)
"
```

Expected output when fully enabled:

```
runtime: kubernetes
attachment_mode: raw_tracepoint
available: True
backend: bcc
```
