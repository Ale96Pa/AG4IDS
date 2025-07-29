# **Attack Graph-Integrated Intrusion Detection System Prototype**

---

## Abstract

Cyber attack detection and response are pivotal to ensuring a robust cybersecurity posture in any organization. This task becomes even more challenging in complex network environments, where attackers may follow multiple sophisticated paths to compromise critical services.

To support threat detection, **Intrusion Detection Systems (IDSs)** are widely used to identify anomalous behaviors. In parallel, **Attack Graphs (AGs)** offer a powerful model for analyzing attacker strategies and informing response decisions.

Traditionally, IDSs and AGs have been treated in isolation, each facing limitations such as high false positive rates, poor scalability, and limited adaptability to evolving threats. Although recent research has begun exploring their integration, a comprehensive and systematic analysis is still missing.

This project addresses that gap by:

- Reviewing **\totalcount{}** relevant works
- Proposing a novel **taxonomy** of IDS–AG integration:
  - **AG-based IDS refinement**
  - **AG-integrated IDSs**
  - **IDS-based AG generation**
  - **Hybrid approaches**

Our findings reveal that most current methods rely on static assumptions, overlooking the dynamic and evolving nature of real-world threats. To address this, we introduce a new **IDS–AG lifecycle** that supports continuous detection and response. We also provide a simple prototype implementation to demonstrate its benefits and highlight future directions for adaptive and resilient network security.

---

## Project Structure

```plaintext
main.py
├── buildIDSnet.py
├── attackgraph/
│   └── agBuilder.py
├── ids/
│   └── run_all.py
├── data/
│   ├── networks/
│   ├── vulns.json
│   ├── vulnsAttack.json
│   ├── TrafficLabelling/
│   └── emerging_rules/
└── results/
```

The folder `attackgraph` includes all the files for the AG generation simulations; the folder `ids` includes all the files for the AG-integrated IDS and IDS refinement simultations; the folder `data` includes all the datasets; the folder `results` reports the results for reproducibility.

## Requirements

- Python 3.x
- JSON files with network and vulnerability data
- Pre-downloaded NVD datasets and Snort rules

## Pipeline Overview

### 1. Dataset Preprocessing

- Prepare initial network topology from `CiC17Net.json` using:
  - `internal_net()`: internal mapping of original topology.
  - `get_dump_nvd()`: downloads and stores NVD vulnerabilities in `vulns.json`.
  - `getVulnsByService("ubu16")`: filters CVEs for Ubuntu 16.
  - `generate_devices()`: generates device profiles with attached vulnerabilities.

### 2. Rule-Based Vulnerability Mapping

- Uses `get_dump_cveList()` to extract CVEs from emerging Snort rules.
- Generates `vulnsAttack.json` representing attack-based vulnerabilities.
- Applies `getVulnsByAlert()` to refine network inventories based on alert traces.

### 3. Attack Graph Generation

For each network variant:

- `build_multiag()`: builds attack graph and outputs `.graphml` file.
- `compute_paths()`: identifies exploitation paths from sources (e.g., `kali`, `fw`, `win81`) to defined goals.
- `plot_risk()`: visualizes network-wide risk exposure using likelihood metrics.

### 4. AG-Integrated IDS Execution

- Runs IDS simulations via:
  - `run_all_blue_box()`: baseline evaluation.
  - `run_all_blue_box_parallel()`: parallelized attack scenarios.
  - `run_all_blue_box_controlled()`: controlled IDS benchmarks.

### 5. AG-Based IDS Refinement

- Applies AG feedback loops for enhanced rule triggering and reduction of false positives:
  - `run_all_orange_box()`
  - `run_all_orange_box_parallel()`
  - `run_all_orange_box_controlled()`

## How to Run

```bash
python main.py


```
