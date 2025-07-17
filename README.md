# Attack Graph-Integrated Intrusion Detection System Prototype

This project demonstrates the lifecycle of integrating attack graph modeling with intrusion detection systems (IDS). It leverages vulnerability databases, network inventories, rule-based alerts, and attack graph (AG) analysis to enhance threat detection and response capabilities.

## Project Structure

main.py
├── buildIDSnet.py
├── attackgraph/
│ └── agBuilder.py
├── ids/
│ └── run_all.py
├── data/
│ ├── networks/
│ ├── vulns.json
│ ├── vulnsAttack.json
│ └── TrafficLabelling/
└── emerging_rules/

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

## Requirements

- Python 3.x
- JSON files with network and vulnerability data
- Pre-downloaded NVD datasets and Snort rules

## How to Run

```bash
python main.py


```
