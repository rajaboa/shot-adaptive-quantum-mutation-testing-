# Shot-Adaptive Quantum Mutation Testing

Optimizing mutation testing of quantum programs through **circuit-level mutation** and **adaptive shot allocation** with equal-size batching and dual early stopping, reducing both per-mutant overhead and total shots while preserving kill/survival accuracy.

## Overview

Quantum mutation testing generates mutant programs by applying small syntactic changes (gate deletion, gate replacement, etc.) and then checks whether test cases can distinguish the mutant from the original by comparing output distributions. On simulators or real quantum hardware, each comparison requires running the circuit many times (shots) to sample the distribution. This creates two performance bottlenecks:

1. **Per-mutant overhead**: Traditional AST-level mutation requires parsing, compiling, and transpiling each mutant circuit — transpilation alone accounts for ~95% of per-mutant time.
2. **Shot allocation**: Fixed-shot strategies waste shots on mutants that could be decided (killed or declared equivalent) much earlier.

This framework addresses both:

### Circuit-Level Mutation

Instead of mutating Python source code via AST transformations (as QMutPy does), the circuit-level engine operates directly on `QuantumCircuit.data` — the list of `CircuitInstruction` objects in an already-transpiled circuit. This bypasses:

- **AST parsing and code generation** — no `ast.parse()`, `compile()`, `exec()`
- **Module compilation** — no dynamic module creation per mutant
- **Per-mutant transpilation** — mutations are applied to the already-transpiled circuit

Gate equivalence is determined by `(num_qubits, num_params)` rather than Python function argument count, and replacements are filtered to backend-supported gates to avoid incompetent mutants.

**Result**: ~500x speedup per mutant (1.4ms vs 773ms) with 100% competent rate vs ~65% for AST-level.

### Adaptive Shot Strategy

Instead of running a fixed number of shots for every mutant, the adaptive strategy uses **equal-size batches** (e.g., 200 shots x 10 batches = 2000 total) with dual early stopping:

- **Kill early-stop**: Chi-squared goodness-of-fit test rejects H0 at any checkpoint (Bonferroni-corrected alpha for intermediate checks, uncorrected for the final check)
- **Equivalence early-stop**: Statistical power analysis (non-central chi-squared) determines that there is sufficient power to detect a minimum effect size, but H0 was not rejected — the mutant is likely equivalent

With 10 equal checkpoints, strong kills are detected in as few as 200 shots, while only borderline cases use the full 2000-shot budget. Bonferroni correction uses alpha_adj = alpha/k for intermediate checks (k=10 checkpoints), with uncorrected alpha at the final check.

### Accuracy Techniques

Achieving high agreement between fixed and adaptive strategies required several refinements, applied progressively:

**1. Precise baseline distributions (100k shots)**
Expected distributions are computed once with 100,000 shots, giving ~±0.001 precision per outcome. Using only 10,000 shots left rounding errors that distorted chi-squared statistics and caused false kills. Impact on QAOA: agreement rose from 90.8% to 97.8%.

**2. Inconclusive verdict (power analysis for fixed strategy)**
The fixed strategy now computes statistical power after a non-rejection. If power ≥ threshold, the mutant is declared **survived** (likely equivalent); if power < threshold, it is **inconclusive** (insufficient evidence either way). The adaptive strategy similarly marks exhausted-budget results as inconclusive when power is low. Inconclusive results are excluded from agreement counting. In practice, power is ≥ 0.99 at 2000 shots for our distributions, so inconclusive does not appear — but the infrastructure is in place for smaller budgets.

**3. Strict significance level (alpha = 0.01)**
Using alpha = 0.05 produced 5 disagreements on QAOA, all borderline kills at p = 0.014–0.037 where random sampling could flip the verdict. Lowering to alpha = 0.01 reclassified these as survived (or equivalently scored by both strategies), reducing disagreements to 1 at no extra cost. Impact: 97.8% → 99.6% agreement.

**4. Majority-vote trials (n_trials = 3)**
The 1 remaining disagreement was a case where the adaptive drew one lucky deviant 400-shot sample out of a single run. Running each strategy 3 times independently and taking the majority verdict (killed if ≥ 2/3 agree) eliminated this noise. The adaptive still benefits from early stopping within each trial, so shot savings are preserved. Impact: 99.6% → 100% mutant-level agreement at 3× cost.

## Results

### Bernstein-Vazirani (BV)
Deterministic circuit (single outcome with probability 1.0). Killed mutants produce impossible outcomes, detected instantly regardless of alpha or n_trials. Survived mutants re-insert gates that preserve the output.

| Metric | Fixed | Adaptive |
|--------|-------|----------|
| Killed | 246 (84.0%) | 246 (84.0%) |
| Survived | 47 (16.0%) | 47 (16.0%) |
| Total shots | 690,000 | 148,200 |
| Shot savings | — | **78.5%** |
| Time savings | — | **47.2%** |
| Mutant agreement | **293/293 (100%)** | |

Config: alpha=0.01, n_trials=1, 2000 shots max, 200/batch.

### QAOA MaxCut
Multi-outcome distribution (9–16 outcomes) from a Quantum Approximate Optimization Algorithm on a 4-qubit ring graph. Tests the statistical machinery on probabilistic circuits where the chi-squared test must distinguish subtle distribution shifts.

| Metric | Fixed | Adaptive |
|--------|-------|----------|
| Killed | 192 (84.2%) | 192 (84.2%) |
| Survived | 36 (15.8%) | 36 (15.8%) |
| Total shots | 1,608,000 | 339,600 |
| Shot savings | — | **78.9%** |
| Time savings | — | **39.9%** |
| Mutant agreement | **228/228 (100%)** | |

Config: alpha=0.01, n_trials=3, 2000 shots max, 200/batch.

**Agreement improvement progression** (QAOA):

| Technique added | Agreement | Disagree |
|-----------------|-----------|----------|
| Baseline (10k shots, alpha=0.05, n=1) | 90.8% | 21 |
| + 100k baseline | 97.8% | 5 |
| + alpha=0.01 | 99.6% | 1 |
| + n_trials=3 majority vote | **100.0%** | 0 |

## Mutation Operators

Quantum-specific mutation operators from [QMutPy](https://github.com/AntoniEsteve/mutpy):

| Operator | Description |
|----------|-------------|
| **QGD** | Quantum Gate Deletion |
| **QGR** | Quantum Gate Replacement |
| **QGI** | Quantum Gate Insertion |
| **QMD** | Quantum Measurement Deletion |
| **QMI** | Quantum Measurement Insertion |

Two mutation modes are supported:

- **AST-level**: QMutPy mutates Python source code via AST transformations
- **Circuit-level**: Direct mutation of `QuantumCircuit.data`, bypassing AST parsing, module compilation, and per-mutant transpilation

## Project Structure

```
experiments/
  adaptive_runner.py      # Main experiment runner (CLI)
  circuit_mutator.py      # Circuit-level mutation engine
  programs/               # Quantum programs (Qiskit 2.x)
    bernstein_vazirani.py
    qaoa_maxcut.py
    deutsch_jozsa.py, simon.py, grover.py, shor.py
  test_cases/             # Declarative test suites (JSON)
    bv_test_cases.json
    qaoa_test_cases.json
  tests/                  # Unit tests for quantum programs
  results/                # Experiment outputs (JSON + Excel)
benchmarks/               # Original pre-migration programs and tests
mutpy/                    # QMutPy mutation testing framework
docs/                     # Reference papers
```

## Usage

```bash
# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
# or: source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install qiskit qiskit-aer scipy numpy openpyxl

# Run BV benchmark (circuit-level, skip baseline recomputation)
cd experiments
python adaptive_runner.py test_cases/bv_test_cases.json --no-baseline --circuit-level -v

# Run QAOA benchmark (circuit-level, skip baseline recomputation)
python adaptive_runner.py test_cases/qaoa_test_cases.json --no-baseline --circuit-level -v

# Recompute baseline distributions (needed after changing baseline_shots)
python adaptive_runner.py test_cases/qaoa_test_cases.json --circuit-level -v
```

## Configuration (JSON test suites)

```json
{
  "algorithm": "qaoa_maxcut",
  "source_file": "programs/qaoa_maxcut.py",
  "baseline_shots": 100000,
  "fixed_shots": 2000,
  "adaptive_batch_size": 200,
  "adaptive_max_shots": 2000,
  "alpha": 0.01,
  "equiv_min_effect": 0.3,
  "equiv_power_threshold": 0.8,
  "n_trials": 3,
  "test_cases": [...]
}
```

Key parameters:
- **baseline_shots**: Shots for computing expected distributions (one-time, 100k recommended)
- **fixed_shots**: Total shots for the fixed-shot strategy
- **adaptive_batch_size**: Shots per adaptive batch (equal-size batches)
- **adaptive_max_shots**: Maximum total shots for the adaptive strategy (batch_size x num_batches)
- **alpha**: Significance level for chi-squared test (0.01 recommended; Bonferroni-corrected for intermediate adaptive checks)
- **equiv_min_effect**: Minimum Cohen's w effect size for equivalence detection
- **equiv_power_threshold**: Statistical power threshold for equivalence early-stop; non-killed results below this are marked inconclusive
- **n_trials**: Number of independent runs per strategy; majority verdict taken (1 = single run, 3 = recommended for QAOA-class circuits)

## Output

- **Console**: Per-mutant table with verdicts, vote counts (n_trials > 1), early-stop info, and agreement
- **Excel** (3 sheets): Mutant Details (with Fixed Power column), Summary, Per-TestCase breakdown
- **JSON**: Full structured results for programmatic analysis

## References

- Fortunato, D., Campos, J.C., & Abreu, R. (2022). Mutation testing of quantum programs written in QISKit. *QP4SE Workshop, ASE 2022*.
- Fortunato, D., Campos, J.C. & Abreu, R. (2022). QMutPy: a mutation testing tool for quantum programs. *ICST 2024*.
