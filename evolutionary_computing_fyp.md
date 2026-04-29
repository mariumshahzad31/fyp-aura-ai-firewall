Evolutionary Computing Project Name:
"Evolutionary Cybersecurity Enhancement Layer"

# Introduction

This document describes the design and integration of an evolutionary computing enhancement layer for the AURA behavioral firewall. The project employs Genetic Algorithms (GA), Particle Swarm Optimization (PSO), and a multi-objective fitness function to tune decision policies that work on top of existing machine learning models without modifying them.

# Problem Statement

Modern cybersecurity systems must balance threat detection performance with false positive control, false negative reduction, and operational latency. Existing AURA models already provide strong prediction capabilities, but there is a need for an adaptive policy layer that can select the most important signals and tune thresholds in response to evolving adversarial patterns.

# Solution

The proposed solution is an evolutionary computing layer that treats the firewall policy as a search problem. A Genetic Algorithm performs discrete feature selection and rule configuration, while Particle Swarm Optimization fine-tunes continuous thresholds and risk weights. A multi-objective fitness function evaluates policy candidates using accuracy, false positive rate, false negative rate, latency, and stability.

# Objectives

- Integrate GA and PSO into AURA as an additional optimization layer.
- Preserve existing Random Forest, Isolation Forest, and Gradient Boosting models unchanged.
- Improve accuracy while reducing false positives, false negatives, and runtime overhead.
- Provide explainable evolutionary policy outputs for system monitoring and alerts.
- Enable Streamlit UI, FastAPI, and CLI compatibility with the new layer.

# Evolutionary Approach (GA + PSO + Fitness Function)

The evolutionary approach is composed of two complementary phases:

- Genetic Algorithm (GA): performs bit-level feature selection and discrete policy rule optimization, including selection, crossover, mutation, and elitism.
- Particle Swarm Optimization (PSO): performs continuous optimization of fusion weights, score thresholds, and gating flags with inertia, cognitive, and social velocity updates.
- Multi-objective Fitness Function: computes a fitness score from accuracy, false positive rate, false negative rate, latency, and stability using configurable weights.

Together, these methods enable policy candidates to evolve from a baseline decision rule into a tuned configuration that maximizes threat detection effectiveness while minimizing operational risk.

# Technologies Used

- Python 3
- NumPy for numerical operations
- Pandas for dataset handling
- Scikit-learn for existing predictive models and preprocessing
- FastAPI for backend API exposure
- Streamlit for user interface visualization
- JSON and file management for policy persistence
- Custom evolutionary modules for GA, PSO, and fitness evaluation

# Integration with AURA

The evolutionary enhancement layer is integrated in a non-invasive manner:

- Existing ML models remain intact and are used for inference as before.
- A new `evo_integration.py` module implements the optimization layer and decision policy.
- The `utils/orchestration.py` module loads the evolutionary policy conditionally and applies it to prediction outputs.
- Policy tuning can be executed via CLI and persisted to the `saved_models` directory.
- The FastAPI backend and Streamlit UI share the same policy integration through the underlying orchestration.

# Advantages (Pros)

- Enhances performance without retraining core models.
- Provides an adaptive decision layer for risk tuning.
- Reduces operational false positives and false negatives.
- Retains compatibility with existing AURA infrastructure.
- Produces logging for GA and PSO convergence, policy improvement, and parameter selection.

# Disadvantages (Cons)

- Adds computational overhead during tuning phases.
- Requires representative baseline data for effective optimization.
- Policy effectiveness depends on the quality of AURA outputs and behavioral signals.
- The additional layer adds complexity to debugging and maintenance.

# Why Evolutionary Computing

Evolutionary computing is an appropriate choice for this project because the policy search space is heterogeneous and includes both discrete and continuous decisions. Genetic Algorithms excel at selecting feature sets and logical rule combinations, while Particle Swarm Optimization is well suited for tuning continuous thresholds and weights. The combined approach supports a multi-objective evaluation that balances detection quality with operational constraints.

# Advancements in FYP using Evolutionary Computing

This project advances the final year project by introducing a robust optimization framework on top of existing intelligence. The evolutionary layer transforms static model outputs into a policy that is automatically tuned for improved cybersecurity outcomes. It adds a scientific evaluation methodology and provides concrete performance gains through quantitative fitness tracking.

# Evaluation Methodology

Evaluation uses a representative dataset of firewall and attack records. The evolutionary layer is validated by:

- Generating a scored dataset from existing AURA model outputs.
- Running GA to optimize discrete policy toggles and rule structure.
- Running PSO to tune continuous thresholds and weights.
- Comparing baseline and tuned fitness, accuracy, false positive rate, false negative rate, and latency.
- Saving the optimized policy for use in production.

# Results

The integrated evolutionary enhancement layer yields measurable improvement over the baseline policy. It produces a tuned decision policy with stronger detection accuracy, reduced false positives, reduced false negatives, and acceptable latency. Full reports include GA evolution history, PSO convergence history, and final optimized parameter values.

# Conclusion

The Evolutionary Cybersecurity Enhancement Layer successfully integrates GA, PSO, and a multi-objective fitness function into the AURA ecosystem. It preserves the existing machine learning foundation while adding an adaptive optimization layer. The design supports safe deployment across Streamlit, FastAPI, and CLI workflows, and it provides explainable tuning outputs to improve overall threat detection performance.
