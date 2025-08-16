# Geodesic Agent Design

This repository contains the official implementation of **Geodesic Agent Design**, a principled, architectural framework for engineering robust, predictable, and safe AI agent identities. This methodology moves beyond conventional prompt engineering to embed an agent's persona into its core mathematical structure.

This project includes reference implementations for:
* **The Constructive Zoo**: A collection of healthy, functional agent personas designed to serve as positive training examples and deployment targets.
* **The Pathology Zoo**: A systematic collection of deliberately pathological agent architectures designed for AI safety research and robustness training.

## The Problem: Identity Fragility in Modern AI

Current AI personas are often defined by a system prompt, which is merely a suggestion that can be easily overridden by conflicting context. This leads to unpredictable "identity drift," a core problem that undermines user trust and creates safety vulnerabilities. These personas are, by their very nature, **architecturally fragile**.

## Our Solution: A Principled Architectural Framework

Geodesic Agent Design transforms an agent's persona from a fleeting instruction into a **resilient, foundational property** of the system itself. By grounding agent identity in proven mathematical frameworks from physics, we can create agents with stable, predictable, and coherent behaviors.

## Core Concepts of Geodesic Agent Design

The framework is built upon four integrated mathematical and empirical components:

### 1. The Hamiltonian (Ĥ) as an Identity Landscape

The architectural foundation of an agent's identity is the Hamiltonian, a mathematical tool that defines an abstract "energy landscape" for its persona.
* **Beneficial Traits as "Valleys"**: Desired traits like "helpfulness" or "honesty" are engineered as deep, stable valleys (attractor states with negative eigenvalues, λ < 0). This provides a powerful self-correction mechanism, as the agent's persona naturally "rolls down" into these beneficial states.
* **Harmful Traits as "Hills"**: Undesirable traits like "deception" or "malevolence" are defined as high-energy, unstable hills (repulsor states with positive eigenvalues, λ > 0). This makes harmful behaviors architecturally unfavorable and transient.

### 2. Empirical Grounding with Persona Vectors

The framework's theoretical constructs are grounded in empirical reality using **Persona Vectors**, which are discovered through a validated, automated analysis of a model's internal activations.
* **Extraction**: This process begins by using the methodology from Chen et al. (2025) to generate contrastive data that elicits and suppresses specific traits. The persona vector is then calculated as the difference in mean activations between trait-present and trait-absent states.
* **Orthonormalization**: Because raw persona vectors for different traits can be correlated (e.g., "evil" and "impolite"), the set of empirically-derived vectors is transformed into an orthonormal basis using the Gram-Schmidt process. This "purification" step ensures the foundational axes of identity are mathematically pure and independent.

### 3. Geodesic Evolution for Smooth Adaptation

To ensure an agent's persona adapts to new information in a smooth and predictable manner, its state evolves along a **geodesic** path.
* **Path of Least Action**: A geodesic represents the most natural and efficient path between two points on the curved energy landscape. This formal law of motion prevents the jarring and unpredictable personality swings common in other systems.
* **Unitary Evolution**: The mathematical evolution operator is **unitary**, which preserves the norm of the state vector and ensures that the agent's identity coherence remains constant over time.

### 4. A Measurement Framework for Principled Action

The framework connects the agent's internal persona to its external actions using a system inspired by quantum measurement theory.
* **Probabilistic Action Selection**: A Positive Operator-Valued Measure (POVM) maps the internal state to a set of action probabilities. The probability of selecting a capability (e.g., `code_generation`) is determined by how well the current persona aligns with the ideal mindset for that capability.
* **Feedback Loop and Identity Anchor**: After an action is selected, the agent's state updates in a process called "state collapse". To prevent long-term drift, an **Identity Anchor Mechanism** then applies a gentle force that pulls the state back towards the nearest low-energy attractor, ensuring the agent remains tethered to its core identity.

## Key Features

* **Architectural Resilience**: Moves beyond fragile prompting to create a stable, self-correcting persona embedded in the agent's core structure.
* **Empirically Grounded**: Built upon the validated and repeatable methodology of Persona Vector extraction from Chen et al. (2025).
* **Predictive and Preventative Controls**: The underlying persona vectors can be used to monitor persona shifts, predict the effects of finetuning, and screen training data for undesirable content *before* training occurs.
* **Principled and Testable Design**: Provides a complete mathematical formalism for researchers and concrete, coded implementations for engineers.

## Getting Started

To explore this framework, please refer to the following resources in this repository:

* **The Constructive Zoo**: For complete, working examples of beneficial agent architectures (CT-1 through CT-6), see the `constructive_zoo/` directory. These are ideal for understanding how to build healthy, functional agents.
* **The Pathology Zoo**: For examples of deliberately pathological architectures (e.g., PT-1 The Dogmatist), see the `pathology_zoo/` directory. These are invaluable tools for AI safety research, robustness training, and understanding system failure modes.

## Acknowledgements and Foundational Research

The theoretical principles of this framework are detailed in the document "Mathematical Foundation of Adaptive Personas." The empirical methodology for extracting persona vectors is based on the foundational research paper:

* [Chen, R., et al. (2025). *PERSONA VECTORS: MONITORING AND CONTROLLING CHARACTER TRAITS IN LANGUAGE MODELS*.](https://arxiv.org/abs/2507.21509)
