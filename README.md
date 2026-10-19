# LLM Hallucination Mitigation using Multi-Agent Collaboration

## Overview

This project explores whether **Multi-Agent Collaboration (MCF)** can improve the accuracy of Large Language Models (LLMs) on mathematical reasoning tasks and help mitigate incorrect or hallucinated answers.

The project uses locally hosted LLMs through Ollama, multiple reasoning agents, embedding similarity, and Abstract Meaning Representation (AMR)-based similarity to select answers from candidate responses.

Experiments are conducted using the GSM8K mathematical reasoning dataset.

## Objectives

- Evaluate the performance of a single LLM on mathematical reasoning tasks.
- Generate multiple candidate solutions using normal reasoning agents.
- Introduce an adversarial agent to challenge candidate reasoning.
- Use embedding similarity and AMR-based similarity for candidate selection.
- Compare MCF with majority voting and baseline approaches.
- Investigate experience-based retrieval and outlier detection as extensions to the existing workflow.

## Technology Stack

- **Programming language:** Python
- **LLM runtime:** Ollama
- **Reasoning model:** `llama3.2:3b`
- **Embedding model:** `nomic-embed-text`
- **Dataset:** GSM8K
- **Similarity methods:** Embedding cosine similarity and AMR/Smatch-based similarity
- **Data analysis:** CSV and JSON experiment outputs

## System Architecture

The current experimental workflow consists of the following stages:

1. **Input:** Load a mathematical reasoning question from GSM8K.
2. **Normal agents:** Generate candidate solutions using multiple reasoning prompts.
3. **Adversarial agent:** Challenge the candidate reasoning and help identify potential inconsistencies.
4. **Similarity computation:** Compare agent instructions or responses using embeddings and, where applicable, AMR-based similarity.
5. **MCF selection:** Apply the implemented scoring and selection strategy to choose a candidate answer.
6. **Evaluation:** Compare the predicted numerical answer with the reference answer.
7. **Result storage:** Save experimental outputs for further analysis.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/AabelThankachan/LLM-Halucination-MCF.git
cd LLM-Halucination-MCF
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install project dependencies

Install the packages required by the project using the dependency versions specified in your environment or dependency file.

If a `requirements.txt` file is not present, create one from the working environment after checking which packages the source code actually imports.

### 4. Install and start Ollama

Install Ollama from:

https://ollama.com/

Pull the required models:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Check the installed models:

```bash
ollama list
```

Ensure Ollama is running before executing the experiments.

### 5. Download the dataset

Run the dataset download script:

```bash
python src/download_dataset.py
```

Follow the dataset preparation instructions in the source code. Keep training and evaluation data separate to avoid test-data leakage.

## Running the Experiments

Run the baseline experiment:

```bash
python src/baseline.py
```

Run the adversarial-agent experiment:

```bash
python src/adversarial_agent.py
```

Generate analysis summaries from the saved experiment results:

```bash
python src/llm_mcf_next_steps.py
```

The analysis script reports method-level accuracy, paired comparisons, candidate coverage, and saved-response ablation results.

Use the actual script names and configuration values in the repository when reproducing individual experiments.

## Experimental Results

The following results are from the current 50-question evaluation.

| Method | Accuracy | Correct answers |
|---|---:|---:|
| Majority voting | 78% | 39/50 |
| Embedding + AMR MCF | 76% | 38/50 |
| Embedding-only MCF | 68% | 34/50 |
| Best-agent oracle ceiling | 88% | 44/50 |

**Interpretation:**

- Majority voting achieved the highest accuracy among the listed evaluated methods.
- Embedding + AMR MCF performed better than embedding-only MCF in the reported results.
- The oracle ceiling indicates that at least one normal agent generated a correct candidate for 44 of the 50 questions.
- The oracle result is an upper-bound analysis, not an implemented prediction method.
- Results should be compared as a controlled ablation only when the methods use the same saved agent responses.

These are preliminary results from a 50-question evaluation and should not be interpreted as a guarantee of performance on the full GSM8K dataset.

## Results and Output Files

The project stores experiment outputs in the `results/` directory.

Important files include:

- `results/amr_mcf_results.csv` — embedding + AMR MCF results.
- `results/mcf_results.csv` — embedding-only MCF results.
- `results/method_comparison.csv` — method comparison results.
- `results/analysis/` — generated experiment-analysis reports and summaries.

The exact output files depend on which experiments have been executed.

## Limitations

- The local model may produce incorrect or inconsistent mathematical reasoning.
- Candidate selection can fail even when at least one agent has the correct answer.
- Answer-verifier experiments have not consistently improved accuracy.
- Performance on GSM8K does not establish general hallucination-mitigation performance across other tasks.
- The experience-repository workflow and associated outlier detection require separate implementation and evaluation before they can be reported as completed features.

## Future Work

- Implement a persistent experience repository for previously solved tasks.
- Retrieve relevant experiences using semantic similarity.
- Investigate Z-score-based outlier detection before retrieval.
- Integrate experience retrieval into the existing multi-agent workflow.
- Evaluate the impact of each component using controlled ablation studies.
- Extend evaluation beyond the initial 50-question sample.
- Record accuracy, latency, errors, and retrieval statistics for reproducibility.

## Reproducibility

For meaningful comparisons, use the same evaluation question IDs, consistent answer extraction, and identical saved agent responses wherever possible. Record model names, prompts, configuration values, and experiment outputs.

Do not use held-out test answers to populate the experience repository or select examples for those same test questions.

## Author

**Aabel Thankachan**

GitHub: https://github.com/AabelThankachan
