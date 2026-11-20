# Mitigating LLM Hallucinations using Multi-Agent Collaborative Filtering with AMR, Experience Retrieval and Answer Consensus

A local, multi-agent framework for evaluating and selecting answers to mathematical word problems. The project uses multiple LLM agents, semantic and Abstract Meaning Representation (AMR) similarity, an adversarial challenge agent, and an improved answer selector. An optional Experience-MCF pipeline retrieves related examples from a repository built from GSM8K training data.

The implementation runs locally with [Ollama](https://ollama.com/).

## 1. Project overview

The project evaluates two related pipelines:

- **Baseline MCF:** three normal reasoning agents solve a problem independently. A fourth, adversarial agent produces a plausible but intentionally flawed solution. Each normal response is compared with the adversarial response using Nomic embedding cosine similarity and AMR Smatch F1. The improved selector then chooses a valid answer using answer validation, consensus, and arithmetic-trace checks.
- **Experience-MCF:** follows the same pipeline, but first checks whether the question is an outlier and, when it is not, retrieves up to three semantically similar validated training experiences. These examples are added to the prompts of the three normal agents. The adversarial agent's prompt is unchanged.


## 2. Architecture

```text
                         GSM8K test question
                                  |
                  +---------------+---------------+
                  |                               |
          Baseline MCF                     Experience-MCF
                  |                               |
          Three normal agents           Outlier detection
                  |                               |
                  |                       Retrieve up to 3
                  |                    validated train examples
                  |                               |
                  +---------------+---------------+
                                  |
                    Three normal-agent responses
                                  |
                    Adversarial challenge agent
                                  |
                 Compare each normal response with
                    the adversarial response
                                  |
                 +-------------------------------+
                 | Nomic embedding cosine score  |
                 | AMR parsing and Smatch F1     |
                 +-------------------------------+
                                  |
                    Improved MCF answer selector
                                  |
            Valid-answer filtering and answer consensus
                 Arithmetic-trace verification (when
                       a parseable trace exists)
                                  |
                    Selected numeric prediction
                                  |
                   Evaluation against GSM8K gold
                                  |
                         CSV and JSON results
```

## 3. Main components

| File | Responsibility |
|---|---|
| `src/amr_mcf_filter.py` | Runs the baseline AMR + embedding MCF pipeline. |
| `src/amr_mcf_with_experience.py` | Runs Experience-MCF with outlier detection and retrieval. |
| `src/improved_mcf_selector.py` | Shared improved selection logic used by both pipelines. |
| `src/experience_repository.py` | Stores, embeds, validates, and retrieves experiences using Ollama embeddings. |
| `src/populate_experience_repository.py` | Builds the experience repository from the GSM8K training split. |
| `src/amr_client.py` | Earlier standalone AMR/Smatch helper; not required by the two main scripts, which define/use the AMR client in `amr_mcf_filter.py`. |
| `src/embedding_client.py`, `src/llm_client.py` | Supporting clients used by earlier/experimental scripts. |
| `src/adversarial_agent.py` | Earlier standalone adversarial-agent implementation. |

## 4. Technology stack

- **Python** – experiment orchestration and evaluation.
- **Ollama** – local LLM and embedding inference.
- **Llama 3.2 3B (`llama3.2:3b`)** – normal and adversarial agent responses.
- **Nomic Embed Text (`nomic-embed-text`)** – text embeddings for semantic similarity and experience retrieval.
- **AMRLib** – parses text into Abstract Meaning Representation graphs.
- **Smatch** – computes graph similarity using Smatch F1.
- **NumPy** – vector and cosine-similarity calculations.
- **GSM8K** – grade-school mathematical word-problem dataset.

## 5. Requirements

- Windows, Linux, or macOS.
- Python environment compatible with the installed PyTorch/AMRLib dependencies. Python 3.11 or 3.12 is a practical starting point; compatibility depends on the versions available for your platform.
- [Ollama](https://ollama.com/) installed and running locally.
- The following Ollama models:
  - `llama3.2:3b`
  - `nomic-embed-text`
- The AMRLib `model_stog` model must be available to AMRLib.
- The project files, including the local GSM8K test file and the prepared training-experience repository.

Install the Python packages:

```bash
python -m pip install --upgrade pip
python -m pip install ollama numpy amrlib smatch datasets
```

`datasets` is needed only when rebuilding the experience repository from the Hugging Face GSM8K training split. If you only run the two pipelines with the supplied local data and repository, it is not needed by the main runtime path.

Download the AMRLib STOG model using:

```bash
python -c "import amrlib; amrlib.download_model('model_stog')"
```

If this command is not supported by the installed AMRLib version, follow the model-download instructions for that installed version and ensure `amrlib.load_stog_model()` can locate `model_stog`.

## 6. Ollama setup

Start Ollama, then pull the required models if they are not already installed:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Check the installed models:

```bash
ollama list
```

The expected model names include `llama3.2:3b` and `nomic-embed-text` (the latter may appear with the `:latest` tag).

Check that Ollama responds:

```bash
ollama run llama3.2:3b "What is 15 multiplied by 12? Give only the answer."
```

Expected answer: `180`.

## 7. Project structure

```text
LLM-Halucination-MCF/
├── data/
│   └── gsm8k_test.jsonl
├── results/
│   ├── amr_mcf_improved_results.csv
│   ├── amr_mcf_improved_raw_responses.json
│   └── experience_mcf_improved/
│       ├── amr_mcf_experience_results.csv
│       └── amr_mcf_experience_raw_responses.json
├── src/
│   ├── amr_mcf_filter.py
│   ├── amr_mcf_with_experience.py
│   ├── improved_mcf_selector.py
│   ├── experience_repository.py
│   └── populate_experience_repository.py
└── storage/
    └── gsm8k_train_experiences.json
```


## 8. Running the project


### 8.1 Run baseline MCF

```bash
python src/amr_mcf_filter.py
```

The script evaluates questions from `data/gsm8k_test.jsonl`. It saves progress after each completed question and skips question IDs already present in its raw result file. To perform a genuinely fresh run, back up and remove both baseline output files before executing the command:

```text
results/amr_mcf_improved_results.csv
results/amr_mcf_improved_raw_responses.json
```

A fresh run generates new LLM responses and may produce different results.

### 8.2 Run Experience-MCF

```bash
python src/amr_mcf_with_experience.py --limit 50
```

The script expects:

```text
data/gsm8k_test.jsonl
storage/gsm8k_train_experiences.json
```

It also resumes from its existing output files. To start a fresh Experience-MCF run, use:

```bash
python src/amr_mcf_with_experience.py --limit 50 --fresh
```



## 9. How the improved selector works

For each of the three normal-agent responses, the selector:

1. Extracts and normalizes the numeric final answer.
2. Marks an answer invalid if a usable numeric answer cannot be extracted.
3. Checks arithmetic equations in the response when the verifier can parse them. The verifier uses a restricted Python AST arithmetic evaluator; it does not use unrestricted `eval()`.
4. Groups valid answers and identifies consensus.
5. Applies the following selection priority:
   - A consensus answer supported by at least one mathematically verified trace.
   - Any other valid answer supported by at least two agents.
   - A single mathematically verified answer.
   - The valid candidate with the lowest original combined AMR/embedding score.
6. Returns no prediction if all three normal-agent answers are invalid.

The gold answer is used to calculate evaluation accuracy after the prediction is selected. It is not used by the selector to choose an answer.

The arithmetic verifier is a lightweight trace checker, not a complete mathematical proof system. If an answer has no parseable arithmetic equations, its trace may be marked `unverified`; an `invalid_trace` status should also be interpreted cautiously because natural-language equation formatting can challenge the parser.

## 10. Similarity calculation

For each normal response and the adversarial response:

- Nomic embeddings are generated for both texts.
- Cosine similarity is calculated.
- The cosine score is normalized from `[-1, 1]` to `[0, 1]` using `(cosine + 1) / 2`.
- AMR graphs are generated and compared using Smatch F1.
- The two normalized similarity scores are combined with equal weights:

```text
combined_score = 0.5 * normalized_embedding_similarity
               + 0.5 * AMR_Smatch_F1
```

The original MCF candidate is the normal response with the minimum combined similarity to the adversarial response. The improved selector retains this score as a fallback after validity, consensus, and verification checks.


## 11. Output files

| Output | Description |
|---|---|
| `results/amr_mcf_improved_results.csv` | Per-question baseline metrics and selected answers. |
| `results/amr_mcf_improved_raw_responses.json` | Full baseline agent responses and intermediate scores. |
| `results/experience_mcf_improved/amr_mcf_experience_results.csv` | Per-question Experience-MCF metrics, retrieval information, and selected answers. |
| `results/experience_mcf_improved/amr_mcf_experience_raw_responses.json` | Full Experience-MCF responses and retrieved experience details. |

The CSV files are convenient for accuracy calculations and spreadsheet analysis. The JSON files preserve more detailed responses and intermediate information.

## 12. Limitations

- The project uses a relatively small local language model (`llama3.2:3b`), so reasoning quality and output consistency are constrained by the model.
- The adversarial agent is prompted to make a plausible mistake; it is not guaranteed to produce a flawed answer on every question.
- Embedding and AMR similarity are proxies for response relationships and do not independently prove correctness.
- The arithmetic verifier only checks equations it can parse and is not a general-purpose proof checker.
- Experience retrieval did not return examples for every evaluated question.

