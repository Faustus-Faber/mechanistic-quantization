# Research Proposal: Mechanistic Interpretability of Quantization-Induced Failures

**Abstract:** To deploy Small Language Models (SLMs) onto edge devices, 4-bit quantization is essential, but this compression disproportionately causes models to hallucinate or "forget" capabilities, notably in non-English/code-switched languages. This work employs mechanistic interpretability to map the specific components (attention heads, SAE features, MLP gates) inside 3B parameter models (e.g., Phi-4, Gemma-3) that break down under quantization. We propose ultra-lightweight "Activation Patches" applied in the Sparse Autoencoder (SAE) feature space to causally rescue these failures without full model retraining.

### Proposed Methodology

1. **Failure Mode Profiling:** Compare unquantized (16-bit) and 4-bit quantized versions of 3B models across multilingual benchmarks to isolate specific token sequences where quantization causes a behavioral flip.
2. **Layer-by-Layer Causal Tracing:** Utilize Attribution Patching (AtP) and activation patching to localize the exact layers and components responsible for precision degradation. 
3. **Monosemantic Subspace Intervention:** Use pre-trained Sparse Autoencoders (SAEs) to map polysemantic neurons into monosemantic features. Identify the specific feature directions responsible for the failure (e.g., "French syntax feature") and compute a corrective activation vector based on the 16-bit model's activations.
4. **Targeted Activation Patching:** Design a lightweight "Activation Patch"—a sparse, conditional bias vector injected into the forward pass—to rescue broken circuits at inference time.
5. **Efficiency Compilation:** To avoid inference overhead, compile the Activation Patch into a localized low-rank adapter (LoRA) or conditionally route it only when specific task-id tokens are present.

### Expected Contributions
- **First Deep-Dive into Compression Failures:** Providing a causal mechanistic analysis of *why* low-bit quantization fails on non-English syntax and underrepresented reasoning tasks.
- **Precision Feature Steering:** Demonstrating that intervention in SAE feature space can rescue targeted capabilities (like minority languages) without catastrophic interference on structured tasks like coding.
- **Inference-Time Fix:** A computationally cheap activation patching technique to restore unquantized capabilities in highly compressed models without full continuous retraining.

### Feasibility & Hardware Constraints (RTX 4060 8GB VRAM)
- **Execution Strategy:** A 3B 4-bit quantized model occupies ~2GB VRAM. However, running the fp16 baseline simultaneously for comparative tracing exceeds 8GB. 
- **Memory Mitigation:** We will employ **layer-by-layer offloading** and localized activation caching (via libraries like `TransformerLens` or `accelerate`). By executing the unquantized baseline iteratively and caching only the target layer’s intermediate activations to disk or system RAM, peak VRAM usage remains strictly under 6GB. SAE weights will also be loaded dynamically per layer.

### Target Venues
TMLR, NeurIPS/ICLR workshops (e.g., MECH Interp or Resource-Constrained Learning), Nature Machine Intelligence.

### PhD Application Impact
AI Safety and Interpretability are heavily prioritized by top-tier US labs (OpenAI, Anthropic, MIT). This project proves not just generic fine-tuning skills, but a deep, structural understanding of Transformer architectures and resource-constrained engineering.

### Overcoming the Bottlenecks (Mitigation Strategies)

- **The VRAM Squeeze:** *Challenge:* Storing complete activation states for a 3B fp16 baseline alongside a quantized variant for mechanistic comparison causes OOM. *Solution:* We will avoid full-graph gradient tracking. Instead, we use Attribution Patching (AtP) for rapid approximation and strictly implement layer-wise processing. The fp16 model will reside largely in system CPU memory, streaming in single layers to the GPU to compute the "clean" activation sets for caching, leaving VRAM free for the primary 4-bit analysis.
- **Polysemanticity Interference:** *Challenge:* Neurons are polysemantic. Fixing a neuron to restore French output risks destroying its overlapping role in Python syntax formatting. *Solution:* We bypass raw neuron patching by performing interventions in the **Sparse Autoencoder (SAE)** feature space. By patching monosemantic, decoupled features, we ensure that the "Activation Patch" only repairs the target language representation without catastrophic interference on programming logic.
- **Inference Overhead:** *Challenge:* Runtime interception and modification of vectors add latency, defeating the speed goals of 4-bit edge inference. *Solution:* We propose **Weight-Folding Patches**. Once the optimal activation steering vector is calculated, it can be folded offline into the model weights as a permanent bias shift for specific MLP layers, or deployed via an ultra-low-parameter LoRA matrix, entirely eliminating dynamic calculation overhead during edge deployment.
