# Section 2: Machine Learning, Deep Learning and Generative AI (Research + Application)

---

## 2.i) ROC Curve (Receiver Operating Characteristic Curve)

### Definition and Purpose

The Receiver Operating Characteristic (ROC) curve is a fundamental diagnostic tool in binary classification that visualizes the trade-off between a classifier's sensitivity and its tendency to produce false alarms across all possible decision thresholds. Originally developed during World War II for radar signal detection, the ROC curve has become one of the most widely used evaluation instruments in machine learning, medical diagnostics, and information retrieval (Fawcett, 2006).

The ROC curve plots the **True Positive Rate (TPR)** on the y-axis against the **False Positive Rate (FPR)** on the x-axis for every possible classification threshold. Formally, these quantities are defined as:

**True Positive Rate (Sensitivity / Recall):**

$$TPR = \frac{TP}{TP + FN}$$

where *TP* denotes true positives (correctly predicted positive instances) and *FN* denotes false negatives (positive instances incorrectly classified as negative). The TPR measures the proportion of actual positives that the classifier successfully identifies.

**False Positive Rate (1 - Specificity / Fall-out):**

$$FPR = \frac{FP}{FP + TN}$$

where *FP* denotes false positives (negative instances incorrectly classified as positive) and *TN* denotes true negatives (correctly classified negative instances). The FPR captures the proportion of actual negatives that the classifier erroneously labels as positive.

### Understanding the ROC Curve in Three Visual Steps

Drawing on the conceptual framework presented by Jadon (2018) in the reference article, the ROC curve can be understood through three progressive visual steps:

**Step 1 -- Threshold Variation and Its Effect on Classification.** Most classifiers produce a continuous probability score rather than a hard binary label. A decision threshold *t* is applied such that instances with predicted probability *p >= t* are classified as positive and those with *p < t* as negative. At the extreme threshold *t = 0*, every instance is classified as positive, yielding TPR = 1 and FPR = 1. At the opposite extreme *t = 1*, no instance is classified as positive, giving TPR = 0 and FPR = 0. As the threshold varies from 0 to 1, the classifier traces a path through the (FPR, TPR) space. A well-calibrated and discriminative model will achieve high TPR values at low FPR values for intermediate thresholds.

**Step 2 -- Plotting TPR vs. FPR at Each Threshold.** For each threshold value, the corresponding (FPR, TPR) pair is computed from the confusion matrix and plotted as a point. Connecting these points produces the ROC curve. A model with strong discriminatory power generates a curve that rises steeply toward the upper-left corner of the plot (FPR = 0, TPR = 1), indicating that it can correctly identify most positive instances without misclassifying many negatives. The diagonal line from (0, 0) to (1, 1) represents a classifier with no discriminative ability, equivalent to random guessing.

**Step 3 -- AUC (Area Under the Curve) Interpretation.** The Area Under the ROC Curve (AUC) provides a single scalar summary of the classifier's overall performance across all thresholds. Formally, AUC is computed as:

$$AUC = \int_0^1 TPR(FPR) \, d(FPR)$$

The AUC has a probabilistic interpretation: it equals the probability that the classifier assigns a higher predicted score to a randomly chosen positive instance than to a randomly chosen negative instance (Hanley & McNeil, 1982).

### AUC Interpretation Guidelines

- **AUC = 1.0 (Perfect classifier):** The model achieves perfect separation between positive and negative classes at some threshold. Every positive instance receives a higher score than every negative instance.
- **AUC = 0.5 (Random classifier):** The model has no discriminative ability whatsoever. The ROC curve coincides with the diagonal line, and the model's predictions are statistically indistinguishable from random coin flips.
- **AUC < 0.5 (Worse than random):** The model's predictions are systematically inverted -- it assigns higher scores to negative instances. In practice, this indicates a labeling error or a model that has learned the opposite of the intended pattern. Inverting the model's predictions would yield AUC > 0.5.

Practically, AUC values are often interpreted on the following scale: 0.9--1.0 (excellent), 0.8--0.9 (good), 0.7--0.8 (fair), 0.6--0.7 (poor), and 0.5--0.6 (fail).

### Application to Our LSTM Project

In our LSTM-based financial time series prediction project, we employed ROC curves to evaluate the directional prediction accuracy of four model architectures (Simple LSTM, Stacked LSTM, BiLSTM, and LSTM+Attention) on BTC/USD and XAU/USD datasets. The directional prediction task was framed as a binary classification problem: predicting whether the next-period price change would be positive (upward movement) or negative (downward movement). The sigmoid-transformed model outputs served as probability scores for computing ROC curves, and the AUC metric provided a threshold-independent measure of each model's ability to discriminate between upward and downward market movements. The generated ROC curve figures (`roc_curves_btcusd_1d.png` and `roc_curves_xauusd_1h.png`) allowed us to compare all four model variants on a single plot, with the AUC values displayed in the legend for immediate comparison.

---

## 2.ii) Transformer Architecture

### Overview

The Transformer architecture, introduced by Vaswani et al. (2017) in the seminal paper "Attention Is All You Need," represents a paradigm shift in sequence-to-sequence modeling. Unlike recurrent architectures (RNNs, LSTMs, GRUs) that process tokens sequentially, the Transformer relies entirely on attention mechanisms to capture dependencies between input and output positions, enabling full parallelization during training and superior handling of long-range dependencies.

The architecture follows an encoder-decoder structure, where the encoder maps an input sequence of symbol representations to a sequence of continuous latent representations, and the decoder generates an output sequence one element at a time in an autoregressive fashion.

### Encoder Block

The encoder consists of a stack of *N* = 6 identical layers, each containing two sub-layers:

1. **Multi-Head Self-Attention:** Each position in the input sequence attends to all positions in the same sequence, enabling the model to capture contextual relationships regardless of distance.
2. **Position-wise Feed-Forward Network:** A fully connected network applied independently to each position.

Each sub-layer is wrapped with a **residual connection** followed by **layer normalization**:

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

The information flow through a single encoder layer proceeds as follows: the input embeddings (augmented with positional encodings) enter the multi-head self-attention mechanism, which computes attention-weighted representations. The output is added back to the input via the residual connection and normalized. This result then passes through the feed-forward network, again with a residual connection and layer normalization.

### Decoder Block

The decoder also consists of *N* = 6 identical layers, but each layer contains three sub-layers:

1. **Masked Multi-Head Self-Attention:** Self-attention over the decoder's own previous outputs, with masking to prevent positions from attending to subsequent positions (ensuring the autoregressive property).
2. **Multi-Head Cross-Attention (Encoder-Decoder Attention):** The decoder attends to the encoder's output representations. Here, the queries come from the previous decoder sub-layer, while the keys and values come from the encoder output.
3. **Position-wise Feed-Forward Network:** Identical in structure to the encoder's feed-forward sub-layer.

Each sub-layer again employs residual connections and layer normalization. During generation, the decoder processes the output sequence autoregressively: at each step, it takes the previously generated symbols as additional input while attending to the full encoder output.

### Multi-Head Self-Attention Mechanism

The core innovation of the Transformer is the scaled dot-product attention. Given query matrix **Q**, key matrix **K**, and value matrix **V**, the attention function is computed as:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where *d_k* is the dimensionality of the key vectors. The scaling factor $\sqrt{d_k}$ prevents the dot products from growing excessively large for high-dimensional keys, which would push the softmax function into regions with extremely small gradients.

Rather than performing a single attention function, the Transformer employs **multi-head attention**, which runs *h* parallel attention operations (heads), each with learned linear projections:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

$$\text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Each head operates in a reduced-dimensional subspace ($d_k = d_v = d_{\text{model}} / h$), allowing different heads to capture different types of relationships (e.g., syntactic vs. semantic dependencies).

### Feed-Forward Networks

Each encoder and decoder layer contains a position-wise fully connected feed-forward network, applied identically to each position:

$$\text{FFN}(x) = \max(0, \, xW_1 + b_1)W_2 + b_2$$

This consists of two linear transformations with a ReLU activation in between. The inner dimension is typically larger than the model dimension (e.g., $d_{ff} = 2048$ vs. $d_{\text{model}} = 512$), creating a bottleneck architecture that allows the network to learn complex transformations.

### Residual Connections and Layer Normalization

**Residual connections** (He et al., 2016) add the sub-layer's input directly to its output, enabling gradient flow through deep networks and facilitating the training of the 6-layer stacks:

$$\text{output} = x + \text{Sublayer}(x)$$

**Layer normalization** (Ba et al., 2016) normalizes activations across the feature dimension (rather than across the batch dimension as in batch normalization):

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma + \epsilon} + \beta$$

where $\mu$ and $\sigma$ are the mean and standard deviation computed across features for each individual sample, and $\gamma$ and $\beta$ are learned affine parameters. Layer normalization stabilizes the hidden state dynamics and accelerates convergence.

### Positional Encoding

Since the Transformer contains no recurrence or convolution, it has no inherent notion of token order. **Positional encodings** are added to the input embeddings to inject sequence position information:

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

where *pos* is the position index and *i* is the dimension index. This sinusoidal formulation was chosen because it allows the model to extrapolate to sequence lengths longer than those seen during training, and because the relative position between any two tokens can be represented as a linear function of their encodings.

### Information Flow Summary

During a complete forward pass: (1) input tokens are converted to embeddings and summed with positional encodings; (2) the combined representation passes through the encoder stack, where each layer refines the representation through self-attention and feed-forward transformations; (3) the encoder output serves as the memory for the decoder; (4) the decoder processes previously generated tokens through masked self-attention, attends to the encoder memory via cross-attention, and applies a feed-forward transformation; (5) the final decoder output is projected through a linear layer and softmax to produce next-token probabilities.

---

## 2.iii) Ensemble Methods: Bagging, Boosting, and Stacking

### Bagging (Bootstrap Aggregating)

Bagging, introduced by Breiman (1996), reduces model variance by training multiple base learners independently on different bootstrap samples (random sampling with replacement) drawn from the training set, and then aggregating their predictions. For classification tasks, the aggregation is typically performed via majority voting; for regression, via averaging.

**Working Principle:** Given a training set of *n* samples, *B* bootstrap samples are drawn, each of size *n* (with replacement, so approximately 63.2% of unique samples appear in each bootstrap). An independent base learner is trained on each bootstrap sample. The final prediction is the aggregated output across all *B* base learners:

$$\hat{f}_{\text{bag}}(x) = \frac{1}{B} \sum_{b=1}^{B} \hat{f}_b(x)$$

The most prominent example is the **Random Forest** algorithm (Breiman, 2001), which combines bagging with random feature subsampling at each split node in a decision tree, further decorrelating the individual trees and improving generalization.

**Key Characteristic:** Bagging primarily reduces **variance** without substantially affecting bias. It is most effective with high-variance, low-bias base learners such as deep, unpruned decision trees.

### Boosting

Boosting methods train base learners **sequentially**, where each subsequent learner focuses on the mistakes made by its predecessors. The final prediction is a weighted combination of all learners' outputs.

**Working Principle:** In the general boosting framework, training instances that were misclassified by the current model receive higher weights, and the next model is trained on this re-weighted distribution. The final ensemble prediction is a weighted sum:

$$\hat{f}_{\text{boost}}(x) = \sum_{m=1}^{M} \alpha_m \hat{f}_m(x)$$

where $\alpha_m$ represents the weight assigned to the *m*-th learner, typically proportional to its accuracy.

**AdaBoost** (Freund & Schapire, 1997) adjusts instance weights after each iteration. **Gradient Boosting** (Friedman, 2001) fits each new model to the negative gradient (pseudo-residuals) of the loss function. **XGBoost** (Chen & Guestrin, 2016) extends gradient boosting with regularization, approximate split-finding algorithms, and system optimizations for scalability.

**Key Characteristic:** Boosting primarily reduces **bias** by iteratively correcting systematic errors. However, it can be prone to overfitting on noisy datasets if not properly regularized.

### Stacking (Stacked Generalization)

Stacking, proposed by Wolpert (1992), combines diverse base learners through a learned meta-model rather than through simple averaging or voting.

**Working Principle:** The process operates in two levels:
- **Level 0 (Base Learners):** Multiple diverse models (e.g., a decision tree, a logistic regression, a support vector machine, and a neural network) are trained on the original training data, typically using k-fold cross-validation to generate out-of-fold predictions that serve as features for the meta-learner.
- **Level 1 (Meta-Learner):** A second-level model is trained on the base learners' out-of-fold predictions, learning how to optimally combine their outputs. Common meta-learners include logistic regression or ridge regression, which are intentionally kept simple to avoid overfitting.

Stacking leverages the complementary strengths of diverse models, and the meta-learner can discover non-trivial combinations that neither voting nor simple averaging could achieve.

### Comparison Table

| Criterion | Bagging | Boosting | Stacking |
|---|---|---|---|
| **Training Strategy** | Parallel, independent on bootstrap samples | Sequential, each learner corrects predecessors | Two-level: base learners + meta-learner |
| **Error Reduction Focus** | Variance reduction | Bias reduction | Both (through diversity + learned combination) |
| **Parallelization** | Fully parallelizable | Inherently sequential | Base learners parallelizable; meta-learner sequential |
| **Overfitting Risk** | Low (averaging reduces variance) | Moderate to high (without regularization) | Moderate (requires careful cross-validation) |
| **Base Learner Requirement** | High-variance, low-bias models | Weak learners (high bias) | Diverse, heterogeneous models |
| **Sensitivity to Noise** | Robust | Can overfit to noise | Depends on meta-learner design |
| **Prominent Examples** | Random Forest, Bagged Trees | AdaBoost, Gradient Boosting, XGBoost, LightGBM | Blending, Super Learner |

---

## 2.iv) Ethical and Technical Challenges of Generative AI

### Hallucination

**Hallucination** refers to the phenomenon whereby generative AI systems produce outputs that are fluent, confident, and syntactically coherent but factually incorrect, fabricated, or entirely ungrounded in reality. This constitutes one of the most pressing technical challenges in modern AI deployment.

**Causes:** Hallucinations arise from several interacting factors: (1) statistical pattern completion -- language models predict the most probable next token rather than the most factually accurate one; (2) gaps in training data -- when the model lacks sufficient coverage of a topic, it interpolates plausibly but inaccurately; (3) compounding errors in autoregressive generation -- a single inaccurate token can cascade into an entirely fabricated narrative; (4) training objective misalignment -- the cross-entropy loss optimizes for token-level likelihood, not factual correctness.

**Recent Examples (2024--2025):** In 2024, multiple documented cases emerged of legal practitioners citing AI-generated case law that did not exist, prompting judicial sanctions. Google's Gemini model generated historically inaccurate images when prompted for depictions of historical figures, demonstrating multimodal hallucination. In the medical domain, LLM-based systems were found to fabricate clinical trial results when prompted with rare disease queries.

**Mitigation Strategies:** Retrieval-Augmented Generation (RAG) grounds model outputs in retrieved factual documents, significantly reducing hallucination rates. Grounding techniques connect the model to verified knowledge bases. Chain-of-thought prompting and self-consistency decoding enable the model to verify its own reasoning. Anthropic's Constitutional AI approach includes harm-avoidance principles that encourage the model to express uncertainty rather than fabricate.

### Bias

**Bias** in generative AI refers to systematic patterns in model outputs that reflect, amplify, or perpetuate societal prejudices present in training data.

**Manifestations:** Gender bias appears when models associate certain professions predominantly with one gender (e.g., "nurse" with women, "engineer" with men). Racial bias emerges in image generation systems that default to particular ethnic representations. Cultural bias manifests when models privilege Western-centric perspectives and knowledge. Socioeconomic bias arises when models generate content that disproportionately reflects affluent, English-speaking demographics.

**Causes:** Training datasets scraped from the internet inevitably contain the biases embedded in human-generated text. Underrepresentation of certain languages, cultures, and demographics in training corpora leads to systematically skewed outputs. Furthermore, RLHF (Reinforcement Learning from Human Feedback) can introduce annotator biases through the human preference data.

**Mitigation:** Building diverse and representative training datasets is a necessary first step. Red-teaming exercises -- where dedicated teams systematically probe models for biased behavior -- have become standard practice at organizations such as OpenAI and Anthropic. RLHF with diverse annotator pools helps calibrate model outputs. Constitutional AI (Bai et al., 2022) defines explicit principles that guide model behavior toward fairness. The EU AI Act (entered into force in August 2024, with phased implementation through 2025--2026) mandates bias auditing and transparency obligations for high-risk AI systems, classifying many generative AI applications as requiring conformity assessments.

### Data Privacy

**Data privacy** concerns in generative AI encompass the risks of personal data exposure through training data memorization, model inversion attacks, and the broader tension between model capability and privacy preservation.

**Memorization and Extraction Risks:** Large language models have been demonstrated to memorize verbatim sequences from their training data, including personally identifiable information (PII), email addresses, phone numbers, and proprietary code (Carlini et al., 2021). Adversarial prompting techniques can extract such memorized content, posing significant privacy risks.

**Model Inversion and Membership Inference:** Model inversion attacks attempt to reconstruct training data from model outputs. Membership inference attacks determine whether a specific data point was included in the training set. Both attack vectors compromise the privacy of individuals whose data was used during training.

**Regulatory Landscape:** The General Data Protection Regulation (GDPR) establishes the right to erasure ("right to be forgotten"), which presents a fundamental tension with the nature of neural network training -- removing an individual's data from a trained model is non-trivial and remains an active research area (machine unlearning). The EU AI Act introduces additional requirements for generative AI systems (categorized as "general-purpose AI"), including obligations to disclose training data summaries and comply with copyright law. In 2024, the Italian Data Protection Authority (Garante) temporarily banned ChatGPT over GDPR concerns, and several ongoing lawsuits challenge the legality of training AI models on copyrighted material.

**Technical Solutions:** Differential privacy injects calibrated noise during training to provide mathematical guarantees about the maximum information leakage about any individual training example. Federated learning enables model training on distributed data without centralizing sensitive information. Anthropic and OpenAI have both published safety practices emphasizing data minimization, access controls, and ongoing monitoring for memorization artifacts. Techniques such as knowledge distillation into smaller models and synthetic data generation are being explored as privacy-preserving alternatives to training on raw personal data.

---

## 2.v) Generative AI Product: LSTM Financial Prediction System via Claude Code

### Tool Used

The generative AI tool employed for this project is **Claude Code**, Anthropic's official command-line interface for Claude. Claude Code operates as an agentic coding assistant capable of reading, analyzing, and generating complex software systems through natural language commands. It leverages Anthropic's Claude large language model to understand software engineering requirements and produce production-quality code with appropriate architecture, documentation, and testing infrastructure.

### Commands and Prompts Given

The primary directive issued to Claude Code was:

> *"Build a production-level LSTM price prediction system with walk-forward validation, 4 model variants, quant-grade metrics, and automated PDF report generation."*

Subsequent iterative prompts refined the system:
- "Add Bidirectional LSTM and LSTM with Bahdanau Attention as additional model architectures"
- "Implement walk-forward cross-validation with expanding windows"
- "Generate ROC curves, confusion matrices, and equity curves for all model-dataset-parameter combinations"
- "Add statistical significance tests: ADF, Ljung-Box, Pesaran-Timmermann, and Diebold-Mariano"
- "Create an automated PDF report with all 141 figures and comprehensive metrics tables"

### Product Generated

Claude Code produced a complete, modular, production-grade LSTM financial prediction system with the following components:

**Four Deep Learning Model Architectures:**
1. **Simple LSTM** -- A baseline single-layer LSTM with fully connected output head.
2. **Stacked LSTM** -- A multi-layer LSTM with hierarchical feature extraction across 2+ layers with inter-layer dropout.
3. **Bidirectional LSTM (BiLSTM)** -- Processes the input sequence in both forward and reverse temporal directions, capturing both past and future context at each time step.
4. **LSTM + Attention** -- A 2-layer LSTM augmented with a Bahdanau (additive) attention mechanism that computes context-aware weighted summaries of hidden states across the entire lookback window.

**Automated Experiment Pipeline:** The system executed 32 experiments on GPU, encompassing all combinations of 4 model architectures, 2 datasets (BTC/USD daily and XAU/USD hourly), 2 lookback windows (30 and 60 periods), and 2 prediction horizons (1 and 5 periods). Each experiment followed a rigorous walk-forward cross-validation protocol with expanding training windows.

**141 Professional Visualization Figures:** The system automatically generated equity curves, prediction scatter plots, residual distributions, training loss curves, ROC curves, confusion matrices, attention heatmaps, model comparison charts, and lookback/horizon sensitivity analyses.

**Quant-Grade Financial Metrics:** Beyond standard regression and classification metrics (RMSE, MAE, R-squared, accuracy, precision, recall, F1), the system computed financial performance indicators including the Sharpe ratio, Sortino ratio, maximum drawdown, profit factor, hit ratio, and cumulative return for each directional trading strategy.

**Statistical Testing Suite:** Augmented Dickey-Fuller tests for stationarity, Ljung-Box tests for residual autocorrelation, Pesaran-Timmermann tests for directional accuracy significance, and Diebold-Mariano tests for pairwise model comparison were implemented and reported with p-values and interpretive text.

**Automated PDF Report Generation:** A complete academic-quality PDF report was generated programmatically using FPDF, incorporating all figures, metrics tables, statistical test results, and interpretive commentary.

### Demonstration of Generative AI Capabilities

This product exemplifies how generative AI can dramatically accelerate complex software engineering workflows. The entire system -- comprising approximately 2,000+ lines of modular Python code across data ingestion, feature engineering, model definition, training, evaluation, visualization, and report generation modules -- was produced through iterative natural language interaction with Claude Code. Tasks that would conventionally require days to weeks of manual development were completed in hours, with the AI assistant handling architectural decisions, boilerplate generation, debugging, and cross-module integration.

*[Note: Screenshots of the Claude Code interaction session, generated figures, and the final PDF report should be included in the submitted report as visual evidence of the generative AI workflow.]*

---

### References

- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization. *arXiv preprint arXiv:1607.06450*.
- Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI feedback. *arXiv preprint arXiv:2212.08073*.
- Breiman, L. (1996). Bagging predictors. *Machine Learning, 24*(2), 123--140.
- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5--32.
- Carlini, N., et al. (2021). Extracting training data from large language models. *30th USENIX Security Symposium*.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*.
- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters, 27*(8), 861--874.
- Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning. *Journal of Computer and System Sciences, 55*(1), 119--139.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics, 29*(5), 1189--1232.
- Hanley, J. A., & McNeil, B. J. (1982). The meaning and use of the area under a ROC curve. *Radiology, 143*(1), 29--36.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*.
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS 2017*.
- Wolpert, D. H. (1992). Stacked generalization. *Neural Networks, 5*(2), 241--259.
