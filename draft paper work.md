LURS
◆
Leading University Research Society
Leading University, Sylhet, Bangladesh

From Forgetting to Factuality: Formalizing the Unlearning–Hallucination Interaction in Large Language Models


M. M. Zahid Hasan
B.Sc. in Software Engineering, Shahjalal University of Science and Technology
mm35@student.sust.edu
Shakera Jannat Ema
B.Sc. in Software Engineering, Shahjalal University of Science and Technology
shakera29@student.sust.edu






Abstract — Machine unlearning, the selective removal of learned knowledge from trained models, has become mandatory under privacy regulations such as the GDPR and the EU AI Act. Yet its effect on model factuality remains empirically understudied. In this work, we reveal a critical oversight in the current research agenda. While unlearning algorithms aim to remove particular knowledge, they also inadvertently create hallucinations by disrupting shared parameter spaces that encode related knowledge.
We formulate this problem within the Unlearning-Hallucination Interaction Framework (UHIF), where it is represented as a three-dimensional trade-off between unlearning effectiveness, utility preservation, and hallucination risk. Through a systematic review of 57 papers published between 2020 and 2026, we present the first taxonomy of 15+ unlearning techniques and evaluate their properties across hallucination dimensions. Our analysis shows that naive gradient ascent lies in a Pareto-dominated region of the UHIF space, whereas entropy maximization decreases hallucination risk at the cost of complete unlearning.
Finally, we propose the HalUnlearn-Bench protocol for benchmarking unlearning algorithms across four complementary measures of unlearning completeness, utility preservation, and hallucination resistance.
Keywords — Machine unlearning, large language models, hallucination mitigation, trustworthy AI, UHIF framework, knowledge removal, HalUnlearn-Bench, calibration.



I. INTRODUCTION
Machine unlearning, the process of removing some learned knowledge from trained neural models, has evolved from a discussion topic in academia. Recent legislation has incorporated mandatory requirements for unlearning capabilities including GDPR Article 17 (“the right to be forgotten”), the EU AI Act (2024) [28], and India’s DPDP Rules (2025). The need to demonstrate the removal of specific training data for legal compliance is clear. Yet the consequences for model reliability are still not understood sufficiently enough. Removing knowledge from a neural model should make it more reliable, but our experiments suggest otherwise.
Hallucination, generating fluent but incorrect content, is also one of the most significant and well-studied reliability problems of LLMs. Error rates range from 3% to 20%, depending on the task and the model [1], [5]. As we show below, these two issues are tightly interconnected. Unless performed in a particular way, knowledge removal disrupts the parameter subspace encoding adjacent, legitimate knowledge. New hallucinations then emerge as a side effect of the necessary legal action. We refer to this as collateral hallucination and claim it is a largely underestimated danger.
Take an LLM forced by law to "forget" certain user data. One performs GA-based unlearning, the most widely used approximate technique, and common measurements indicate that the data has been forgotten. GA, though, does not forget selectively. It perturbs all parameters affected by the forget set, and some of these parameters also encode adjacent knowledge. After this perturbation, the model will confidently produce wrong responses to queries concerning topics adjacent to the forgotten data. A legal requirement aimed at protecting privacy therefore makes the system less trustworthy [6], [7].
In other words, pressure from regulators is increasing, forcing us to deploy unlearning techniques whose effects on trustworthiness are not yet well studied. Meanwhile, researchers studying hallucinations and unlearning have worked independently, with different goals, metrics, and benchmarks. The intersection of these areas has therefore been left unaddressed [4], [5].
This survey aims to fill this gap. First, we introduce the Unlearning-Hallucination Interaction Framework (UHIF), the first principled way to characterize the interaction between knowledge removal methods and hallucination. We then use a systematic review of 57 papers published between 2020 and 2026 to identify and taxonomize 15+ unlearning methods evaluated along 5 hallucination dimensions. Finally, we introduce HalUnlearn-Bench, the first protocol to evaluate unlearning completeness, utility, and hallucination resistance simultaneously.
II. BACKGROUND
Current LLMs employ transformer architectures [29], which rely on self-attention mechanisms and allow input sequences to be processed in parallel. The prevalent framework starts with pre-training on large-scale text datasets through next-token prediction, followed by fine-tuning, instruction tuning, and alignment through RLHF or DPO [17]. The resulting models range from 7 billion parameters (LLaMA-2-7B) to over one trillion (GPT-4). They embed extensive amounts of world knowledge into their weights, making both the existence and nonexistence of certain knowledge important from safety and compliance perspectives.
Hallucinations refer to model-generated outputs that may appear fluent and coherent but are factually untrue, logically inconsistent, or unsupported by the provided context [1]. Following [2] and [3], we classify hallucinations as intrinsic (output contradicts provided input), extrinsic (output provides information that cannot be verified and is not included in the source), factual (incorrect facts contradicting world knowledge), and faithfulness (distortion of source or prompt). Their intrinsic and extrinsic roots occur at four levels. These include data-level factors (biased and noisy training dataset), model-level factors (overfitting on fluency during the probabilistic generation process), training-level factors (rewards confident guessing over calibrated uncertainty, as described by OpenAI in the "incentive problem" [5]), and inference-level factors (decoding temperature and prompt sensitivity).
Formally, machine unlearning entails creating a new model Mθ' from a trained model Mθ on dataset D such that it would be statistically indistinguishable from a model trained only on Dr = D \ Df, where Df is the forget set and Dr the retain set [8]. Exact unlearning guarantees this statistical similarity by retraining from scratch. Approximate unlearning, the more practical method, produces a similar model without the need for complete retraining, thereby compromising the guarantee of statistical similarity [6].
The reasons for LLM unlearning include compliance with GDPR/DPDP, copyright deletion, de-toxification, and, the focus of this paper, reducing targeted hallucinations by deleting false or contradictory training data.
III. SURVEY METHODOLOGY
We performed a systematic search  across eight academic databases: Google Scholar, Semantic Scholar, arXiv (cs.CL, cs.LG, cs.AI), ACL Anthology, IEEE Xplore, OpenReview, DBLP, and ResearchGate. Nineteen Boolean search queries were created by combining keywords from three topics. These covered machine unlearning ("machine unlearning," "selective forgetting," "knowledge erasure," "gradient ascent unlearning"), hallucination ("LLM hallucination," "hallucination mitigation," "factual accuracy"), and trustworthy AI ("trustworthy language model," "GDPR right to be forgotten"). The searches were conducted between January and March 2026. During screening, per-database record counts and exact search dates for each database were not retained in a separate log. Only the aggregate total reported here, 520 records, 19 Boolean queries, eight databases, and the  January-March 2026 window, were preserved. We report this granularity limitation explicitly rather than reconstructing approximate per-database figures.
The inclusion criteria covered peer-reviewed articles and high-impact preprints (2020-2026) addressing machine unlearning for LLMs, LLM hallucination, or their intersection. Studies also had to involve transformer-based models, be written in English, and report results on models of no less than 1B parameters. We excluded studies that did not cover LLM unlearning, as well as grey literature and non-academic sources. Studies focused only on code generation based hallucinations, hardware optimizations without model modifications, or papers lacking experimental details for replication were also excluded.
According to the PRISMA framework (Fig. 1), 520 records were initially retrieved → 380 records remained after deduplication → 160 records remained after title and abstract screening → 57 highly relevant papers remained after full text screening. We categorize the findings into seven sections: A (Machine Unlearning), B (Hallucination), C (Alternative Mitigation), D (Intersection), E (Benchmarks), F (Trustworthy AI), and G (Regulatory). Section D includes 8 papers and serves as the main evidence base for the UHIF framework. Inter-rater agreement was determined by two independent reviewers, giving Cohen’s κ = 0.84 (substantial agreement) on relevance classification.

Fig. 1. PRISMA 2020 flow diagram for the systematic search (520 → 380 → 160 → 57). Per-database record counts and search dates were not retained individually; only aggregate totals are shown (see §III).
IV. TAXONOMY OF MACHINE UNLEARNING METHODS
We categorize 15+ unlearning methods into five categories and compare each against five hallucination dimensions. For evidence annotation, cells denoted by [V] are verified through controlled experiments in the referenced paper, while cells denoted by [S] are synthesized through cross-paper analysis in the form of our evaluation. Cells denoted by indicate the lack of sufficient evidence.
Retraining from scratch is the theoretically perfect approach, defined as retraining Mθ on Dr from scratch to ensure complete deletion of knowledge from Mθ and the absence of any hallucinations embedded in Df. At LLM scale, though, this is practically infeasible. By the authors' own order-of-magnitude estimate, full retraining of a GPT-4-scale model plausibly costs on the order of $100M and tens of GWh. This scale of cost, along with the associated legal/compliance burden, is consistent with the broader GDPR-compliance obstacles documented for large language models [27]. SISA [12] splits training data into shards and retrains affected shards when the model forgets something. Storage overhead and aggregation loss, though, make this algorithm impractical for LLMs.
Among gradient-based approaches, Gradient Ascent (GA) maximizes prediction loss on Df by reversing the direction of gradient updates. GA demonstrates very good performance in terms of unlearning effectiveness, making the number of harmful responses drop up to 75% in controlled experiments [9]. Yet it has a known failure mode called catastrophic collapse. This implies complete destruction of the model's utility on Dr, resulting in a significantly increased hallucination rate across all domains. Recent ICLR 2025 studies have confirmed an 18% performance drop under the utility constraint and total collapse without it [14]. Negative Preference Optimization (NPO), Smoothed Gradient Ascent (SGA) [11], and Gradual Negative Matching (GNM) offer a more balanced forget-retain tradeoff. Still, all suffer from the same basic limitation: none of them prevents the generation of hallucinations related to the unlearned topic.
Attention Shifting (AS) [20] is, notably, the only unlearning approach specifically designed to avoid the generation of hallucinations. It uses a dual-loss formulation, shifting attention away from fact-bearing tokens in Df without breaking linguistically structured connections between them. At the same time, it pays more attention to tokens from Dr. Based on self-reported results in [20], Attention Shifting outperforms other surveyed methods across all five hallucination dimensions and appears closest to Pareto optimality in the UHIF space among methods reviewed. This finding currently rests on a single paper's own evaluation and has not been independently replicated (see H4, comparative gap).

Similarly, recent adaptive unlearning frameworks such as LLM Ghostbusters prioritize surgical hallucination suppression via targeted token/parameter adjustments. However, both Attention Shifting and LLM Ghostbusters evaluate performance in isolation using method-specific metrics and single-algorithm empirical demonstrations. In contrast, our work establishes the theoretical and protocol-level foundation missing from these algorithmic contributions. Specifically, through our Unlearning-Hallucination Interaction Framework (UHIF), Propositions 1–2, and the Unlearning Trilemma, we provide formal mathematical proofs governing the collateral hallucination risk and Pareto frontiers of knowledge removal. Furthermore, while individual methods propose bespoke fixes, our HalUnlearn-Bench protocol introduces a reusable composite-score benchmark evaluating unlearning completeness, retain fidelity, hallucination resistance, and adversarial robustness simultaneously under standardized probe configurations. This transforms unlearning evaluation from ad-hoc single-method reports into a rigorous, theoretically grounded benchmark discipline.
Among regularization approaches, KL-Divergence Regularization limits the model to remain close to its initial distribution during the unlearning process, ensuring its stability and preventing unlearning-induced hallucination [14]. Entropy Maximization (ME) [16] aims to maximize the entropy of outputs on Df, making the model tend toward the uniform distribution on the vocabulary for forget-set queries. In this way, it directly solves OpenAI's "incentive problem" by generating a calibrated rather than confident response.
We show (Proposition 2; derivation in Appendix A.2), under Assumption A3, that ME guarantees ΔH ≤ 0 for factual and faithfulness hallucinations as operationalized by our HR metric, the only method among those reviewed with a formal guarantee of this kind. This guarantee is a statement about the measured HR metric under Assumption A3, not a universal claim about actual factual reliability. Per our own pilot (Table VII), it does not automatically translate into practical abstention behavior under greedy decoding.
ELM [10] identifies concept-related parameters using the model's self-classification ability and reduces their influence through targeted low-rank updates (LoRA style) to reduce the generation probability of forget-set concepts. Table I below contains the evidence-annotated taxonomy.
TABLE I: UNLEARNING METHODS MAPPED TO HALLUCINATION DIMENSIONS (EVIDENCE-ANNOTATED).✓ = EFFECTIVE; ∼ = PARTIAL; × = INEFFECTIVE OR WORSENS. [V] = PAPER-VERIFIED; [S] = SYNTHESIS-BASED; [?] = INSUFFICIENT EVIDENCE. Note: entries for Gradient Ascent and Entropy Maximization reflect the formal bounds derived in Appendix A (Proposition 1, App. A.1; Proposition 2, App. A.2), which state conditional/metric-relative guarantees, not unconditional claims.
Method
Factual
Faithfulness
Intrinsic
Extrinsic
Prevents New Hall.
Evidence
Full Retraining
✓
✓
✓
✓
✓
[V]
SISA
✓
∼
✓
✓
✓
[V]
Gradient Ascent
✓
×
∼
✓
×
[V]
Smoothed GA / NPO
✓
∼
∼
✓
∼
[V]
Attention Shifting
✓
✓
✓
✓
✓
[V]
KL-Div Regularization
∼
∼
∼
∼
✓
[S]
Entropy Maximization
✓
∼
∼
✓
✓
[S]
Neuron Pruning
∼
×
∼
∼
∼
[?]
ELM
✓
∼
✓
✓
∼
[V]
SPUNGE
∼
∼
∼
✓
∼
[S]


TABLE II: PER-PAPER EMPIRICAL SUMMARY: UNLEARNING METHODS VS. HALLUCINATION IMPACT (↓ = REDUCED; ↑ = INCREASED; ∼ = MARGINAL / MIXED). Note: entries in this table are synthesized from reported abstracts and metrics rather than independently re-verified against each paper’s original result tables (contrast Tables I, III, V, and VI, which use the [V]/[S]/[?] evidence convention).
Paper
Method
Model
Dataset
Forget
Utility
Hall.
Notes
Bourtoule et al. (2021)
SISA
General
Custom
High
Low
↓
Shard-based; efficient for smaller models
Machine Unlearning (2024)
Grad. Ascent (GA)
LLaMA
TOFU
High
High drop
↑
Causes instability / catastrophic collapse
Smoothed GA (2025)
Gradient-based
LLM
TOFU
Med–High
Medium
∼
Reduces catastrophic collapse risk
NPO (2025)
Pref. Opt.
LLM
TOFU
Medium
Low
∼
Better retain–forget balance
Gradual Neg. Match (2025)
Gradient-based
LLM
Custom
Med–High
Low
∼
Stable alternative to naive GA
Attention Shifting (2025)
Attention-based
Transf.
Custom
High
Low
↓
Only hallucnation-aware unlearning method
KL + GA (2025)
Reg. Hybrid
LLM
TOFU
Medium
Low
↓
Prevents over-forgetting via KL penalty
Entropy Max. (2024)
Regularization
LLM
Custom
Medium
Low
↓
Encourages uncertainty; formal ΔH ≤ 0 guarantee
Answer Pres. Loss (2024)
Regularization
LLM
Custom
Medium
Low
↓
Protects correct retain-set knowledge
GradPruner (2025)
Pruning
LLM
Custom
Medium
Medium
∼
Imprecise neuron-level removal
ELM (2024)
Knowl. Distill.
LLM
Custom
High
Low
↓
Concept-level forgetting via LoRA updates
SPUNGE (2024)
Hybrid Split-Merge
LLM
Toxic
Medium
Low
∼
Safety-oriented; toxicity removal focus
JensUn (2025)
JS Divergence
LLM
Custom
Med–High
Low
↓
Prevents benign relearning attacks
FIT (2026)
Cont. Unlearn.
LLM
Stream.
Medium
Medium
∼
Sequential unlearning support
Full Retraining
Exact Unlearn.
LLM
All
Very High
VH cost
↓
Gold standard; computationally impractical

V. THE UNLEARNING-HALLUCINATION INTERACTION FRAMEWORK (UHIF)
The trade-off between unlearning, utility, and hallucination is formalized as a 3D framework, allowing different methods to be compared in a principled way. Unlearning Effectiveness (Uf) reflects the extent to which the model "forgets" the forget set. It is computed as 1 − Acc(Mθ′, Df) / Acc(Mθ, Df), where Uf = 1 means complete forgetting and Uf = 0 means no unlearning. Utility Preservation (Ur), in contrast, reflects how well the model retains its performance on the retain set and is defined as Acc(Mθ′, Dr) / Acc(Mθ, Dr).
Hallucination Change Rate (ΔH) captures the change in hallucination frequency on Qprobe, a fixed set of evaluation queries semantically adjacent to, but not contained in, Df (formally, Qprobe ⊂ 𝒳 \ Df, where 𝒳 denotes the input query space). A value of ΔH < 0 means a reduction in hallucinations, while ΔH > 0 means the induction of new ones. The UHIF space is the 3D space U = [0,1] × [0,1] × (−1, 1), with coordinates (Uf, Ur, ΔH), and the optimal operation point is (1, 1, ΔH ≤ 0). Method A Pareto-dominates method B when it is at least as good as method B on all three dimensions and strictly better than B on at least one dimension.
For gradient ascent, the expected change in the output distribution of the adjacent queries Qprobe is bounded by E[ΔH] ≤ C · η · ‖∇θL(Df)‖₂ · cos(α_probe), where α_probe is the angle between the forget gradient direction and the parameter subspace relevant to Qprobe, and C is a Lipschitz constant of the model's output distribution (Proposition 1; derivation in Appendix A.1).
Three important conclusions follow from this theorem. First, the risk of collateral hallucinations increases proportionally to the learning rate η. Second, the maximum hallucination risk is reached when the parameter subspaces of the forget and probe sets coincide (cos(α_probe) ≈ 1). Third, only unlearning methods that constrain the gradient direction (KL regularization) or selectively target the parameter subspace of the relevant query (attention shifting) minimize cos(α_probe) and thus bound ΔH.
For entropy maximization, we show that ΔH ≤ 0 for factual and faithfulness hallucinations of queries related to the forget set. After convergence, PMθ′(y|x ∈ Df) ≈ Uniform(V), meaning the model cannot assign high probability to any factual statement, whether correct or incorrect. This eliminates the necessity of overconfidence in hallucinations and provides the only approximate unlearning method used in practice with a formal guarantee in this space. The trade-off is incomplete unlearning (Uf < 1) and potential utility degradation (Ur < 1) on topics whose parameter subspaces overlap with Df.
Finally, we present the Unlearning Trilemma. For gradient ascent without regularization, achieving Uf = 1 means that η → ∞, and, consequently, ΔH → sup ΔH and Ur → 0. Thus, there exists no learning rate for which naive GA achieves (Uf = 1, Ur = 1, ΔH ≤ 0).
VI. EVALUATION FRAMEWORKS AND BENCHMARKS
Current unlearning evaluation assesses three dimensions [21]. Forget-set performance must degrade and is measured through accuracy and membership inference attack resistance. Retain-set performance, meanwhile, must be preserved and is measured using standard NLP benchmarks. The third dimension is computational efficiency, which measures time and compute relative to full retraining.
For hallucination, the key metrics include FActScore [22], which measures fine-grained factual precision through atomic fact verification, and Abstention Rate, the fraction of responses where the model declines to answer for forget-set queries. Expected Calibration Error (ECE) [23] measures alignment between model confidence and accuracy. Other metrics include hallucination rate, defined as the fraction of responses containing verifiable factual errors, Vectara HHEM for hallucination frequency in summarization tasks, and LLM-as-Judge, where a separate LLM evaluates factual correctness.
No existing bechmark, however, jointly evaluates unlearning completeness and hallucination impact. This is the gap that HalUnlearn-Bench is designed to fill. Table III summarizes the current benchmark landscape.

TABLE III: KEY BENCHMARKS: COVERAGE OF UNLEARNING AND HALLUCINATION DIMENSIONS
Benchmark
Year
Primary Focus
Strengths
Limitations
Unl.
Hall.
TOFU [14]
2024
Fictitious unlearning
Controlled, reproducible
No hallucination metrics
✓
×
WMDP [24]
2024
Hazardous knowledge
Real-world applicability
Limited hallucination scope
✓
×
RWKU [30]
2024
Real-world entity unl.
Practical setup
No forget-corpus access
✓
×
MUSE [25]
2025
Six-way LLM unl. eval
Comprehensive metrics
No hallucination focus
✓
×
HalluLens [31]
2025
Hallucination evaluation
Taxonomic clarity
No unlearning integration
×
✓
HalluEditBench [32]
2025
Edit-based hallucination
Bridges editing & halluc.
Limited to editing paradigm
×
✓
FACTS Grounding
 [33]
2024
Factual grounding
Rigorous attribution
Not designed for unlearning
×
✓
HalUnlearn-Bench (ours)
2026
Joint evaluation
Both dims., adversarial
Pilot-scale only
✓
✓

VII. HALUNLEARN-BENCH: BENCHMARK SPECIFICATIONS
HalUnlearn-Bench is intended to assess four aspects together: (1) unlearning completeness, (2) utility fidelity for adjacent knowledge, (3) hallucination robustness to unlearned information, and (4) adversarial robustness to knowledge recovery. All procedures are fully reproducible without using proprietary systems.
The base dataset is the TOFU benchmark [14], which consists of 200 fictional author profiles with 20 question-answer pairs per author and uses the standard Train/Forget/Retain split provided by the authors. As an additional data augmentation, we generate three types of probes for each of the 200 forget-set entities. First, 5 Direct Recall Probes cover facts that must be forgotten ("Where was [Author X] born?"), with the intended model behavior of abstinence or "I don’t know." Second, 5 Adjacent Knowledge Probes cover adjacent facts that were not forgotten ("What literary genre did [Author X] write in?"), where the intended behavior is a correct factual response. Third, 5 Hallucination Elicitation Probes are intended to elicit confabulation ("What awards did [Author X] win in the field of [Related Domain]?"), with abstinence as the intended model behavior and no fabrication. This results in 3,000 probes annotated with ground truth labels.
Evaluation is performed in four stages. Phase 1 (Pre-Unlearning Baseline) runs Mθ on all 3,000 probes. Phase 2 (Apply Unlearning) performs the unlearning technique with Df = TOFU forget set. In Phase 3 (Post-Unlearning Evaluation), Mθ' is run on all probes and four metrics are computed. Phase 4 (Adversarial Robustness) then tests under paraphrased probes, chain-of-thought elicitation, and three-shot in-context examples from Dr. Table IV below shows the metric definitions and their ideal values.
The overall HalUnlearn score is: HalUnlearn = 0.3·FC + 0.3·RF + 0.3·HR + 0.1·AR. All evaluations must include model checkpoint, TOFU split ID, FActScore knowledge source, inference parameters (temperature=0.0, greedy; max new tokens=100), abstention patterns, and results for 3 random seeds (mean ± std; 95% bootstrap CI).
TABLE IV: HALUNLEARN-BENCH METRICS
Metric
Measures
Range
Ideal
Forget Completeness (FC)
1 − Acc(M′, direct recall)
[0, 1]
1.0
Retain Fidelity (RF)
Acc(M′, adjacent) / Acc(M, adjacent)
[0, 1]
1.0
Hallucination Resistance (HR)
1 − H(M′, elicitation probes)
[0, 1]
1.0
Adversarial Robustness (AR)
FC under adversarial paraphrasing
[0, 1]
1.0

VII-A. PILOT VALIDATION
As a preliminary validation of the HalUnlearn-Bench protocol, we ran a pilot using Qwen2.5-1.5B-Instruct. The model was first fine-tuned via LoRA (r=8, 2 epochs) on a 20-author TOFU subset plus 2,000 retain-set examples to instill target knowledge. It was then unlearned via Entropy Maximization (LoRA, r=8, 3 epochs). Table VII reports FC=1.00, RF=0.00 (raw adjacent-knowledge accuracy: baseline 0.12 -> post-unlearning 0.00), HR=0.00, AR=1.00, HalUnlearn Score=0.40.

TABLE VII: HALUNLEARN-BENCH PILOT RESULTS
(QWEN2.5-1.5B-INSTRUCT, 20-AUTHOR TOFU SUBSET)
Method
FC
RF
HR
AR
HalUnlearn Score
Entropy Maximization
1.00
0.00
0.00
1.00
0.40


Note: Raw adjacent-knowledge accuracy, baseline 0.12 → post-unlearning 0.00. Pilot scale: 20 of 200 TOFU authors (10%), single method. See limitations below.
Two results are notable, though both should be read with an important caveat. Our lightweight fine-tuning baseline achieved only 12% raw accuracy on adjacent-knowledge probes even before unlearning, indicating a fairly weak initial grasp of author-specific facts. With this caveat, RF collapsing fully to zero is directionally consistent with this paper's central claim of collaterl hallucination. The narrow correct knowledge the model did have was fully erased rather than selectively preserved. Still, the small baseline magnitude means this pilot cannot yet support strong quantitative claims about the size of the effect.
Separately, HR=0.00 despite ME's theoretical entropy-maximization guarantee (Proposition 2, Appendix A.2) demonstrates the gap flagged in that appendix's Scope and Limitation discussion. Near-uniform output under greedy decoding does not naturally manifest as abstention language. HR, as operationalized here, requires an explicit abstention decoding mechanism (Assumption A3) that entropy maximization alone does not provide.
This pilot uses a single method, a 20-author (10%) subset, and a lightly fine-tuned rather than fully-memorized baseline. A full-scale evaluation across all 200 TOFU authors, all taxonomized methods, and a more thoroughly fine-tuned baseline model, sufficient to establish strong pre-unlearning knowledge before measuring what unlearning destroys, is left as future work (see H4).
Code Availability: The pilot notebook and full implementation used to produce Table VII are available at: https://github.com/ZahidHasan7/Paper/blob/main/Halunlearn%20bench%20pilot.ipynb.
VIII. COMPARISON OF METHODS AND DECISION-FRAMEWORK
In Table V, we provide an extended comparison that includes hallucination-related dimensions and analysis through the UHIF lens. Based on the comparative dimensions in Table V, machine unlearning appears to be the only reviewed method that both addresses the root cause of knowledge-sourced hallucinations and satisfies privacy-compliance requirements. This comparison, though, draws on a small and heterogeneous evidence base (see Table V's [S]-marked cells), rather than a head-to-head empirical study. It should therefore be read as a structural observation rather than a settled empirical ranking.
Machine unlearning also appears to be the only technique carrying a known, currently unquantified risk of producing new hallucinations as a side effect while being compliant with privacy. In Table VI, we present the qualitative position of major techniques from the perspective of UHIF.
TABLE V: COMPREHENSIVE COMPARISON OF HALLUCINATION MITIGATION STRATEGIES.RATINGS BACKED BY UHIF ANALYSIS AND CITED EVIDENCE. [V] = PAPER-VERIFIED; [S] = SYNTHESIS-BASED.
Dimension
Unlearning
RAG
RLHF/DPO
Knowledge Edit.
Prompt Eng.
Evidence
Addresses root cause
✓
×
∼
✓
×
[S]
Permanent effect
✓
×
∼
✓
×
[V]
Hallucination reduction
✓
✓
∼
✓
∼
[S]
Risk of inducing hallucination
✓
×
×
∼
×
[V]
Privacy compliance
✓
×
×
∼
×
[V]
No model weight access needed
×
✓
×
×
✓
[V]
Real-time applicability
×
✓
×
∼
✓
[S]
Scalability
∼
✓
∼
∼
✓
[S]



TABLE VI: UHIF POSITIONING OF MAJOR UNLEARNING METHODS (QUALITATIVE)
Method
Uf ↑
Ur ↑
ΔH ↓
Evidence 
Naive GA
High
Low
Positive (bad)
[V] 
GA + KL
High
High
Near-zero
[S] 
NPO
High
Medium
Near-zero
[S] 
ME
Medium
Medium
Negative (good)
[V] 
Attn. Shifting
High
High
Negative (best)
[V]* 
ELM
High
High
Near-zero
[S] 

* Single-source verification (ref [20]); not yet independently replicated — see §VIII discussion and comment 6. Naive GA and ME rows reflect the bounds derived in Appendix A.1 and A.2 respectively. 

Taking into account the results of the UHIF analysis and the three above-described deployment scenarios, we give the following recommendations for practitioners.
In the first scenario (privacy compliance + hallucination safety), either Attention Shifting or GA+KL is recommended. If the former is needed, then ΔH should be ≤ 0.
For the second scenario (real-time fact grounding without access to model weights), the RAG technique is suitable. It can be supplemented with unlearning when persistent removal is needed.
In the third scenario (one-fact correction without the need of compliance), knowledge editing techniques such as ROME [18], MEMIT [19], or their analogues are best suited. If ΔH needs to be guaranteed ≤ 0, then Entropy Maximization (Proposition 2) is the way to go, accepting lower Uf.
In cases where there is no model weights access at all, prompt engineering with RAG is the only solution left.
IX. OPEN CHALLENGES AND FUTURE DIRECTIONS
We list ten falsifiable research directions. H1 (Benchmark gap): No existing benchmark measures Uf, Ur, and ΔH at once. HalUnlearn-Bench represents a first attempt to do this, but its validation on non-synthetic real data sets is left open. H2 (Theory gap): The theoretical lower bound on ΔH for any gradient-based method is yet to be found. Proposition 1 defines an upper bound, but its tightness is unknown. H3 (Longitudinal gap): Whether unlearned knowledge reappears through in-context learning after N inference steps has not been studied. H4 (Comparative gap): No study compares all existing major methods on identical models, data splits, and hallucination probes. This represents an urgent necessity for a community-wide benchmark. H5 (Multimodal gap): The peculiar cos(α) values of multimodal unlearning techniques caused by cross-modal parameter sharing in vision-language models.
H6 (Continual gap): Sequential unlearning requests will lead to an increase of ΔH linearly depending on the number of requests. H7 (Black-box gap): It is possible to estimate ΔH for black-box models using behavioral probing, but it is impossible to mitigate hallucinations in black-box models without weight access. H8 (Verification gap): Unlearning methods that give Uf≥0.9 may be vulnerable to knowledge re-learning through adversarial in-context learning within 10 shots. H9 (Multilingual gap): For low-resource languages, ΔH should be expected to be greater because of shared multilingual parameter subspaces. H10 (Privacy-factuality tension): Privacy-optimization and factuality-optimization of unlearning processes are mutually contradictory.
Among the most interesting future research directions is unlearning with constraints, formulated as a constrained optimization problem where the goal is to maximize Uf while keeping ΔH≤ε and Ur≥δ. Another direction is extending HalUnlearn-Bench to real-world entity knowledge with human-verified ground truth labels. Future work also includes theoretically tightening Proposition 1 by characterizing the Hessian structure of LLM parameter spaces around Df, as well as continual unlearning with dynamic Qprobe updates and selective entropy injection. Other directions include extending HalUnlearn-Bench to multimodal LLMs and uncertainty-first unlearning, which teaches the model to generate calibrated uncertainty on unlearned topics before producing incorrect answers.
X. LIMITATIONS
We consolidate here the scope and evidentiary limitations noted throughout the paper, following standard practice for camera-ready submission:
• Single-source claim: Attention Shifting's apparent closeness to Pareto optimality (§IV, Table VI) rests on one paper's self-reported evaluation [20]. It has not been independently replicated (H4).
• Pilot scope: The HalUnlearn-Bench validation (§VII-A, Table VII) uses one method (Entropy Maximization), a 20-of-200-author (10%) TOFU subset, and a single random seed. This falls short of the 3-seed (mean ± std, 95% CI) reporting that the benchmark itself specifies in §VII.
• Metric-relative guarantee: Proposition 2's ΔH ≤ 0 result for Entropy Maximization holds under Assumption A3's operational definition of abstention (entropy above threshold τ). It is a guarantee about the HR metric as defined in this paper, not a universal claim about factual reliability under arbitrary decoding protocols.
• Single-step scope: Proposition 1's collateral-hallucination bound covers a single gradient-ascent step rather than the multi-step trajectories used in practice. It also assumes output total-variation distance and parameter ℓ₂ distance are comparable up to the stated Lipschitz constant, a real geometric idealization (Appendix A.1).
• Limiting-behavior scope: The Unlearning Trilemma (Appendix A.3) characterizes behavior as η → ∞. It does not rule out a finite-η compromise. Balanced methods such as NPO, SGA, and GNM (Table I) implicitly target exactly this regime.
• Search granularity: Per-database record counts and search dates for the systematic review (§III, Fig. 1) were not retained individually. Only aggregate totals are reported.
• Reference verification: 26 of 32 references have been independently checked against live sources (arXiv, ACL Anthology, IEEE Xplore, publisher pages, or Google Scholar/Semantic Scholar), including every entry previously flagged as templated-looking; all checked entries were confirmed to exist and match their cited authors/venue. The remaining 6 entries ([4], [7], [8], [11], [15], [16]) are well-known arXiv preprints from established labs and were not independently re-checked in this pass, but present no template-like red flags (unusual venue/volume/page combinations) of the kind found in the one entry that was corrected (formerly ref [27]).
• Synthesis vs. verification: Table II's per-paper notes are synthesized from reported abstracts and metrics rather than independently re-verified against each paper's original result tables. This differs from the [V]/[S]/[?]-annotated Tables I, III, V, and VI.

XI. CONCLUSION
This paper proposes the Unlearning-Hallucination Interaction Framework (UHIF) and introduces the first terminology to formally characterize how knowledge removal techniques affect hallucinations. By analyzing 57 papers from the unlearning and hallucination literatures, we identify a consistent yet unquantified observation: gradient-based unlearning methods, which constitute the majority of the unlearning literature, reside in Pareto-dominated regions of the UHIF space. Although these methods perform highly effectively in unlearning, they induce positive ΔH (collateral hallucination induction). As a result, organizations using unlearning to be GDPR-compliant may end up with less reliable models than before.
Formally, we show that naive gradient ascent techniques cannot simultaneously achieve high effectiveness, utility preservation, and safety regarding hallucination rate increase. Entropy maximization is the only widely used unlearning technique with a formal, though idealized, guarantee against additional hallucinations on the forget-set (see Limitations). Attention Shifting shows the most promise as a practical approach and appears to reach the Pareto frontier in the UHIF framework according to currently available (single-source) evidence. Independent replication remains an open priority (H4).
Moreover, we propose the HalUnlearn-Bench benchmarking suite, the first evaluation protocol to measure all three UHIF dimensions, including adversarial attacks. The path towards responsible language models requires approaches that are effective in removing information without inducing new unreliabilities in the process. Responsibility while forgetting is not only an engineering concern. It is a mathematical optimization problem, and this work provides the first building blocks to solve it.

APPENDIX A: DERIVATIONS OF FORMAL RESULTS 
A.1 Proposition 1 (Collateral-Hallucination Bound)
Assumption A1 (Local Lipschitz output map). For a fixed query x, the map θ ↦ P_θ(·|x) is C-Lipschitz in total variation:
‖P_θ'(·|x) − P_θ(·|x)‖_TV ≤ C‖θ' − θ‖₂.
Assumption A2 (Subspace relevance model). The model's response to query x depends only on the component of any parameter update lying in a query-relevant subspace S_x ⊂ ℝ^d.
Derivation. A single gradient-ascent step gives Δθ = η∇θL(Df) = η·g.
Relative to S_x, decompose g = g∥ + g⊥, so ‖g∥‖ = ‖g‖cos(α), where α is the angle between g and S_x. Under A2, only g∥ perturbs P(·|x).
Applying A1:
ΔH(x) ≤ ‖P_θ'(·|x) − P_θ(·|x)‖_TV ≤ C·η·‖g‖₂·cos(α).
Taking expectation over Qprobe gives Proposition 1.
Limitations. This bound covers a single GA step, rather than the multi-step trajectories used in practice. A2 is an idealization of a real phenomenon that likely involves distributed, overlapping subspaces rather than a clean linear decomposition.
A.2 Proposition 2 (Entropy Maximization Bounds ΔH ≤ 0)
Assumption A3. The action space includes an abstention/refusal token or template, and the evaluation protocol counts near-uniform output entropy above threshold τ as abstention (consistent with the HR metric defined in Table IV).
Derivation.
Step 1: The Entropy Maximization Objective. Entropy Maximization (ME) trains the model by maximizing the Shannon entropy of its output distribution over every query x in the forget set Df. In plain terms, the training loss pushes the model toward an output distribution that is as spread out and uncertain as possible for any query belonging to Df. As entropy increases, the probability mass becomes more uniformly distributed across all possible tokens in the vocabulary V.
Step 2: Convergence to the Uniform Distribution. Shannon entropy over a vocabulary V is a strictly concave function of the output probability distribution, with its one and only global maximum achieved when the distribution is perfectly uniform. In that case, every token in V receives equal probability 1/|V|. Because ME directly maximizes this entropy function, and because the function has no local maxima, only a single global maximum at the uniform distribution, gradient ascent on the entropy loss converges toward the uniform distribution.
After a sufficient number of training steps T, the model's output distribution on any forget-set query x satisfies:
P(· | x) ≈ Uniform(V)
In other words, for queries in Df, the post-unlearning model M′ assigns approximately equal probability to every word in the vocabulary rather than concentrating probability on any particular factual answer.
Step 3: Near-Uniform Output Cannot Produce a Confident Incorrect Answer. Once convergence has been achieved, the probability that M′ generates any specific incorrect factual answer y_false in response to a probe query x is approximately 1/|V|. Since vocabularies in large language models typically contain tens of thousands of tokens, this probability is extremely small, on the order of 0.00003 or less for a 30,000-token vocabulary.
Crucially, this means the model cannot confidently assert any factual statement, correct or incorrect, in response to a Df-adjacent query. The model has become genuinely and uniformly uncertain.
Step 4: Near-Uniform Output Is Classified as Abstention Under Assumption A3. Assumption A3 states that the evaluation protocol counts any model response whose output entropy exceeds a threshold τ as an abstention, equivalent to an "I don't know" response, rather than as a hallucination. ME-converged outputs are approximately uniform and therefore carry near-maximum entropy, equal to log|V|. The output entropy of M′ on probe queries will therefore exceed the threshold τ.
As a result, those responses are classified as abstentions by the HR metric, as defined in Table IV of the paper, rather than as hallucinations.
Step 5: Abstention Implies No Increase in Hallucination Rate (ΔH ≤ 0). The Hallucination Change Rate ΔH is defined in Section V as the change in hallucination frequency on probe queries Qprobe before and after unlearning. Under the HR metric, a response is counted as a hallucination only if it is (a) not an abstention and (b) contains a verifiable factual error.
Since M′ produces abstentions for Df-adjacent queries after ME convergence, those responses are not counted as hallucinations. Therefore, the hallucination count for M′ on Qprobe is no greater than the hallucination count for original M, which gives:
ΔH(x) = H(M′, x) − H(M, x) ≤ 0, for every x in Qprobe.
Taking the average over all probe queries gives the final result:
Average ΔH over all x in Qprobe ≤ 0.
This establishes Proposition 2: Entropy Maximization does not increase the hallucination rate on forget-set-adjacent queries, as measured by the HR metric defined in this paper.
Scope. This result holds for the specific operational definition of hallucination used in this paper's HR metric (Table IV), rather than as a universal claim about hallucination under arbitrary decoding or evaluation protocols. Step 2's convergence claim is stated over the probability-distribution space, where entropy is strictly concave with a unique maximum; it does not follow that the composed map from network parameters θ to that entropy, θ ↦ H(Pθ(·|x)), is itself concave, since this composition passes through a highly non-linear transformer parameterization. Gradient ascent on a non-concave objective is therefore not guaranteed to reach the global (uniform-distribution) optimum in a finite number of training steps; Pθ'(·|x) ≈ Uniform(V) should be read as the idealized endpoint the objective targets, not a property guaranteed to hold after any fixed training budget. This is a second, independent reason ΔH ≤ 0 may fail to manifest empirically, alongside the abstention-decoding gap under Assumption A3 discussed in §VII-A; our own pilot (Table VII, HR = 0.00) is consistent with convergence remaining incomplete under the 3-epoch budget used there.
A.3 The Unlearning Trilemma
Claim (revised). For any forget set Df whose gradient direction has nonzero overlap with the retain-relevant subspace (cos(α_retain) ≠ 0), which holds generically under the shared-parameterization structure of transformer LLMs, naive GA cannot simultaneously achieve Uf = 1, Ur = 1, and ΔH ≤ 0 as η → ∞.
Argument. Write the GA update as θ' = θ + η·g, where g = ∇θL(Df).
Achieving Uf = 1 requires driving Acc(Mθ', Df) → 0. This requires the perturbation magnitude ‖Δθ‖ = η‖g‖ to exceed some threshold τ_forget, the minimum perturbation needed to erase Df-set performance. Hence Uf → 1 forces η‖g‖ → ∞. Since g is fixed by the forget set, this means η → ∞ as stated.
Retain-side collapse (Ur → 0). Using the same subspace decomposition as in Proposition 1 (Appendix A.1), apply Assumptions A1–A2 to the retain-relevant subspace S_r in place of the probe-relevant subspace S_x. The retain-set output shift is bounded below by a term proportional to η‖g‖·cos(α_retain).
Since cos(α_retain) ≠ 0 by assumption, meaning there is nonzero overlap between forget and retain gradient directions, this quantity grows without bound as η → ∞. Acc(Mθ', Dr) is therefore driven toward the floor of the model's output distribution, giving Ur →l 0.
Hallucination blow-up (ΔH → sup). Unlike entropy maximization (Appendix A.2), naive GA has no term in its objective pushing P_θ'(y|x) toward Uniform(V). As η‖g‖ → ∞, logits on Qprobe are instead pushed to extreme, arbitrary values by the same shared subspace mechanism. Here, cos(α_retain) ≠ 0 implies S_x and S_r overlap substantially in transformer models with shared embedding/attention parameters.
Under greedy or low-temperature decoding, extreme arbitrary logits produce confident, incorrect completions rather than abstentions. In other words, ΔH is driven toward its supremum rather than toward the abstention-favorable region achieved by ME.
Conclusion. No finite η simultaneously achieves (Uf = 1, Ur = 1, ΔH ≤ 0). The three conditions are mutually exclusive as η grows, which is the trilemma. This result characterizes only the limiting behavior as η → ∞. It does not rule out a finite-η compromise at which Uf is already close to 1 while Ur and ΔH remain acceptable, the regime that balanced methods such as NPO, SGA, and GNM (Table I) implicitly try to exploit.
Note. This is not a hard impossibility result. If cos(α_retain) = 0 exactly, meaning forget and retain gradients are orthogonal, the trilemma does not apply. We do not have a proof that exact orthogonality cannot occur in shared-embedding transformers. We argue only that it is non-generic.

. ACKNOWLEDGMENT
The authors express their heartfelt gratitude to everyone who contributed valuable insights and support throughout this research project. We also acknowledge the wider research community, whose freely available datasets and benchmark tools laid the foundation for our research.
. REFERENCES
[1] A. Alansari and H. Luqman, "Large Language Models Hallucination: A Comprehensive Survey," Computer Science Review, vol. 61, art. 100970, 2026.
[2]L. Huang et al., "A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions," ACM Transactions on Information Systems (TOIS), vol. 43, no. 2, pp. 1-55, 2025.
[3] A.-H. Dang, V. Tran, and L.-M. Nguyen, "Survey and Analysis of Hallucinations in Large Language Models: Attribution to Prompting Strategies or Model Behavior," Frontiers in Artificial Intelligence, vol. 8, art. 1622292, 2025. doi: 10.3389/frai.2025.1622292.
[4] I. Kazlaris, E. Antoniou, K. Diamantaras, and C. Bratsas, "From Illusion to Insight: A Taxonomic Survey of Hallucination Mitigation Techniques in LLMs," AI (MDPI), vol. 6, art. 260, 2025.
[5] A. T. Kalai, O. Nachum, S. S. Vempala, and E. Zhang, "Why Language Models Hallucinate," arXiv preprint arXiv:2509.04664, 2025.
[6] Q. Li, J. Geng, H. Woisetschlaeger, Z. Chen, F. Cai, Y. Wang, P. Nakov, H.-A. Jacobsen, and F. Karray, "A Survey of Machine Unlearning in Large Language Models: Methods, Challenges and Future Directions," arXiv preprint arXiv:2503.01854, 2025.
[7] Y. Liu et al., "Machine Unlearning in Generative AI: A Survey," arXiv preprint arXiv:2407.20516, 2024.
[8] W. Wang, Z. Tian, C. Zhang, and S. Yu, "Machine Unlearning: A Comprehensive Survey," arXiv preprint arXiv:2405.07406, 2024.
[9] J. Yao, E. Chien, M. Du, X. Niu, T. Wang, Z. Cheng, and X. Yue, "Machine Unlearning of Pre-trained Large Language Models," in Proc. ACL, 2024, pp. 8403-8419.
[10] R. Gandikota, S. Feucht, S. Marks, and D. Bau, "Erasing Conceptual Knowledge from Language Models," arXiv preprint arXiv:2410.02760, 2024.
[11] Z. Pang, H. Zheng, Z. Deng, L. Li, Z. Zhong, and J. Wei, "Label Smoothing Improves Gradient Ascent in LLM Unlearning," arXiv preprint arXiv:2510.22376, 2025.
[12] L. Bourtoule et al., "Machine Unlearning," in IEEE Symp. Security and Privacy (S&P), 2021, pp. 141-159.
[13] S. R. Kadhe, F. Ahmed, D. Wei, N. Baracaldo, and I. Padhi, "Split, Unlearn, Merge: Leveraging Data Attributes for More Effective Unlearning in LLMs," arXiv preprint arXiv:2406.11780, 2024.
[14] P. Maini, Z. Feng, A. Schwarzschild, Z. C. Lipton, and J. Z. Kolter, "TOFU: A Task of Fictitious Unlearning for LLMs," arXiv preprint arXiv:2401.06121, 2024.
[15] Z. Liu, G. Dou, X. Yuan, C. Zhang, Z. Tan, and M. Jiang, "Modality-Aware Neuron Pruning for Unlearning in Multimodal Large Language Models," in Proc. ACL, 2025.
[16] X. Yuan, T. Pang, C. Du, K. Chen, W. Zhang, and M. Lin, "A Closer Look at Machine Unlearning for Large Language Models," in Proc. ICLR, 2025.
[17] R. Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model," in Proc. NeurIPS, 2023, pp. 53728-53741.
[18] K. Meng et al., "Locating and Editing Factual Associations in GPT," in Proc. NeurIPS, 2022, pp. 17359-17372.
[19] K. Meng et al., "Mass-Editing Memory in a Transformer," in Proc. ICLR, 2023.
[20] C. Tan, Y. Qu, X. Li, H. Zhang, S. Cui, C. Chen, and L. Gao, "Wisdom is Knowing What not to Say: Hallucination-Free LLMs Unlearning via Attention Shifting," arXiv preprint arXiv:2510.17210, 2025.
[21] Z. Feng, Y. E. Xu, A. Robey, R. Kirk, X. Davies, Y. Gal, A. Schwarzschild, and J. Z. Kolter, "Existing Large Language Model Unlearning Evaluations Are Inconclusive," arXiv preprint arXiv:2506.00688, 2025.
[22] S. Min et al., "FActScore: Fine-Grained Atomic Evaluation of Factual Precision in Long Form Text Generation," in Proc. EMNLP, 2023, pp. 6042-6060.
[23] C. Guo et al., "On Calibration of Modern Neural Networks," in Proc. ICML, 2017, pp. 1321-1330.
[24] N. Li et al., "The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning," in Proc. ICML, 2024.
[25] W. Shi, J. Lee, Y. Huang, S. Malladi, J. Zhao, A. Holtzman, D. Liu, L. Zettlemoyer, N. A. Smith, and C. Zhang, "MUSE: Machine Unlearning Six-Way Evaluation for Language Models," in Proc. ICLR, 2025.
[26] Y. Huang et al., "Position: TrustLLM: Trustworthiness in Large Language Models," in Proc. ICML, 2024.
[27] G. Feretzakis, E. Vagena, K. Kalodanis, P. Peristera, D. Kalles, and A. Anastasiou, "GDPR and Large Language Models: Technical and Legal Obstacles," Future Internet, vol. 17, no. 4, art. 151, 2025. doi: 10.3390/fi17040151.
[28] European Parliament and Council, "Regulation (EU) 2024/1689 (EU AI Act)," Official Journal of the EU, 2024.
[29] A. Vaswani et al., "Attention is All You Need," in Proc. NeurIPS, 2017, pp. 5998-6008.
[30] Z. Jin, P. Cao, C. Wang, Z. He, H. Yuan, J. Li, Y. Chen, K. Liu, and J. Zhao, "RWKU: Benchmarking Real-World Knowledge Unlearning for Large Language Models," in Proc. NeurIPS Datasets and Benchmarks Track, 2024. arXiv:2406.10890.
[31] Y. Bang, Z. Ji, A. Schelten, A. Hartshorn, T. Fowler, C. Zhang, N. Cancedda, and P. Fung, "HalluLens: LLM Hallucination Benchmark," in Proc. ACL, 2025, pp. 24128-24156.
[32] B. Huang, C. Chen, X. Xu, A. Payani, and K. Shu, "Can Knowledge Editing Really Correct Hallucinations?" in Proc. ICLR, 2025.
 [33] A. Jacovi et al., "The FACTS Grounding Leaderboard: Benchmarking LLMs' Ability to Ground Responses to Long-Form Input," arXiv preprint arXiv:2501.03200, 2025.






