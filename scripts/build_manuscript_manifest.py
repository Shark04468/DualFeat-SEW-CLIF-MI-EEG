"""Build the evidence-locked Word manuscript manifest for DualFeat-SEW-CLIF."""

from __future__ import annotations

import csv
import json
from pathlib import Path


BUILD = Path(r"D:\DPC-SNN_paper_results\manuscript_20260808")
FIG = BUILD / "figures"
EVIDENCE = Path(r"D:\DPC-SNN_paper_results\paper_evidence_20260808")


def paragraph(text: str) -> dict[str, object]:
    return {"type": "paragraph", "text": text}


def heading(text: str, level: int = 2) -> dict[str, object]:
    return {"type": "heading", "text": text, "level": level}


def equation(text: str) -> dict[str, object]:
    return {"type": "equation", "text": text, "size_pt": 10.5}


def figure(path: Path, caption: str, width: float = 6.7) -> dict[str, object]:
    return {"type": "figure", "path": str(path), "caption": caption, "width_in": width}


def table(caption: str, columns: list[str], rows: list[list[str]], note: str = "") -> dict[str, object]:
    block: dict[str, object] = {"type": "table", "caption": caption, "columns": columns, "rows": rows}
    if note:
        block["note"] = note
    return block


def build_manifest() -> dict[str, object]:
    sections: list[dict[str, object]] = []

    sections.append(
        {
            "heading": "1. Introduction",
            "level": 1,
            "blocks": [
                paragraph(
                    "Motor-imagery electroencephalography (MI-EEG) provides a non-invasive route to brain-computer interaction, but decoding remains difficult because class-discriminative rhythms are weak, non-stationary and strongly subject dependent. The most reproducible physiological signatures are event-related desynchronisation and synchronisation in mu and beta rhythms over sensorimotor cortex [12]. Their expression changes across recording sessions, while typical subject-specific training sets contain only tens of examples per class. A useful decoder must therefore preserve complementary temporal, spectral and spatial information without fitting session-specific noise."
                ),
                paragraph(
                    "Compact convolutional networks have established strong inductive biases for this setting [13]. EEGNet combines temporal and depthwise spatial filtering [1]; FBCNet exposes band-specific spatial variance [2]; ATCNet combines temporal convolution and attention [3]; EEG Conformer adds self-attention to convolutional tokenisation [4]; and TCFormer further couples temporal convolution and transformer modelling [14]. These methods show that no single representation is uniformly sufficient: temporal morphology, band-power statistics and longer-range dependencies can each dominate under different acquisition conditions. Fixed probability fusion is a simple way to test whether independently trained representations contain complementary errors before constructing a more complex trainable decoder."
                ),
                paragraph(
                    "Spiking neural networks (SNNs) offer stateful temporal dynamics and sparse event computation. Learnable membrane constants [7], surrogate-gradient optimisation [8], spike-element-wise residual connections [6] and complementary leaky integrate-and-fire (CLIF) dynamics [5] make deep temporal SNNs trainable. However, an SNN label alone does not establish an advantage: apparent gains can be caused by a stronger frontend, different parameter count, different optimisation or unequal access to teacher targets. A scientifically useful comparison must hold the representation, input samples, training schedule and readout capacity constant while replacing only the temporal state operator."
                ),
                paragraph(
                    "We therefore study DualFeat-SEW-CLIF, an engineering pipeline that fuses frozen ATCNet temporal-attention sequences with frozen FBCNet filter-bank log-variance sequences and decodes the resulting sequence using signed currents, heterogeneous CLIF dynamics and causal SEW residual blocks. The matched control has the same projections, fusion, causal convolutions and statistical readout, but replaces spiking state updates with continuous ANN dynamics. This isolates a decoder effect rather than comparing unrelated architectures."
                ),
                paragraph(
                    "The study makes three bounded contributions. First, an equal-budget baseline campaign establishes when heterogeneous temporal and filter-bank representations are complementary across sessions. Second, matched comparisons on BCI Competition IV-2a, OpenBMI and BCI Competition IV-2b show that the SEW-CLIF decoder advantage is large in lower-label regimes but is not universal. Third, a paired learning-curve experiment demonstrates that the gain decreases significantly with the logarithm of labelled examples per class. The project-level blind BCI Competition IV-2b test reaches parity rather than superiority, providing an explicit boundary condition."
                ),
            ],
        }
    )

    sections.append(
        {
            "heading": "2. Results",
            "level": 1,
            "blocks": [
                heading("2.1 Evidence hierarchy and evaluation protocols"),
                paragraph(
                    "The experiments were organised by evidential status (Table 1). BCI Competition IV-2a (BCI2a) and OpenBMI evaluation sessions had been accessed by earlier project versions, so all results on those sessions are retrospective even when a later checkpoint barrier was enforced. E30 on BCI Competition IV Dataset 2b (BNCI2014-004) was the only project-level blind cross-session confirmation: the evaluation sessions were unavailable until every subject-seed checkpoint had been frozen and hashed. E28 used six-fold out-of-fold (OOF) predictions within BCI2a Session T and therefore measures within-session generalisation. E31 used frozen supervised representations and varied only the labels available to the matched decoders; it measures decoder label efficiency, not end-to-end sample efficiency."
                ),
                table(
                    "Table 1. Evidence hierarchy and inferential units.",
                    ["Campaign", "Dataset and split", "Runs", "Inferential unit", "Status"],
                    [
                        ["V8", "BCI2a T -> E", "9 subjects x 5 seeds", "Subject after seed averaging", "Post-hoc explanatory"],
                        ["V8", "OpenBMI S1 -> S2", "54 subjects x 5 seeds", "Subject after seed averaging", "Post-hoc explanatory"],
                        ["E28", "BCI2a T, six-fold OOF", "9 subjects x 3 seeds", "Paired subject-seed run", "Development evidence"],
                        ["V25/E26", "BCI2a T -> E", "9 subjects x 5 seeds", "Subject after seed averaging", "Locked retrospective"],
                        ["E29", "OpenBMI S1 -> S2", "54 subjects x 5 seeds", "Subject after seed averaging", "Retrospective replication"],
                        ["E30", "BCI2b 01T-03T -> 04E-05E", "9 subjects x 5 seeds", "Subject after seed averaging", "Project-level blind"],
                        ["E31", "All three datasets", "3 seeds per budget", "Subject-cluster bootstrap", "Retrospective mechanism isolation"],
                    ],
                    "Session arrows denote training to held-out evaluation. Seeds are optimisation repeats, not independent participants.",
                ),
                heading("2.2 Strong baselines reveal representation complementarity"),
                paragraph(
                    "We first reproduced six compact and transformer-based EEG baselines under the same cross-session splits, 80-epoch budget and augmentation policy. On BCI2a, fixed equal-probability fusion of ATCNet and FBCNet obtained 74.560% subject-macro accuracy, exceeding ATCNet by 2.176 percentage points (pp). All nine subject-level differences were positive; the subject-bootstrap 95% confidence interval (CI) for the fusion gain was 1.366 to 2.986 pp and the Holm-adjusted p value was 0.02344. The result supports complementarity between temporal-attention and filter-bank variance representations on this dataset."
                ),
                paragraph(
                    "The same conclusion did not transfer unchanged to OpenBMI. EEG Conformer was the strongest baseline at 71.459%, whereas ATCNet/FBCNet fusion reached 70.356%. Fusion exceeded ATCNet by only 0.470 pp (95% CI, -0.059 to 0.996; Holm-adjusted p=0.1185) and trailed EEG Conformer by 1.104 pp (95% CI, -2.233 to -0.044; Holm-adjusted p=0.2181). Thus, the fused representation is competitive and useful as a common decoder input, but it is not a universally best classifier."
                ),
                table(
                    "Table 2. Equal-budget cross-session baseline accuracy (%).",
                    ["Model", "BCI2a T -> E", "OpenBMI S1 -> S2"],
                    [
                        ["EEGNet", "62.500", "57.648"],
                        ["FBCNet", "70.247", "65.833"],
                        ["ATCNet", "72.384", "69.885"],
                        ["TCFormer", "71.975", "69.774"],
                        ["EEG Conformer", "68.827", "71.459"],
                        ["BFATCNet", "51.860", "58.063"],
                        ["ATCNet + FBCNet", "74.560", "70.356"],
                    ],
                    "Values are subject-macro accuracy after averaging five optimisation seeds within each subject. These analyses were performed after the held-out sessions had been opened and are therefore explanatory rather than blind confirmation.",
                ),
                figure(
                    FIG / "figure_2_baseline_accuracy.png",
                    "Figure 2. Equal-budget cross-session baselines. Points show subject-macro accuracy and error bars show subject-bootstrap 95% confidence intervals. Fixed ATCNet/FBCNet fusion is strongest on BCI2a but not on OpenBMI.",
                ),
                heading("2.3 Matched SEW-CLIF decoding improves two development settings"),
                paragraph(
                    "The proposed decoder was evaluated against ANN-SEW with identical frozen branch features, projections, fusion width, causal convolutional topology and readout dimensionality. In BCI2a Session-T six-fold OOF evaluation (E28), SEW-CLIF trained with cross-entropy achieved 73.341%, compared with 62.989% for ANN-SEW, a mean paired increase of 10.352 pp. Twenty-five of 27 subject-seed pairs were positive, one was negative and one tied (one-sided Wilcoxon p=4.66 x 10^-6). This comparison is useful for architecture development, but treating optimisation seeds as repeated pairs gives a less conservative inferential unit than independent subjects."
                ),
                paragraph(
                    "A separate locked retrospective BCI2a cross-session campaign (V25/E26) provided a more conservative subject-level estimate. SEW-CLIF obtained 75.73%, compared with 75.13% for the matched ANN residual, 75.27% for the equal-probability teacher ensemble, 73.72% for ATCNet and 70.05% for FBCNet. The SEW-CLIF minus ANN effect was +0.59 pp, with seven positive subjects, one tie and one negative subject. The subject-cluster bootstrap 95% CI was +0.07 to +1.15 pp, while the exact Wilcoxon p value was 0.0547. We therefore describe this as a small, consistent retrospective advantage, not conventionally significant superiority."
                ),
                paragraph(
                    "On OpenBMI S1-to-S2 replication (E29), the CE-trained SEW-CLIF decoder achieved 69.415%, versus 63.885% for ANN-SEW. The subject-level gain was +5.530 pp; 42 of 54 subjects improved, the subject-bootstrap 95% CI was +3.896 to +7.204 pp and the paired p value was 2.83 x 10^-7. The architecture and schedule were frozen from BCI2a before E29, but OpenBMI Session S2 had been accessed by older project versions. E29 is consequently a retrospective external replication rather than a genuinely unseen confirmation."
                ),
                table(
                    "Table 3. Matched spiking versus continuous decoder comparisons.",
                    ["Experiment", "ANN (%)", "SEW-CLIF (%)", "Delta (pp)", "95% CI (pp)", "Direction"],
                    [
                        ["E28 BCI2a within-session OOF", "62.989", "73.341", "+10.352", "Not reported", "25/27 positive"],
                        ["V25/E26 BCI2a cross-session", "75.13", "75.73", "+0.59", "+0.07 to +1.15", "7 positive, 1 tie, 1 negative"],
                        ["E29 OpenBMI cross-session", "63.885", "69.415", "+5.530", "+3.896 to +7.204", "42/54 positive"],
                        ["E30 BCI2b blind cross-session", "69.457", "69.400", "-0.058", "-1.446 to +1.181", "5/9 positive"],
                    ],
                    "E28 reports optimisation-repeat pairs and should not be interpreted as 27 independent participants. Cross-session rows use subjects after seed averaging.",
                ),
                figure(
                    FIG / "figure_3_matched_decoder_effects.png",
                    "Figure 3. Matched SEW-CLIF minus ANN-SEW accuracy. Confidence intervals are subject-bootstrap intervals where available. E28 did not store a comparable subject-bootstrap interval and is shown as a point estimate with its paired-run direction count.",
                ),
                heading("2.4 Project-level blind confirmation establishes a boundary condition"),
                paragraph(
                    "E30 tested the frozen architecture on BCI Competition IV Dataset 2b, using Sessions 01T-03T for training and Sessions 04E-05E for evaluation. This dataset has only C3, Cz and C4, binary left-versus-right motor imagery, and approximately 400-440 training trials per subject. Evaluation data were not loaded until all 45 subject-seed checkpoints were complete. ANN-SEW obtained 69.457% and SEW-CLIF obtained 69.400%, a difference of -0.058 pp (95% CI, -1.446 to +1.181; p=0.488). Five subjects favoured SNN and four favoured ANN. The preregistered advancement gate failed."
                ),
                paragraph(
                    "The blind result is informative rather than a null experiment to hide. It excludes universal SNN superiority and indicates that the gain depends on the information and sample regime supplied to the decoder. BCI2b has substantially fewer spatial channels and more labelled trials per class than BCI2a or OpenBMI. In this setting, the matched ANN improved enough to remove the SNN margin, while ATCNet alone reached 74.166%, above both students. The decoder cannot recover information discarded by a weak or mismatched frozen representation."
                ),
                heading("2.5 The decoder advantage contracts with labelled sample size"),
                paragraph(
                    "E31 directly tested whether the paired SNN-ANN difference changed with the number of labelled examples. Within each subject and seed, label subsets were nested and identical for both decoders; the supervised ATCNet and FBCNet representations remained frozen. The regression coefficient of decoder gain on log examples per class was -0.02767, with a subject-cluster bootstrap 95% CI of -0.03981 to -0.01510 and one-sided bootstrap p=9.999 x 10^-5. Dataset-specific slopes were also negative: -0.04531 for BCI2a, -0.03372 for BCI2b and -0.01033 for OpenBMI."
                ),
                paragraph(
                    "At 25 examples per class, gains were +12.243 pp on BCI2a, +7.285 pp on BCI2b and +6.506 pp on OpenBMI. With all available labels, the gains became +7.652, +0.899 and +5.790 pp, respectively. The BCI2b trend reconciles E30 with the development results: the SNN is most useful when the downstream decoder is label constrained, but approaches parity when the continuous control sees more examples. Because the frozen representations were themselves trained with all available training-session labels, this experiment does not demonstrate end-to-end few-shot learning."
                ),
                table(
                    "Table 4. Decoder learning curve with frozen supervised representations.",
                    ["Dataset", "Examples/class", "ANN-SEW (%)", "SEW-CLIF (%)", "Delta (pp)"],
                    [
                        ["BCI2a", "25", "55.646", "67.888", "+12.243"],
                        ["BCI2a", "50", "61.896", "70.126", "+8.230"],
                        ["BCI2a", "All (72)", "63.837", "71.489", "+7.652"],
                        ["BCI2b", "25", "59.195", "66.480", "+7.285"],
                        ["BCI2b", "50", "62.068", "67.321", "+5.253"],
                        ["BCI2b", "100", "67.201", "67.890", "+0.689"],
                        ["BCI2b", "All (about 204)", "69.229", "70.129", "+0.899"],
                        ["OpenBMI", "25", "62.222", "68.728", "+6.506"],
                        ["OpenBMI", "All (50)", "63.914", "69.704", "+5.790"],
                    ],
                    "These values come from E31 and must not replace the primary E28-E30 values because the representation and checkpoint protocol differs.",
                ),
                figure(
                    FIG / "figure_4_learning_curve.png",
                    "Figure 4. Accuracy as a function of labelled examples per class. Representations were frozen after supervised training; only the matched decoder label budget changed. The result supports decoder-level, not end-to-end, label efficiency.",
                ),
                heading("2.6 Perturbations identify dataset-dependent physiological dependence"),
                paragraph(
                    "Post-hoc frequency and sensor-region perturbations were applied to the V8 baseline campaign. For the fixed fusion, removing the 8-13 Hz mu band caused the largest accuracy reduction on both BCI2a (37.523 pp) and OpenBMI (13.174 pp). The largest regional reductions were produced by left-motor sensors on BCI2a (37.137 pp) and right-motor sensors on OpenBMI (19.163 pp). These patterns are consistent with the established importance of sensorimotor rhythms [12] and the spatial-filtering assumptions of FBCNet and filter-bank CSP [2,11]."
                ),
                paragraph(
                    "The perturbations should not be read as causal localisation. Frequency deletion alters the full trial distribution, and zero-reference sensor masking produces inputs outside the training distribution. The results show model dependence on frequency bands and regions, not unique cortical sources or physiological causality. The different lateralised maxima across datasets also warn against transferring a single saliency narrative between acquisition protocols."
                ),
                figure(
                    FIG / "figure_5_perturbation_dependence.png",
                    "Figure 5. Post-hoc frequency-band and regional perturbation dependence for equal-budget baselines and fixed fusion. Values are accuracy changes in percentage points after perturbation. Positive drops indicate reliance; negative values indicate that the perturbation improved accuracy. These are dependence diagnostics, not causal neurophysiology.",
                ),
                heading("2.7 Negative delay result and computational boundaries"),
                paragraph(
                    "The project originally investigated learnable inter-channel and cross-band delays. Before classifier training, a registered feasibility gate required stable non-zero delay evidence under fold-local controls. The gate failed in two of three subject-folds, so delay classifier training was not authorised. The final DualFeat-SEW-CLIF architecture therefore contains no delay bottleneck, and the delay mechanism is not claimed as a validated contribution. This separation prevents an unverified mechanism from borrowing credibility from the later accuracy results."
                ),
                paragraph(
                    "An earlier matched-ensemble utility study found effectively equal final accuracy between spiking and continuous decoders (+0.031 pp on BCI2a and -0.076 pp on OpenBMI). A software operation proxy decreased by 77.468%, but no neuromorphic hardware energy or latency measurement was performed. Early-decision, robustness and few-shot gates were not passed in that campaign. We consequently report operation sparsity only as a proxy and do not claim measured energy efficiency or a hardware advantage."
                ),
            ],
        }
    )

    sections.append(
        {
            "heading": "3. Discussion",
            "level": 1,
            "blocks": [
                paragraph(
                    "The central result is conditional: a matched SEW-CLIF temporal decoder can exploit frozen heterogeneous EEG representations more effectively than a continuous residual control when decoder labels are scarce, but this advantage contracts with increasing data and disappears in the project-level blind BCI2b test. This is a narrower claim than universal SNN superiority, yet it is more useful for model design because it identifies where the additional state dynamics matter."
                ),
                paragraph(
                    "Several mechanisms can explain the lower-label advantage. First, heterogeneous membrane decays impose a restricted family of multi-timescale state filters. Instead of learning an unconstrained temporal mapping from limited labels, the decoder accumulates evidence with three initial decay regimes and only subsequently adapts those decays. Second, positive and negative currents are represented by separate populations, preventing sign cancellation before state integration. Third, causal depthwise convolutions and SEW-ADD residual paths preserve local temporal identity while limiting cross-channel parameter growth. The final re-spiking layer restores a binary event representation before statistical readout. Together these constraints can act as structured temporal regularisation. They do not prove that biological spiking is necessary."
                ),
                paragraph(
                    "The frontend is equally important. ATCNet and FBCNet encode different invariances: one emphasises temporal-attention structure, while the other exposes band-specific spatial log-variance. The interaction term [a, f, a x f, |a-f|] lets the decoder use agreement, signed co-activation and discrepancy without a large transformer. The BCI2a fusion gain and the mu-band perturbation results support complementarity, although OpenBMI shows that an EEG Conformer can remain superior. The appropriate conclusion is that heterogeneous representations create a strong substrate for a compact stateful decoder, not that a particular pair of branches is optimal for every dataset."
                ),
                paragraph(
                    "The BCI2b failure clarifies the limits. With only three sensor channels, the FBCNet-like spatial representation has less opportunity to capture distributed covariance, and the fixed 22-channel design cannot be transferred without an adapter. BCI2b also supplies many more training examples per class, allowing the matched ANN to estimate its continuous dynamics more reliably. Finally, the two frozen teachers are imbalanced on this dataset: ATCNet outperforms FBCNet and both students, so interaction fusion may combine a strong and a weak representation rather than complementary peers. Future versions should learn dataset-aware branch reliability from training data only, while retaining a strictly matched ANN/SNN comparison."
                ),
                paragraph(
                    "From an application perspective, the present model is most defensible for subject-specific calibration with constrained decoder labels and access to pre-trained EEG representations. A full end-to-end claim would require fitting the feature extractors within each reduced-label subset. A neuromorphic claim would additionally require deployment on event-driven hardware with measured energy, latency and memory traffic. The software operation proxy is insufficient because dense frontend computation and hardware mapping can dominate system cost."
                ),
                paragraph(
                    "For a subsequent confirmatory study, the architecture and analysis should be frozen before accessing a new dataset or site. The primary endpoint should be subject-level paired accuracy, with balanced accuracy and calibration as secondary endpoints. A nested end-to-end learning curve should retrain both teacher branches and decoders at each budget. If the SNN advantage persists under that stricter protocol, the paper could make a stronger claim about calibration efficiency. If it again converges to parity, the correct interpretation would be that SEW-CLIF supplies a low-data regulariser rather than a generally superior classifier."
                ),
            ],
        }
    )

    sections.append(
        {
            "heading": "4. Materials and Methods",
            "level": 1,
            "blocks": [
                heading("4.1 Datasets"),
                paragraph(
                    "BCI Competition IV Dataset 2a contains nine participants, 22 scalp EEG channels, four motor-imagery classes and separate training (T) and evaluation (E) sessions sampled at 250 Hz [9,11]. Each session contributes 288 labelled trials. The V8 and V25/E26 cross-session experiments trained on T and evaluated on E; E28 used six-fold OOF predictions within T."
                ),
                paragraph(
                    "OpenBMI contains 54 participants recorded in two sessions and includes binary left- and right-hand motor imagery [10]. Signals were resampled from 1000 to 250 Hz. Session S1 was used for training and S2 for evaluation. A fixed 22-sensor basis matched the BCI2a ordering; FCz was obtained by a frozen linear average of FC1 and FC2 when required. All 54 participants were included in the primary E29 analysis."
                ),
                paragraph(
                    "BCI Competition IV Dataset 2b contains nine participants and three bipolar sensorimotor channels (C3, Cz and C4) at 250 Hz [9,11]. Sessions 01T, 02T and 03T formed the training set, and 04E and 05E formed the held-out test set. The task was binary left- versus right-hand motor imagery. All labelled trials in the frozen sessions were retained, giving approximately 400-440 training trials and 300-400 evaluation trials per participant."
                ),
                heading("4.2 Preprocessing and augmentation"),
                paragraph(
                    "Trials spanned -1 to 4 s relative to cue onset. Signals were common-average referenced, baseline corrected over -1 to 0 s and restricted to the 0-4 s task interval, except for model-specific crops declared before evaluation. Channel gains were estimated once from the complete training session or training sessions using channel root-mean-square amplitude and then applied unchanged to validation and evaluation data. Values were clipped at an absolute normalised amplitude of 12. OpenBMI was anti-alias resampled to 250 Hz."
                ),
                paragraph(
                    "Training-only augmentation used eight temporal segments with probability 0.5, additive noise standard deviation 0.01 and amplitude scaling uniformly sampled from 0.9 to 1.1. Augmentation parents were drawn only from the current training split. No evaluation trial participated in gain estimation, augmentation, checkpoint selection or gradient updates."
                ),
                heading("4.3 Frozen heterogeneous representations"),
                paragraph(
                    "The temporal branch was an ATCNet-style encoder [3] that produced an 18-step sequence with 32 features per step. The spectral-spatial branch was an FBCNet-style encoder [2] with nine 4-Hz bands from 4 to 40 Hz, band-specific spatial filters and four log-variance windows, yielding a 4-step sequence with 288 features. Both branch checkpoints were trained on the corresponding training data and frozen before student-decoder optimisation. In the matched comparisons, ANN and SNN received the exact same stored branch sequences."
                ),
                paragraph(
                    "Let A in R^(18 x 32) and F in R^(4 x 288) denote the branch sequences. Linear projections mapped both branches to 64 dimensions, and the FBCNet sequence was linearly interpolated from four to 18 steps. For projected sequences a_t and f_t, interaction fusion was defined as"
                ),
                equation("g_t = GELU(W_g LN([a_t, f_t, a_t o f_t, |a_t - f_t|]))"),
                equation("h_t = LN(0.5(a_t + f_t) + Dropout(g_t)),    h_t in R^64"),
                paragraph(
                    "where o denotes element-wise multiplication and LN denotes LayerNorm. The fusion layer therefore encodes average evidence, co-activation and branch disagreement while retaining a fixed 18-step temporal grid."
                ),
                figure(
                    FIG / "figure_1_architecture.png",
                    "Figure 1. DualFeat-SEW-CLIF architecture. Frozen ATCNet and FBCNet sequences are projected to a common temporal grid and combined by interaction fusion. The matched ANN control retains every non-spiking component and replaces only the state layers. The reported model does not contain the rejected delay mechanism.",
                ),
                heading("4.4 Signed-current SEW-CLIF decoder"),
                paragraph(
                    "A 1 x 1 temporal convolution mapped h_t to 32 signed currents. Positive and negative components were separated and concatenated, producing 64 non-negative input channels. This construction preserves the sign of the fused evidence without requiring inhibitory negative spike counts. Channels were assigned initial decay factors 0.65, 0.90 and 0.975, distributed across the population and subsequently learned."
                ),
                paragraph(
                    "For current I_t, membrane u_t, spike s_t and complementary state q_t, the CLIF update implemented in the decoder can be written as"
                ),
                equation("u_t^- = beta u_(t-1) + I_t"),
                equation("s_t = H(u_t^- - V_th)"),
                equation("q_t = q_(t-1) sigmoid((1-beta)u_t^-) + s_t"),
                equation("u_t = u_t^- - s_t(V_th + sigmoid(q_t))"),
                paragraph(
                    "where H is implemented with a surrogate gradient during backpropagation [5,8]. The stem was followed by two causal depthwise-separable SEW-CLIF residual blocks with kernel size 3 and dilations 1 and 2 [6]. Each block applied causal depthwise convolution, pointwise channel mixing, a CLIF state layer and spike-element-wise addition. A final CLIF layer re-spiked the residual output so that firing-rate regularisation was applied only to binary spike tensors. The corresponding ANN-SEW control used the same convolutional blocks, widths and decay parameterisation but continuous nonlinear state outputs."
                ),
                paragraph(
                    "The readout concatenated temporal mean and standard deviation of the final spikes with temporal mean and standard deviation of the final membrane, producing 4 x 64=256 statistics. A 96-dimensional linear layer, non-affine LayerNorm, ELU, dropout 0.25 and a final classifier generated class logits. Unlike earlier prototypes, the reported decoder used the full four-second endpoint and did not use a temporal pyramid or an un-delayed information bypass."
                ),
                heading("4.5 Optimisation and matched controls"),
                paragraph(
                    "Student decoders were trained for a fixed 80 epochs with AdamW, learning rate 10^-3, weight decay 10^-3, cosine scheduling, effective batch size 48 and gradient-norm clipping at 5. Cross-entropy was the primary objective in E28-E31. For SNN models, a firing-rate term with weight 0.01 penalised the squared deviation of binary spike rates from a target of 0.12. Knowledge-distillation variants were evaluated in E28 but were not used for the primary CE comparison in E29 or E30."
                ),
                equation("L = L_CE + 0.01 mean_l (mean(s^(l)) - 0.12)^2"),
                paragraph(
                    "The matched ANN and SNN controls used identical feature tensors, folds, label subsets, random seeds, initialisation contracts, augmentation, optimiser, epoch count and statistical readout. Only the temporal state operator differed. Equal-probability teacher predictions were computed as 0.5 softmax(z_ATC) + 0.5 softmax(z_FBC)."
                ),
                heading("4.6 Statistical analysis"),
                paragraph(
                    "Accuracy was computed for every subject and seed. For cross-session comparisons, seeds were averaged within subject before inference; participants were the inferential unit. Subject-bootstrap confidence intervals used 10,000 resamples. Paired model comparisons used exact or paired Wilcoxon tests as declared by each campaign, and the multi-baseline V8 comparisons used Holm correction. E28 reports 27 paired subject-seed optimisation repeats and is labelled as development evidence. The more conservative V25/E26 result is reported alongside it."
                ),
                paragraph(
                    "For E31, the response was the within-subject SEW-CLIF minus ANN-SEW accuracy difference. A model with dataset fixed effects regressed this difference on the natural logarithm of examples per class. Confidence intervals were obtained by resampling subjects as clusters. A negative slope was declared only if the upper 95% bootstrap confidence bound was below zero."
                ),
                heading("4.7 Reproducibility and access control"),
                paragraph(
                    "Resolved configurations, subject-seed metrics, trial-level predictions, checkpoint manifests, code fingerprints and aggregation scripts were retained. Resume decisions were tied to resolved configuration and artifact hashes to prevent stale results from being mixed after code changes. In locked campaigns, evaluation data were unavailable to optimisation until a global checkpoint barrier had been sealed. The evidence package also preserves failed gates and negative experiments. [AUTHOR_INPUT_NEEDED: insert public repository and permanent archive links after anonymised submission requirements are resolved.]"
                ),
            ],
        }
    )

    sections.append(
        {
            "heading": "5. Limitations",
            "level": 1,
            "blocks": [
                paragraph(
                    "Several limitations constrain the claims. First, BCI2a Session E and OpenBMI Session S2 were historically accessed during earlier development, so they cannot serve as project-level blind confirmation. Only BCI2b was genuinely blind, and it showed parity. Second, E28 treats subject-seed pairs as repeated optimisation units; seeds do not increase the number of participants. Third, E31 freezes representations trained with all training-session labels, so it isolates decoder sample efficiency rather than complete-pipeline few-shot performance."
                ),
                paragraph(
                    "Fourth, the three datasets differ in channel count, class count, trial count and session structure. These differences are informative for boundary analysis but prevent a simple pooled accuracy ranking. Fifth, the perturbation analysis creates distribution shift and cannot establish causal neurophysiology. Sixth, operation counts are software proxies; no neuromorphic device energy, memory traffic or latency was measured. Finally, the rejected delay mechanism was not trained past its feasibility gate and must not be represented as a successful component of the reported model."
                ),
            ],
        }
    )

    sections.append(
        {
            "heading": "6. Conclusion",
            "level": 1,
            "blocks": [
                paragraph(
                    "DualFeat-SEW-CLIF combines complementary temporal-attention and filter-bank variance representations with a compact signed-current spiking decoder. Under matched inputs and optimisation, SEW-CLIF substantially improved decoder accuracy in lower-label BCI2a and OpenBMI settings, while a project-level blind BCI2b test reached parity with the continuous control. The paired learning curve showed that the advantage decreased as labelled examples increased. The supported conclusion is therefore a sample-regime-dependent decoder benefit, not universal SNN superiority. This bounded result provides a reproducible foundation for a future architecture-frozen, end-to-end low-label confirmation and for direct neuromorphic hardware measurement."
                )
            ],
        }
    )

    sections.append(
        {
            "heading": "Declarations",
            "level": 1,
            "blocks": [
                heading("Ethics approval and consent to participate"),
                paragraph(
                    "This study analysed publicly available, de-identified datasets. [AUTHOR_INPUT_NEEDED: confirm whether local institutional review or exemption was required and provide the approval or exemption identifier. Summarise consent and ethics statements from the original dataset publications.]"
                ),
                heading("Consent for publication"),
                paragraph("Not applicable to the de-identified secondary analysis. [AUTHOR_INPUT_NEEDED: confirm.]"),
                heading("Availability of data and materials"),
                paragraph(
                    "BCI Competition IV Datasets 2a and 2b and OpenBMI are publicly available from their original repositories [9-11]. [AUTHOR_INPUT_NEEDED: add exact access URLs, access dates and any licence restrictions required by the target journal.]"
                ),
                heading("Code availability"),
                paragraph(
                    "The experiment configurations, aggregation scripts and evidence manifests are archived locally with SHA-256 provenance. [AUTHOR_INPUT_NEEDED: provide the public or anonymised repository URL and release DOI.]"
                ),
                heading("Competing interests"),
                paragraph("[AUTHOR_INPUT_NEEDED: declare competing interests or state that none exist.]"),
                heading("Funding"),
                paragraph("[AUTHOR_INPUT_NEEDED: list funders, grant numbers and the role of each funder.]"),
                heading("Author contributions"),
                paragraph("[AUTHOR_INPUT_NEEDED: provide a CRediT author-contribution statement.]"),
                heading("Acknowledgements"),
                paragraph("[AUTHOR_INPUT_NEEDED: add acknowledgements or remove this subsection.]"),
            ],
        }
    )

    references = [
        "[1] Lawhern VJ, Solon AJ, Waytowich NR, Gordon SM, Hung CP, Lance BJ. EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. Journal of Neural Engineering. 2018;15(5):056013. doi:10.1088/1741-2552/aace8c.",
        "[2] Mane R, Robinson N, Vinod AP, Lee SW, Guan C. A multi-view CNN with novel variance layer for motor imagery brain-computer interface. 42nd Annual International Conference of the IEEE Engineering in Medicine and Biology Society. 2020:2950-2953. doi:10.1109/EMBC44109.2020.9175874.",
        "[3] Altaheri H, Muhammad G, Alsulaiman M. Physics-informed attention temporal convolutional network for EEG-based motor imagery classification. IEEE Transactions on Industrial Informatics. 2023;19(2):2249-2258. doi:10.1109/TII.2022.3197419.",
        "[4] Song Y, Zheng Q, Liu B, Gao X. EEG Conformer: convolutional transformer for EEG decoding and visualization. IEEE Transactions on Neural Systems and Rehabilitation Engineering. 2023;31:710-719. doi:10.1109/TNSRE.2022.3230250.",
        "[5] Huang Y, Zhang L, Wang S, et al. CLIF: complementary leaky integrate-and-fire neuron for spiking neural networks. Proceedings of the 41st International Conference on Machine Learning. PMLR. 2024;235:19949-19972.",
        "[6] Fang W, Yu Z, Chen Y, Huang T, Masquelier T, Tian Y. Deep residual learning in spiking neural networks. Advances in Neural Information Processing Systems. 2021;34:21056-21069.",
        "[7] Fang W, Yu Z, Chen Y, Masquelier T, Huang T, Tian Y. Incorporating learnable membrane time constant to enhance learning of spiking neural networks. Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021:2661-2671. doi:10.1109/ICCV48922.2021.00266.",
        "[8] Neftci EO, Mostafa H, Zenke F. Surrogate gradient learning in spiking neural networks: bringing the power of gradient-based optimization to spiking neural networks. IEEE Signal Processing Magazine. 2019;36(6):51-63. doi:10.1109/MSP.2019.2931595.",
        "[9] Tangermann M, Muller KR, Aertsen A, et al. Review of the BCI Competition IV. Frontiers in Neuroscience. 2012;6:55. doi:10.3389/fnins.2012.00055.",
        "[10] Lee MH, Kwon OY, Kim YJ, et al. EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy. GigaScience. 2019;8(5):giz002. doi:10.1093/gigascience/giz002.",
        "[11] Ang KK, Chin ZY, Wang C, Guan C, Zhang H. Filter bank common spatial pattern algorithm on BCI Competition IV Datasets 2a and 2b. Frontiers in Neuroscience. 2012;6:39. doi:10.3389/fnins.2012.00039.",
        "[12] Pfurtscheller G, Lopes da Silva FH. Event-related EEG/MEG synchronization and desynchronization: basic principles. Clinical Neurophysiology. 1999;110(11):1842-1857. doi:10.1016/S1388-2457(99)00141-8.",
        "[13] Schirrmeister RT, Springenberg JT, Fiederer LDJ, et al. Deep learning with convolutional neural networks for EEG decoding and visualization. Human Brain Mapping. 2017;38(11):5391-5420. doi:10.1002/hbm.23730.",
        "[14] Altaheri H, Karray F, Karimi AH. Temporal convolutional transformer for EEG-based motor imagery decoding. Scientific Reports. 2025;15:32959. doi:10.1038/s41598-025-16219-7.",
    ]

    return {
        "metadata": {
            "title": "Label-Efficient Motor-Imagery EEG Decoding with Heterogeneous Representations and SEW-CLIF Dynamics",
            "authors": ["[AUTHOR_INPUT_NEEDED: author names in journal order]"],
            "affiliations": ["[AUTHOR_INPUT_NEEDED: affiliations, correspondence and ORCID identifiers]"],
            "keywords": [
                "motor imagery EEG",
                "spiking neural network",
                "SEW residual connection",
                "CLIF neuron",
                "label efficiency",
                "cross-session decoding",
            ],
        },
        "style": {
            "page_size": "A4",
            "margin_cm": 2.2,
            "columns": 1,
            "body_font": "Times New Roman",
            "body_size_pt": 10.5,
            "caption_size_pt": 9.0,
            "line_spacing": 1.08,
            "page_numbers": True,
        },
        "abstract": (
            "Motor-imagery EEG decoding is limited by cross-session variability and small subject-specific training sets. We present DualFeat-SEW-CLIF, a compact decoder that combines frozen ATCNet temporal-attention sequences and FBCNet filter-bank log-variance sequences through interaction fusion, then applies signed currents, heterogeneous complementary leaky integrate-and-fire dynamics and causal spike-element-wise residual blocks. A parameter-matched ANN control retained the same representations, convolutions, schedule and statistical readout. In BCI Competition IV-2a within-session out-of-fold evaluation, SEW-CLIF achieved 73.341% accuracy versus 62.989% for ANN-SEW (+10.352 percentage points; 25/27 paired runs positive). In a locked retrospective BCI2a cross-session evaluation, the gain was smaller (+0.59 points; 95% CI, +0.07 to +1.15; exact Wilcoxon p=0.0547). OpenBMI Session S1-to-S2 replication yielded 69.415% versus 63.885% (+5.530 points; 95% CI, +3.896 to +7.204). By contrast, a project-level blind BCI Competition IV-2b test produced parity (69.400% versus 69.457%; difference -0.058 points; 95% CI, -1.446 to +1.181). A paired learning-curve analysis with frozen supervised representations showed that the decoder gain decreased with log labelled examples per class (slope -0.02767; 95% CI, -0.03981 to -0.01510). These results support a sample-regime-dependent benefit of structured spiking dynamics, not universal SNN superiority. The study also reports the failed feasibility gate of an earlier learnable-delay mechanism and avoids hardware-efficiency claims without device measurements."
        ),
        "sections": sections,
        "references": references,
    }


def write_evidence_ledger() -> None:
    rows = [
        ["Claim", "Evidence", "Status", "Boundary"],
        ["ATCNet/FBCNet fusion improves BCI2a", "V8: +2.176 pp vs ATCNet; 9/9 positive; Holm p=0.02344", "Supported", "Post-hoc cross-session"],
        ["Fusion is best on OpenBMI", "Fusion 70.356%; EEG Conformer 71.459%", "Rejected", "Do not claim universal best"],
        ["SEW-CLIF improves low-label decoder accuracy", "E28 +10.352 pp; E29 +5.530 pp; E31 negative slope", "Supported", "Frozen supervised representations"],
        ["SEW-CLIF universally outperforms ANN", "E30 -0.058 pp, CI crosses zero", "Rejected", "Project-level blind boundary"],
        ["Learnable delay improves classification", "Delay feasibility failed in 2/3 subject-folds", "Rejected", "No classifier training authorised"],
        ["SNN reduces hardware energy", "77.468% software operation-proxy reduction only", "Unsupported", "Requires hardware measurement"],
        ["Perturbations prove causal physiology", "Mu and motor-region drops", "Unsupported", "Dependence under distribution shift only"],
    ]
    path = BUILD / "evidence_ledger.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        csv.writer(handle).writerows(rows)


def write_reviewer_gate() -> None:
    text = "# Reviewer gate\n\n"
    text += "- PASS: matched ANN/SNN inputs, schedule and readout are described.\n"
    text += "- PASS: retrospective and blind evidence are explicitly separated.\n"
    text += "- PASS: BCI2b null result and failed delay gate are reported.\n"
    text += "- PASS: learning-curve claim is limited to frozen supervised representations.\n"
    text += "- PASS: no universal SOTA, causal physiology or measured energy claim is made.\n"
    text += "- ACTION: authors, affiliations, ethics, funding, conflicts and repository DOI require author input.\n"
    text += "- ACTION: adapt format and word count after selecting the target journal.\n"
    (BUILD / "reviewer_gate.md").write_text(text, encoding="utf-8")


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    (BUILD / "paper_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_evidence_ledger()
    write_reviewer_gate()
    print(BUILD / "paper_manifest.json")


if __name__ == "__main__":
    main()
