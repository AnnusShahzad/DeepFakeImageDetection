

# AI-Generated Image Detection and Generator Attribution: A Unified Study on MS-COCO-Derived Synthetic Images

Muhammad Annus Shahzad

Abd Ur Rehman

Zohaib Hussain

## I. INTRODUCTION

### A. Importance

The rapid maturation of text-to-image generative models has made photorealistic synthetic imagery universally accessible. Systems such as Stable Diffusion [1], DALL-E 3, and MidJourney v6 produce images that are visually indistinguishable from photographs. Such images circulate widely on social media, news platforms, and the open web. This development carries serious societal risks because synthetic imagery can be weaponized for misinformation campaigns, fabricated evidence, non-consensual content, and fraud. Trust in visual evidence underpins journalism, law, and public discourse. Robust automatic detection of AI-generated images is therefore a problem of first-order importance [2].

### B. Background

Early work on synthetic image forensics targeted images produced by Generative Adversarial Networks (GANs). Wang et al. [3] showed that GAN images leave subtle and consistent artifacts such as spectral anomalies that a CNN classifier can exploit, which yields near-perfect in-distribution accuracy. Frequency-domain analyses demonstrated that upsampling operations in GANs fail to reproduce natural spectral statistics [4], [5]. The field has since shifted. Latent diffusion models (LDMs) [1] now dominate image synthesis. Their generation process differs fundamentally from GANs and leaves different and weaker artifacts. Several studies have shown that detectors trained on GAN imagery generalize poorly to diffusion imagery [6], [7]. In response, a second family of approaches has emerged that builds on foundation models. DE-FAKE [8] and most prominently Ojha et al. [9] demonstrated that frozen CLIP [10] feature spaces combined with a simple nearest-neighbor or linear-probing classifier generalize far better to unseen generators than end-to-end trained CNNs do. A parallel and less mature line of work addresses *generator attribution*, that is, identifying which generative model produced a synthetic image. Known approaches use fingerprint inversion [11], hidden model fingerprints [12], or supervised classification over generator classes [8]. The recently released MS-COCOAI dataset [13] provides for the first time a large controlled benchmark built on MS-COCO [14] with five modern generators (Stable Diffusion 3, Stable Diffusion 2.1, SDXL, DALL-E 3, and MidJourney v6) that supports both tasks under a single experimental protocol.

### C. Problem Statement

Given an image that either originates from MS-COCO or was synthesized from an MS-COCO caption by one of five modern generative models (Stable Diffusion 3, Stable Diffusion 2.1, SDXL, DALL-E 3, MidJourney v6), we address two problems. **(P1) Binary detection** classifies the image as real or AI-generated. **(P2) Generator attribution** identifies which of the five generators produced an image that is classified as fake. We study both problems within a unified framework. We compare artifact-driven CNN detectors against foundation-model-based approaches and ask whether a single multi-task model can perform both roles.

### D. Research Gap

Three gaps in the existing literature motivate this work. **First**, most well-known detectors and benchmarks were developed in the GAN era or on face-centric data (FFHQ, CelebA). Evaluations on natural scene imagery from the diffusion era with controlled prompts remain scarce [6], [15]. **Second**, detection and attribution have evolved as disjoint research threads. Detection papers report real-versus-fake accuracy only, while attribution papers (e.g., fingerprint inversion [11] and fingerprint verification [12]) assume the image is already known to be fake and typically operate on GAN models rather than modern diffusion-based services. Very few works evaluate both tasks jointly on the same data. **Third**, existing attribution studies rarely cover closed-source generators such as DALL-E 3 and MidJourney v6 because no suitable benchmark existed. MS-COCOAI [13] fills exactly this void but has, at the time of writing, little published detector or attribution analysis, leaving the practical difficulty of these two tasks on this benchmark open.

### E. Evidence

Several observations substantiate the urgency and feasibility of this study. (i) The realism of synthetic images has reached the point where human observers fail. Prior work reports that untrained humans classify AI-synthesized faces as real in the majority of trials [16]. Automated detectors show large cross-generator drops. CNNSpot-style detectors that reach 100% AP on their training generator fall to roughly 92.6% average AP across 11 generators even after augmentation. On unseen diffusion models they fall far lower [3], [6]. (ii) Generative-model misuse is rising globally. Reported deepfake incident counts have grown steeply year over year and major social platforms host millions of synthetic-media

posts monthly [2]. (iii) The MS-COCOAI benchmark [13] offers a controlled testbed of 96,000 images with paired real and synthetic captions and balanced five-generator attribution labels. Its published frequency-domain ResNet-50 baseline reaches only 0.80 accuracy on binary detection and 0.45 on attribution, which shows both that substantial headroom remains and that attribution is markedly harder.

### F. Aims

The overall aim of this project is to develop and rigorously evaluate deep learning models for AI-generated image detection and generator attribution on modern diffusion-era imagery. A further aim is to characterize how detection difficulty varies across generators.

### G. Objectives

- 1) **O1:** Fine-tune and evaluate at least two CNN-based detector architectures (e.g., ResNet-50 and Efficient-Net) with standard anti-overfitting augmentations on the MS-COCOAI detection task (Label\_A), reporting Accuracy, F1, and AUC.
- 2) **O2:** Fine-tune and evaluate at least one vision-language or foundation-model-based detector (e.g., frozen CLIP features with linear probing) under identical splits and metrics.
- 3) **O3:** Build and evaluate multi-class generator attribution models on Label\_B (six classes: real plus five generators), reporting Accuracy, macro-F1, and per-class confusion analysis.
- 4) **O4:** Investigate a unified multi-task model with a shared backbone and two heads (detection and attribution). Compare it against the separate single-task models of O1 to O3.
- 5) **O5:** Analyze per-generator difficulty by identifying which generators are hardest to detect or to attribute. Include qualitative visualizations of representative successes and failures.

## II. LITERATURE REVIEW

### A. Survey of Reviewed Papers

Table I summarize the 17 research papers that we reviewed. They cover detection and attribution methods, datasets, metrics, headline performance, and key limitations. Papers are grouped into four thematic families discussed in Section II-B.

For a consistent reading of the survey we define the evaluation metrics that appear in its Metric(s) column. Let TP, FP, TN, and FN denote true positives, false positives, true negatives, and false negatives for the real-versus-fake decision.

- **Accuracy (ACC)** is  $(TP + TN) / (TP + TN + FP + FN)$ , the fraction of correctly classified images. It is meaningful on balanced test sets such as the ones used by the reviewed studies.
- **Average Precision (AP)** summarizes the precision-recall curve as the precision averaged over all recall levels. It is robust to class imbalance and threshold choice, and is the standard headline metric in the CNNSpot evaluation protocol.

- **Area Under the ROC Curve (AUC)** is the probability that a detector scores a randomly chosen fake image higher than a randomly chosen real image. It aggregates detector quality across all operating thresholds.
- **F1 score** is the harmonic mean of precision and recall,  $2 \cdot (\text{precision} \cdot \text{recall}) / (\text{precision} + \text{recall})$ , and is preferred when false positives and false negatives carry comparable cost.
- **FPR@95% TPR** (false positive rate at 95% true positive rate) is the fraction of real images misclassified as fake when the detector is tuned to catch 95% of fake images. It measures how many benign images a deployed detector would flag at a realistic recall target, so lower is better.
- **Pd@1% FAR** (probability of detection at 1% false alarm rate) is the complement setting of the same idea: the fraction of fake images detected when at most 1% of real images may be falsely flagged.

Where a paper reports scores averaged over several held-out generators, the table marks the value with “avg.”.

### B. Critical Discussion

*1) Theme 1: Artifact-driven CNN detectors of the GAN era.:* The first family of works [3], [4], [5], [17], [18] treats detection as a supervised classification problem and exploits low-level generative artifacts such as checkerboard upsampling patterns, anomalous frequency spectra, and color-detail inconsistencies. CNNSpot [3] is the canonical result. With blur and JPEG augmentation it reaches 92.6% average AP across 11 GAN generators. This was a landmark improvement over un-augmented baselines that collapse on unseen generators. Frequency analyses [4], [5] provided the interpretive foundation: GAN upsampling fails to reproduce natural spectral falloff. The collective limitation of this family is *artifact overfitting*. The artifacts it exploits are properties of the GAN training stack rather than of “fakeness” itself. When latent diffusion models replaced GANs as the dominant generators, these assumptions broke. Ricker et al. [6] and Corvi et al. [7], [15] both show severe performance drops of GAN-era detectors on diffusion imagery. Gragnaniello et al. [17] had already documented brittleness within the GAN family itself. The lesson is that detectors must be retrained or rethought for each new generation technology.

*2) Theme 2: Diffusion-era detectors and reconstruction-based cues.:* The second family adapts to diffusion models either by retraining artifact detectors on diffusion data [6] or by introducing diffusion-specific signals. DIRE [19] is the strongest representative of the latter. Because a real image reconstructs itself under DDIM inversion-reconstruction better than a generated one does, the reconstruction error map is a discriminative feature, yielding 99.9% ACC and 100% AP on unseen diffusion models. However, DIRE requires two full diffusion forward passes per test image. Its mechanism is also inherently diffusion-specific. A detector tied to one generation paradigm trades the previous generation’s blindness for the next’s. This theme illustrates an arms-race pattern in which each detector family is coupled to the artifacts of its era’s generators.

TABLE I. LITERATURE SURVEY OF AI-GENERATED IMAGE DETECTION AND ATTRIBUTION (PART I: GAN-ERA AND DIFFUSION-ERA DETECTORS).

| Reference                | Year | Method                                                                            | Dataset(s)                                               | Metric(s)      | Performance                                                                                                   | Key Limitation(s)                                                                       |
|--------------------------|------|-----------------------------------------------------------------------------------|----------------------------------------------------------|----------------|---------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| Wang et al. [3]          | 2020 | CNNSpot: ResNet-50 with blur and JPEG augmentation                                | ProGAN train set; test on 11 GAN generators              | AP             | 100% AP on the ProGAN testbed and 92.6% avg AP across 11 generators (Blur+JPEG 0.1)                           | GAN-centric with strong degradation on unseen diffusion-era generators                  |
| Frank et al. [4]         | 2020 | Spectral features with a CNN on the DCT spectrum                                  | GAN-generated CelebA images                              | ACC            | Clear spectral separation of GAN and real with large ACC gains over the RGB baseline                          | GAN upsampling artifacts vanish with diffusion models and post-processing               |
| Durall et al. [5]        | 2020 | Spectral and RGB dual-branch CNN                                                  | GAN images (multiple generators)                         | ACC            | Improved cross-generator ACC over RGB-only baselines                                                          | Spectral cues are generator-specific and brittle to recompression                       |
| Gragnaniello et al. [17] | 2021 | ResNet variants with spatial pyramid and no down-sampling in the first layer      | GAN imagery (ProGAN etc.)                                | ACC, AUC       | Critical analysis showing detection is easy on seen GANs and drops off-distribution                           | Confirms poor generalization of GAN-era detectors                                       |
| Chen et al. [18]         | 2022 | SLADD: self-adversarial learning that synthesizes augmented fakes during training | Faces (FFHQ and StyleGAN family)                         | ACC, AUC, AP   | Improved cross-manipulation generalization on face benchmarks                                                 | Face-centric with an adversarial training loop; diffusion generators untested           |
| Ricker et al. [6]        | 2024 | Re-training state-of-the-art GAN detectors on diffusion (ADM) data                | ADM-generated LSUN; unseen DMs                           | AUC, Pd@1% FAR | >90% Pd@1% FAR on unseen DMs after retraining while GAN detectors fail on DM images                           | Only one diffusion family (ADM) at training time and scene-limited                      |
| Corvi et al. [7]         | 2022 | Fingerprint-aware detectors on diffusion images                                   | ProGAN, StyleGAN2/3, ADM, SD                             | ACC, AUC       | For example Grag2021 reaches 99.9 ACC and 100 AUC on ProGAN but 63.3 ACC and 94.8 AUC on compressed StyleGAN2 | Shows diffusion detectors are not yet robust at small scale                             |
| Rombach et al. [1]       | 2022 | Latent diffusion models (background)                                              | LAION                                                    | FID            | SOTA synthesis quality; LDM = dominant modern generator                                                       | Not a detector but motivates the threat model                                           |
| Radford et al. [10]      | 2021 | CLIP vision-language pretraining (background)                                     | WebImageText with ImageNet zero-shot                     | ACC            | 76.2% zero-shot ImageNet top-1                                                                                | Not a detector but the foundation for later detectors                                   |
| Sha et al. [8]           | 2022 | DE-FAKE: CLIP and SimCLR features with SVM for detection and attribution          | COCO captions; SD, LDM, GLIDE, DALL-E 2                  | ACC            | Image-only SD-trained reaches 83.4% ACC on LDM and 55.4% on DALL-E 2 while hybrid reaches 88.5 to 94.5%       | Pre-commercial generators only and modest baselines                                     |
| Wang et al. [19]         | 2023 | DIRE: DDIM inversion and reconstruction error map with MLP                        | Diffusion- ewline Forensics (ADM and SD-v1 among others) | ACC, AP        | 99.9% ACC and 100% AP avg on unseen diffusion models                                                          | Requires two diffusion forward passes per image (slow) and is diffusion-only            |
| Ojha et al. [9]          | 2023 | UniFD: frozen CLIP feature space with nearest neighbor or linear probing          | ProGAN train; 11-generator testbed                       | AP, ACC        | +19.49 mAP and +23.39% ACC over CNNSpot on unseen diffusion and autoregressive generators                     | Frozen features may miss low-level artifacts and gains depend on the CLIP domain        |
| Corvi et al. [15]        | 2023 | Systematic study of GAN vs. diffusion fingerprints                                | GAN + DM imagery, multiple datasets                      | ACC, AUC       | GAN detectors perform poorly on DM images; periodic spectral patterns in both families                        | Diagnostic study that proposes no new strong detector                                   |
| Laszkiewicz et al. [11]  | 2024 | FLIPAD: final-layer inversion and anomaly detection for single-model attribution  | GAN models on CelebA/LSUN                                | ACC            | 97.7 to 99.7% attribution accuracy across four GAN families                                                   | Requires white-box model access and is GAN-only and binary (target vs. non-target)      |
| Keita et al. [20]        | 2024 | Bi-LORA: VLM (LLaVA-style) with LoRA adapters for synthetic image detection       | SD, GAN, unseen diffusion                                | ACC            | 93.41% avg ACC on unseen generation models                                                                    | Heavy compute and prompt sensitivity; limited attribution detail                        |
| Yao & Juarez [12]        | 2025 | SPRINT: secret pixel reconstruction fingerprints with a verifier network          | GAN and diffusion models (12-model pool)                 | ACC, FPR       | 99.17% closed-world attribution ACC with open-world FPR of 2.2% at the 95% TPR operating point                | Requires fingerprinting at model-distribution time; threat model differs from detection |
| Roy et al. [13]          | 2026 | MS-COCOAI benchmark release: 96k real/synthetic images, 5 generators, 2 tasks     | MS-COCO with SD3, SD2.1, SDXL, DALL-E 3, and MJ v6       | ACC            | Frequency-domain ResNet-50 baseline: 0.80 detection ACC, 0.45 attribution ACC                                 | Newly released and few independent detector evaluations published yet                   |

*3) Theme 3: Foundation-model-based detectors.:* The third family bet on transfer from large pretrained vision and vision-language encoders. DE-FAKE [8] pioneered the idea on text-to-image data with CLIP and SimCLR features plus shallow classifiers and achieved 0.885 to 0.945 hybrid detection accuracy across LDM, GLIDE, and DALL-E 2. It also proposed one of the first multi-generator attribution setups (classifying the source model among SD, LDM, GLIDE, and DALL-E 2). Ojha et al. [9] systematized the insight. The *feature space* of a frozen CLIP, used with nearest-neighbor or linear probing, generalizes to unseen generators far better than trained forensic features, gaining +19.49 mAP over CNNSpot on diffusion and autoregressive testbeds. More recent VLM-based approaches such as Bi-LORA [20] push further with language-model adapters and reach 93.41% average ACC on unseen models. This family’s strengths are generality and training efficiency. Its weaknesses are dependence on the encoder’s domain coverage and potential insensitivity to subtle low-level artifacts that a forensic CNN would catch.

*4) Theme 4: Generator attribution.:* Attribution, that is answering *which model* rather than *real or fake*, is far less mature. FLIPAD [11] inverts the final layer of a white-box GAN and applies anomaly detection. It reaches 97.7 to 99.7% attribution accuracy across the four GAN families but requires model parameter access and evaluates only on GANs. SPRINT [12] plants secret reconstruction targets at training time and attains 99.17% closed-world attribution accuracy with an open-world false positive rate of 2.2% at the 95% TPR operating point. However, it presumes cooperation of the model provider. This assumption fails for closed-source services such as DALL-E 3 and MidJourney v6. DE-FAKE [8] offers the most practically relevant prior attribution protocol (supervised multi-class classification of generator identity) but on pre-2023 generators. The dataset paper itself reports a frequency-domain ResNet-50 baseline on MS-COCOAI that scores 0.80 on binary detection yet only 0.45 on five-way attribution [13]. This gap confirms that attribution is the harder of the two tasks on modern generators. To our knowledge no published work performs multi-class attribution across the five modern generators covered by MS-COCOAI [13], including the closed-source ones, under a protocol shared with binary detection.

*5) Synthesis and gap.:* Across the four themes three consistent findings emerge. First, artifact-coupled detectors do not survive generational shifts in synthesis technology. Second, foundation-model feature spaces currently offer the best cross-generator generalization. Third, detection and attribution have been studied separately on disjoint datasets, with attribution lagging badly, especially for closed-source and diffusion-era generators. The MS-COCOAI benchmark [13] creates for the first time a controlled setting where both tasks can be evaluated jointly across five modern generators. Our project occupies exactly this gap. It delivers a unified evaluation of artifact-driven CNN detectors and foundation-model detectors for detection together with supervised multi-class attribution over five generators on a single benchmark.

## III. METHODOLOGY AND EXPERIMENTAL SETUP

### A. Proposed Methodology

The proposed methodology is designed to evaluate whether conventional artifact-driven convolutional neural networks (CNNs) and foundation-model representations behave differently on modern diffusion-generated imagery. The experimental pipeline therefore consists of two complementary CNN detectors, a frozen vision-language foundation model, and a unified multi-task architecture. ResNet-50 is retained as the primary CNN baseline because the MS-COCOAI dataset paper reports a frequency-domain ResNet-50 baseline of 0.80 binary detection accuracy and 0.45 attribution accuracy [13]. EfficientNet-B0 is introduced as a compute-efficient second CNN architecture, while CLIP ViT-B/16 is used as the foundation-model-based detector in a frozen-feature linear-probing setting motivated by UniFD [9].

All models are evaluated on the same official MS-COCOAI train, validation, and test splits. The experimental sequence is: (1) prepare and normalize the images; (2) fine-tune the CNN backbones for binary detection and attribution; (3) extract frozen CLIP image embeddings and train lightweight classifiers; (4) train a shared-backbone multi-task model with separate detection and attribution heads; and (5) compare the resulting models using common metrics and per-generator analyses. This design isolates the effect of representation family while keeping the dataset protocol fixed.

The primary binary task is to predict whether an input image is real or AI-generated. The attribution task predicts one of six provenance classes, namely real or one of the five generators. The final analysis reports both aggregate performance and generator-specific behavior so that errors are not hidden by a single overall score.

### B. Model Architecture

*1) CNN-based detectors:* Two ImageNet-pretrained CNN backbones are used. The first is ResNet-50, which employs residual blocks to facilitate the optimization of a deep convolutional network [21]. The second is EfficientNet-B0, which uses compound scaling to obtain a substantially smaller model while maintaining competitive visual representation capacity [22]. For each backbone, the original ImageNet classification layer is replaced with a task-specific fully connected head.

For binary detection, the model produces a single logit  $z_d$  and the probability of an AI-generated image is

$$p_d = \sigma(z_d), \quad (1)$$

where  $\sigma(\cdot)$  denotes the sigmoid function. An image is classified as AI-generated when  $p_d \geq 0.5$ .

For generator attribution, the same backbone is followed by a six-class linear classification head. The output logits are converted into a probability distribution with the softmax function,

$$p_{a,k} = \frac{\exp(z_{a,k})}{\sum_{j=0}^5 \exp(z_{a,j})}, \quad (2)$$

where  $k \in \{0, 1, \dots, 5\}$  denotes the attribution class. Class 0 represents real MS-COCO imagery, while classes 1–5 represent the five generators specified by the MS-COCOAI label definition [13].

2) *Foundation-model detector*: The foundation-model branch uses the image encoder of CLIP with a ViT-B/16 visual backbone. CLIP learns transferable image representations jointly with natural-language supervision [10], and prior synthetic-image detection work has shown that a frozen CLIP feature space can generalize across generators when paired with a lightweight classifier [9]. To keep computation manageable, the CLIP encoder is frozen during training and only a linear classification layer is optimized.

Given an input image  $x$ , the frozen encoder produces an image embedding  $h = f_{\text{CLIP}}(x)$ . The binary detector then computes  $z_d = w_d^T h + b_d$ , while the attribution detector computes  $z_a = W_a h + b_a$ . Thus, no gradient updates are applied to the CLIP backbone. The same official data splits and labels used by the CNN experiments are used for the linear probes, allowing a direct comparison between learned forensic features and transferable foundation-model features.

3) *Unified multi-task model*: The final architecture investigates whether detection and attribution can benefit from a shared visual representation. EfficientNet-B0 is selected as the shared backbone because it provides a practical compromise between representational capacity and training cost. The backbone feature vector is passed to two task-specific heads,

$$h = f_{\text{EffB0}}(x), \quad z_d = g_d(h), \quad z_a = g_a(h), \quad (3)$$

where  $g_d$  is the binary detection head and  $g_a$  is the six-class attribution head. The joint objective is

$$\mathcal{L}_{\text{MT}} = \lambda_d \mathcal{L}_{\text{BCE}} + \lambda_a \mathcal{L}_{\text{CE}}, \quad (4)$$

where  $\mathcal{L}_{\text{BCE}}$  is binary cross-entropy,  $\mathcal{L}_{\text{CE}}$  is six-class cross-entropy, and the initial task weights are set to  $\lambda_d = 1$  and  $\lambda_a = 1$ . Equal weighting provides a neutral starting point; any alternative weighting is treated as a separately reported ablation rather than silently changing the objective.

TABLE II. SUMMARY OF THE PROPOSED MODEL CONFIGURATIONS.

| Model           | Backbone                            | Training                      | Outputs          |
|-----------------|-------------------------------------|-------------------------------|------------------|
| ResNet-50       | CNN, pretrained                     | End-to-end fine-tuning        | Binary / 6-class |
| EfficientNet-B0 | CNN, pretrained                     | End-to-end fine-tuning        | Binary / 6-class |
| CLIP-B/16       | ViT- Vision Transformer, pretrained | Frozen encoder + linear probe | Binary / 6-class |
| Multi-task      | EfficientNet-B0, pretrained         | Shared backbone + two heads   | Binary + 6-class |

### C. Dataset and Preprocessing

The experiments use the MS-COCOAI dataset introduced in the dataset description above [13]. The benchmark contains 96,000 images distributed over official train, validation, and test splits of 42,000, 9,000, and 45,000 images, respectively. Synthetic samples are generated using Stable Diffusion 3, Stable Diffusion 2.1, SDXL, DALL-E 3, and MidJourney v6

from MS-COCO captions, while real samples are original MS-COCO photographs. Because the real and synthetic samples are caption-aligned, the benchmark reduces semantic-content confounding and supports both detection and attribution under a common protocol.

No re-splitting of the benchmark is performed. The official training split is used for optimization, the validation split is used for model selection and early stopping, and the test split is accessed only for the final evaluation. This prevents information from the test set from influencing training decisions. The test split remains untouched until the model and hyperparameter configuration have been fixed.

All images are converted to three-channel RGB and resized to  $224 \times 224$  pixels. A fixed input size is used to keep the CNN and CLIP experiments computationally comparable and to control the effect of generator-dependent output resolutions identified as a possible shortcut in the dataset description. For the CNN models, images are normalized using the standard ImageNet mean and standard deviation associated with the pretrained backbones. The CLIP branch uses the normalization specified by the pretrained CLIP image encoder. Apart from model-specific normalization, the geometric preprocessing is kept consistent across experiments.

During CNN training, augmentation is applied only to the training split. The primary augmentation pipeline consists of random resized cropping to  $224 \times 224$ , random horizontal flipping with probability 0.5, moderate color jitter, and occasional Gaussian blur. Random JPEG compression is additionally included to reduce reliance on fragile encoding artifacts and to improve tolerance to common image transformations, following the motivation of augmentation-based synthetic-image detection [3]. Validation and test images receive deterministic resizing and center cropping without stochastic augmentation.

For attribution, class frequencies are computed from the training split and class-weighted cross-entropy is used for Label\_B to reduce the effect of the real-class imbalance noted in the dataset description. The attribution weight for class  $k$  is computed as  $w_k = N/(Kn_k)$ , where  $N$  is the number of training samples,  $K = 6$ , and  $n_k$  is the number of samples in class  $k$ . Binary detection is evaluated with the natural Label\_A distribution; no test-time rebalancing is performed. The same augmentation policy and official splits are maintained across CNN experiments to ensure that observed differences primarily reflect architecture rather than data handling.

### D. Training Configuration

The CNN models are initialized from ImageNet-pretrained weights and fine-tuned end-to-end. A batch size of 32 images is used with mixed-precision training and gradient accumulation over two steps, giving an effective batch size of 64. AdamW is used as the optimizer with an initial learning rate of  $10^{-4}$  and weight decay of  $10^{-4}$ . The learning rate follows cosine decay after a one-epoch linear warm-up. Training is limited to 30 epochs, with early stopping based on validation loss using a patience of five epochs. The best validation checkpoint is restored before test evaluation.

The CLIP ViT-B/16 image encoder remains frozen. Its image embeddings are computed from the training, validation,

and test images and the lightweight binary or attribution classifier is trained using AdamW with a learning rate of  $10^{-3}$ , weight decay of  $10^{-4}$ , and a maximum of 15 epochs. The validation loss is monitored with an early-stopping patience of three epochs. Because only the classification head is updated, this branch substantially reduces backpropagation memory and training cost while retaining the pretrained representation.

The unified multi-task EfficientNet-B0 follows the CNN configuration, using an effective batch size of 64, AdamW, an initial learning rate of  $10^{-4}$ , weight decay of  $10^{-4}$ , cosine learning-rate decay, and a 30-epoch maximum with four-epoch early stopping. The two task losses are initially weighted equally. Checkpoints are selected using the sum of the normalized validation losses for the detection and attribution heads. All experiments use a fixed random seed for data-loader shuffling, augmentation sampling, and model initialization where applicable, and the exact software versions and hardware used for the final runs are recorded with the experiment logs.

TABLE III. TRAINING CONFIGURATION USED FOR THE PRINCIPAL EXPERIMENTS.

| Parameter             | ResNet-50 / EfficientNet-B0 | CLIP B/16              | ViT- | Multi-task EfficientNet-B0 |
|-----------------------|-----------------------------|------------------------|------|----------------------------|
| Input size            | 224 × 224<br>RGB            | 224 × 224<br>RGB       |      | 224 × 224<br>RGB           |
| Batch size            | 32, effective<br>64         | 64                     |      | 32, effective<br>64        |
| Optimizer             | AdamW                       | AdamW                  |      | AdamW                      |
| Initial learning rate | $10^{-4}$                   | $10^{-3}$              |      | $10^{-4}$                  |
| Weight decay          | $10^{-4}$                   | $10^{-4}$              |      | $10^{-4}$                  |
| Maximum epochs        | 30                          | 15                     |      | 30                         |
| Schedule              | Warm-up + cosine decay      | Warm-up + cosine decay |      | Warm-up + cosine decay     |
| Early stopping        | Patience 5                  | Patience 3             |      | Patience 5                 |
| Precision             | Mixed precision             | Mixed precision        |      | Mixed precision            |
| Backbone training     | Fine-tuned                  | Frozen                 |      | Fine-tuned                 |

### E. Evaluation Metrics

Binary detection is evaluated using Accuracy, F1 score, and ROC Area Under the Curve (AUC), which are complementary measures for the balanced real-versus-fake decision and are already defined in the literature review. Accuracy measures the fraction of correctly classified images, F1 summarizes the balance between precision and recall, and AUC evaluates ranking quality across all classification thresholds. Precision and recall are additionally reported to make the false-positive and false-negative behavior explicit.

For generator attribution, overall Accuracy and macro-F1 are reported. Macro-F1 gives equal weight to each of the six attribution classes and is therefore informative when class frequencies are unequal. A  $6 \times 6$  confusion matrix is generated for each attribution model to identify systematic confusion between generators. Per-generator detection results are also reported by restricting the fake subset to each generator in turn.

This analysis directly supports the objective of characterizing generator-specific detection difficulty without collapsing all synthetic sources into a single aggregate score.

To assess robustness, the final selected models are additionally evaluated under controlled image degradations, including moderate JPEG recompression and Gaussian blur. These tests are performed only at evaluation time so that the robustness result measures sensitivity to post-processing rather than augmentations seen during training. Representative true-positive and false-positive examples are visualized for qualitative error analysis. Together, the quantitative and qualitative evaluations address both overall performance and the failure modes that may reveal shortcut learning or generator-specific artifacts.

The principal comparison therefore consists of four representation families/configurations: ResNet-50, EfficientNet-B0, frozen CLIP ViT-B/16, and the unified multi-task EfficientNet-B0. All models use the same official splits, common input resolution, and identical evaluation protocol. This controlled design permits comparison of artifact-driven CNN features, an efficient CNN representation, transferable foundation-model features, and a shared multi-task representation within the same MS-COCOAI benchmark.

## REFERENCES

- [1] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022, arXiv:2112.10752.
- [2] H. Farid, “Creating, using, misusing, and detecting deep fakes,” *Journal of Online Trust and Safety*, vol. 1, no. 4, 2022.
- [3] S.-Y. Wang, O. Wang, R. Zhang, A. Owens, and A. A. Efros, “CNN-generated images are surprisingly easy to spot... for now,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020, arXiv:1912.11035.
- [4] J. Frank, T. Eisenhofer, L. Schönherr, A. Fischer, D. Kolossa, and T. Holz, “Leveraging frequency analysis for deep fake image recognition,” in *Proceedings of the 37th International Conference on Machine Learning (ICML)*, 2020, arXiv:2003.08685.
- [5] R. Durall, M. Keuper, and J. Keuper, “Watch your up-convolution: CNN based generative deep neural networks are failing to reproduce spectral distributions,” in *International Conference on Pattern Recognition (ICPR)*, 2020, arXiv:2003.01826.
- [6] J. Ricker, S. Damm, T. Holz, and A. Fischer, “Towards the detection of diffusion model deepfakes,” in *International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISAPP)*, 2024, arXiv:2210.14571.
- [7] R. Corvi, D. Cozzolino, G. Zingarini, G. Poggi, K. Nagano, and L. Verdoliva, “On the detection of synthetic images generated by diffusion models,” in *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2023, arXiv:2211.00680.
- [8] Z. Sha, Z. Li, N. Yu, and Y. Zhang, “DE-FAKE: Detection and attribution of fake images generated by text-to-image generation models,” *arXiv:2210.06998*, 2022.
- [9] U. Ojha, Y. Li, and Y. J. Lee, “Towards universal fake image detectors that generalize across generative models,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023, arXiv:2302.10174.
- [10] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” in *Proceedings of the 38th International Conference on Machine Learning (ICML)*, 2021, arXiv:2103.00020.
- [11] M. Laszkiewicz, J. Ricker, J. Lederer, and A. Fischer, “Single-model attribution of generative models through final-layer inversion,” in *International Conference on Machine Learning (ICML)*, 2024, arXiv:2306.06210.

- [12] K. Yao and M. Juarez, “SPRINT: Robust model attribution of generated images via secret pixel reconstruction,” *arXiv:2508.05691*, 2025.
- [13] R. Roy, A. Aziz, S. Bajpai, N. Imanpour, G. Singh, S. Biswas, K. Wanaskar, P. Patwa, S. Ghosh, S. Dixit, N. R. Pal, V. Rawte, R. Garimella, A. Das, A. Sheth, G. Jena, V. Sharma, A. N. Reganti, V. Jain, and A. Chadha, “A comprehensive dataset for human vs. AI generated image detection,” *arXiv:2601.00553*, 2026.
- [14] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick, “Microsoft COCO: Common objects in context,” in *European Conference on Computer Vision (ECCV)*, 2014.
- [15] R. Corvi, D. Cozzolino, G. Poggi, K. Nagano, and L. Verdoliva, “Intriguing properties of synthetic images: From generative adversarial networks to diffusion models,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2023, arXiv:2304.06408.
- [16] S. J. Nightingale and H. Farid, “AI-synthesized faces are indistinguishable from real faces and more trustworthy,” *Proceedings of the National Academy of Sciences*, vol. 119, no. 8, p. e2120481119, 2022.
- [17] D. Gragnaniello, D. Cozzolino, F. Marra, G. Poggi, and L. Verdoliva, “Are GAN generated images easy to detect? A critical analysis of the state-of-the-art,” in *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2021, arXiv:2104.02617.
- [18] L. Chen, Y. Zhang, Y. Song, L. Liu, and J. Wang, “Self-supervised learning of adversarial example: Towards good generalizations for deepfake detection,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022, arXiv:2203.12208.
- [19] Z. Wang, J. Bao, W. Zhou, W. Wang, H. Hu, H. Chen, and H. Li, “DIRE for diffusion-generated image detection,” in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023, arXiv:2303.09295.
- [20] M. Keita, W. Hamidouche, H. Bougueffa Eutamene, A. Hadid, and A. Taleb-Ahmed, “Bi-LORA: A vision-language approach for synthetic image detection,” *arXiv:2404.01959*, 2024.
- [21] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016.
- [22] M. Tan and Q. V. Le, “Efficientnet: Rethinking model scaling for convolutional neural networks,” in *Proceedings of the 36th International Conference on Machine Learning (ICML)*, 2019, arXiv:1905.11946.