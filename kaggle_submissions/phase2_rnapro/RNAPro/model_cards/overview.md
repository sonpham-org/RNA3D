# Model Overview

## Description:
RNAPro is a model for predicting RNA 3D structure from sequence, combining AF3-like co-folding architectures with RNA foundation models, MSAs, and template-based modeling. It tackles the challenge of accurate RNA structure prediction by providing a computational alternative to expensive, time-consuming wet-lab structure determination, thereby supporting structure-driven drug discovery. The primary users are computational biologists, drug discovery researchers, and developers working on representation learning and generative AI for biological data. The model integrates into drug discovery pipelines by reducing wet-lab burden for target identification, validation, and downstream RNA generative modeling. This model is ready for commercial and non-commercial use.<br>



## License/Terms of Use

Governing Terms: Use of this model is governed by the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

## Deployment Geography:
Global <br>

## Use Case: <br>

- RNAPro can be used by RNA therapeutics developers for understanding RNA function and designing RNA-based therapeutics. <br>

## Release Date:  <br>
GitHub 01/09/2026 via https://github.com/NVIDIA-Digital-Bio/RNAPro <br>
Hugging Face 01/09/2026 via:
- https://huggingface.co/nvidia/RNAPro-Private-Best-500M
- https://huggingface.co/nvidia/RNAPro-Public-Best-500M
<br>

NGC 01/09/2026 via https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/resources/rnapro <br>



## Model Architecture:
RNAPro consists of an input embedder, an MSA module, a gating module, and a template module that process the input sequence, MSA, RNA foundation model features, and templates. The Pairformer block is used to update the single and pair representations. Finally, a diffusion module takes these updated single and pair representations to predict the 3D structure.

Number of model parameters: 488,301,921
<br>

## Input: <br>
**Input Type(s):** Text (RNA sequence, MSA), Binary (templates)  <br>
**Input Format:** 
- Text: CSV (RNA sequence), FASTA (MSA)
- Binary: Templates

**Input Parameters:** <br>
- Text: 1D 
- Binary: 3D

**Other Properties Related to Input:** RNA sequence, MSA, and templates are automatically cropped to the 512 length.


## Output: <br>
**Output Type(s):** RNA 3D structure coordinates <br>
**Output Format:** CIF <br>
**Output Parameters:** 3D <br>
**Other Properties Related to Output:** CIF files including all atom structures will be saved.

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions. <br>

## Software Integration:
**Runtime Engine(s):**
* PyTorch


**Supported Hardware Microarchitecture Compatibility:** <br>
* NVIDIA Ampere <br>
* NVIDIA Hopper
* NVIDIA Blackwell


**Preferred/Supported Operating System(s):**
* Linux <br>


The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment. <br>


## Model Version(s):
We are releasing two best models — one each for the private and public target datasets — based on the leaderboard test datasets from the [Stanford RNA 3D Folding Kaggle Competition.
](https://www.kaggle.com/competitions/stanford-rna-3d-folding)

- RNAPro-Private-Best-500M <br>
- RNAPro-Public-Best-500M <br>

## Training, Testing, and Evaluation Datasets:

### Training Datasets:

(1) Stanford RNA 3D Folding Kaggle dataset **Link:** [Stanford RNA 3D Folding
](https://www.kaggle.com/competitions/stanford-rna-3d-folding)   <br>

**Data Modality** <br>
* Text (RNA sequence, MSA, structures) <br>

**Properties:** The Stanford RNA 3D Folding dataset contains 5,135 RNA sequences and structure labels. MSA files are included for some RNA sequences. Training labels include only alpha-carbon structures. <br>

**Non-Audio, Image, Text Training Data Size:** The dataset contains corresponding CSVs, FASTAs, and CIFs totaling 65.1GB.

**Data Collection Method:**
* Human <br>

**Labeling Method by Dataset:**
* Human <br>


(2) Stanford RNA 3D Folding all-atom **Link:** [Stanford RNA 3D Folding all atom
](https://www.kaggle.com/datasets/rhijudas/stanford-rna-3d-folding-all-atom-train-data)   <br>

**Data Modality** <br>
* Text (RNA sequence, MSA, structures) <br>

**Properties:** Stanford RNA 3D Folding all atom dataset contains 5,135 RNA sequences and structures. Training labels include all-atom structures.  <br>

**Non-Audio, Image, Text Training Data Size:** The dataset contains corresponding CSVs, totaling 108.49 GB.

**Data Collection Method:**
* Human <br>

**Labeling Method by Dataset:**
* Human  <br>

(3) Protenix dataset **Link:** [Protenix dataset
](https://github.com/bytedance/Protenix/blob/main/docs/training.md)   <br>

**Data Modality** <br>
* Text (biomolecular CIF files, JSON files, MSA files) <br>

**Properties:** This dataset includes biological sequences, molecules, and structure files to train a biomolecule structure prediction model called Protenix. We used the files `components.v20240608.cif` and `components.v20240608.cif.rdkit_mol.pkl` for this project. <br>

**Non-Audio, Image, Text Training Data Size:** The dataset contains pre-processed files, MSAs, structure files, totaling 1TB.

**Data Collection Method:**
* Human <br>

**Labeling Method by Dataset:**
* Human <br>


### Evaluation Datasets:

Stanford RNA 3D Folding private dataset **Link:** [Stanford RNA 3D Folding
](https://www.kaggle.com/competitions/stanford-rna-3d-folding)   <br>

**Data Modality** <br>
* Text (RNA sequence, MSA, structures) <br>

**Properties:** The Stanford RNA 3D Folding private dataset contains recently synthesized RNA sequences and structure labels. <br>

**Non-Audio, Image, Text Training Data Size:** The dataset contains corresponding CSVs and CIFs.

**Data Collection Method:**
* Human <br>

**Labeling Method by Dataset:**
* Human <br>

## Inference:
**Acceleration Engine:** cuEquivariance <br>
**Test Hardware:** A100, H100, GB300 <br>

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

For more detailed information on ethical considerations for this model, please see the Model Card++ Bias, Explainability, Safety & Security, and Privacy Subcards.

Users are responsible for ensuring the physical properties of model-generated molecules are appropriately evaluated and comply with applicable safety regulations and ethical standards. <br>

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).  <br>
