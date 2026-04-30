# Model Overview

## Model NSpect ID
NSPECT-LEW5-UKZO
 
### Description:
GENMO is a generalist model that unifies human motion estimation and generation into a single framework. Developed by NVIDIA, GENMO was created to bridge the gap between motion estimation and generation tasks, which were traditionally handled by separate, specialized models. This unified framework allows GENMO to effectively leverage shared representations, leading to synergistic benefits, and enabling motion generation or estimation from multiple modalities. 

This model is ready for non-commercial research use.  <br>

### License/Terms of Use

[Visit the NVIDIA Legal Release Process](https://nvidia.sharepoint.com/sites/ProductLegalSupport) for instructions on getting legal support for a license selection: <br> 

-If you are releasing under an open source license (such as Apache 2.0, MIT), contact the [Open Source Review Board](https://confluence.nvidia.com/pages/viewpage.action?pageId=800720661) (formerly SWIPAT) by filing a contribution bug request [here](https://nvbugspro.nvidia.com/bug/2885991). <br> 

-If your release is for non-commercial or research purposes only, file a new bug [here](https://nvbugspro.nvidia.com/bug/3508089). <br> 
https://nvbugspro.nvidia.com/bug/5524776

-If your release allows for commercial purposes, submit [Product Legal Support Form](https://forms.office.com/pages/responsepage.aspx?id=FT0IQ3NywUC32znv2czBejILt4CYhTJKv0O6I4gccylUMVlMSE4xSFhYMUYyT1VMNVNCREk4RlE1NS4u&route=shorturl). <br> 

### Deployment Geography:
Global <br>

### Use Case: <br>
GENMO would be used by individuals and professionals in fields such as machine learning and computer vision research, gaming, animation, and 3D content creation. These users require a system that provides precise and intuitive control over motion sequences.

The model's capabilities allow users to:

Generate complex motion sequences by integrating multiple modalities. For example, a user could start a motion from a video clip, transition to a movement described by text, synchronize it with audio cues, and even align it with another video.

Create realistic human animations for creative applications like gaming and film. This includes generating lifelike character animations for game designers and creating immersive experiences for virtual reality developers.

Reconstruct accurate motion trajectories from a variety of observations, such as videos and 2D keypoints. This is useful for tasks like recovering global human motion from videos with dynamic cameras.

Perform motion in-betweening, where the model generates the motion between specified keyframes, which is a valuable tool for animators.

Conduct biomechanical analyses to study human movement, which has applications in sports, rehabilitation, and medicine. For instance, a sports coach could use the model to analyze an athlete's movements, or a physical therapist could use it to evaluate a patient's progress after an injury. <br>

### Release Date:  <br>
Github [Insert MM/DD/YYYY] via [URL] <br> 

## References(s):
Project Page: https://research.nvidia.com/labs/dair/genmo/ 

Paper: https://arxiv.org/pdf/2505.01425

## Model Architecture:
**Architecture Type:** Transformer <br>

**Network Architecture:** Name Other Not Listed <br>

** Number of model parameters [(Recorded with at least two significant figures e.g. 7.3*10^10)] <br>
5.2*10^8

(Internal Only for GPAI Models: Not To Be Published) <br> 
** Describe design choices related to initialization techniques, hyperparameter tuning, regularization techniques, model optimization, damping, and training parameters. [CAN BE EXTRACTED FROM MODEL CARD GENERATOR- 100 Words or Less]  <br> 

## Computational Load (Internal Only: For NVIDIA Models Only) 
**Cumulative Compute:** [Follow Instructions](https://nvidia.sharepoint.com/:w:/s/TrustworthyAI/EbsO-zwZrZlLiavYNk3zVRABuonEHXLJjpzl0jWx5u6S1A?e=5Rxgqw) <br>
**Estimated Energy and Emissions for Model Training:** <br> [Follow Instructions](https://nvidia.sharepoint.com/:w:/s/TrustworthyAI/Ecb23D6b-kNLv1c6LCdhSiMBjWFz0MKncBaRcryDsfv-vA?e=lIcO9z) <br>


## Input: <br>
**Input Type(s):** Audio, Video, Text, Person Bounding Box, 2D Pose, 3D Keyframes <br>
**Input Format:** Red, Green, Blue (RGB), String, Audio <br>
**Input Parameters:** [(1D, 2D, 3D, etc.) Please write out in first mention of instance] <br>
**Other Properties Related to Input:** [Specific Resolution/Minimum or Maximum Resolution or Characters (Including Restrictions), Image Range Needed (W x Y x Z), Pre-Processing Needed, Alpha Channel, Bit, Please State Explicity;] 
(FOR GPAI Models Only): State size and length limits for each input modality.] <br>

## Output: <br>
**Output Type(s):** Human body motion in SMPL or NVHuman format. <br>
**Output Format:** array of floats <br>
**Output Parameters:**  1D <br>
**Other Properties Related to Output:** NA <br>

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions. <br> 

## Software Integration:
**Runtime Engine(s):** 
* [Not Applicable (N/A)- Name Platform If Multiple] <br> 

**Supported Hardware Microarchitecture Compatibility [List in Alphabetic Order]:** <br>
* [NVIDIA Ampere or specific models] <br>
* [NVIDIA Blackwell or specific models] <br>
* [NVIDIA Jetson or specific models]  <br>
* [NVIDIA Hopper or specific models] <br>
* [NVIDIA Lovelace or specific models] <br>
* [NVIDIA Pascal or specific models] <br>
* [NVIDIA Turing or specific models] <br>
* [NVIDIA Volta or specific models] <br>

**Preferred/Supported Operating System(s):**
* [Linux] <br>

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment. <br>

## Model Version(s):
GENMO v1.0  <br>

GENMO can be integrated into an AI system as a specialized module within a larger pipeline, serving as the core engine for all human motion-related tasks. Since it is designed to handle multiple modalities and tasks, it can serve as a central hub for motion generation and estimation.


## Training, Testing, and Evaluation Datasets:

### Dataset Overview [Beginning of GPAI Model Specific Section:]
** Total Size: 2.1*10^9 <br>
** Total Number of Datasets: 10 <br>   

** Dataset partition: Training 99.94%, Validation 0.06% <br> 
** Time period for training data collection: N/A - public datasets <br> 
** Time period for testing data collection: N/A - public datasets <br> 
** Time period for validation data collection: N/A - public <br>

** A general description of the data processing involved in transforming the acquired data into the training data for the model (recommended 200 words) <br>
We used a collection of public datasets for training. All datasets are handled through separate data readers that bring them to a unified structure. 

# Public Datasets
1. AMASS - https://amass.is.tue.mpg.de/
1. BEDLAM - https://bedlam.is.tue.mpg.de/ 
1. Human3.6M - http://vision.imar.ro/human3.6m/description.php 
1. 3DPW - https://virtualhumans.mpi-inf.mpg.de/3DPW/ 
1. AIST++ - https://google.github.io/aistplusplus_dataset/factsfigures.html 
1. HumanML3D - https://github.com/EricGuo5513/HumanML3D 
1. Motion-X - https://github.com/IDEA-Research/Motion-X 
1. RICH - https://rich.is.tue.mpg.de/ 
1. EMDB - https://eth-ait.github.io/emdb/ 
1. Beat2 - https://is.mpg.de/ps/en/projects/beat2-dataset-for-holistic-co-speech-gesture-generation 



*List of 'main/large' datasets (above 3% of the overall data in this category) with unique identification, links + period of collection <br> 
 
### [End of GPAI Model Specific Section] <br>

## Training Dataset [The dataset the model was trained on]:

**Link:** [Link or name to dataset used for training the model.  Share nSpect IDs.  nSpect IDs will be internal-only]  <br>

** Data Modality <br>
* [Audio] <br>
* [Image] <br>
* [Text] <br>
* [Video] <br>
* [Other: Specify] <br>

** Audio Training Data Size (If Applicable) <br>
* [Less than 10,000 Hours] <br>
* [10,000 to 1 Million Hours] <br>
* [More than 1 Million Hours] <br>

** Image Training Data Size (If Applicable) <br>
* [Less than a Million Images] <br>
* [1 Million to 1 Billion Images] <br>
* [More than 1 Billion Images] <br>

** Text Training Data Size (If Applicable) <br>
* [Less than a Billion Tokens] <br>
* [1 Billion to 10 Trillion Tokens] <br>
* [More than 10 Trillion Tokens] <br>

** Video Training Data Size (If Applicable) <br>
* [Less than 10,000 Hours] <br>
* [10,000 to 1 Million Hours] <br>
* [More than 1 Million Hours] <br>

** Non-Audio, Image, Text Training Data Size (If Applicable) <br>
*  Specify the approximate size and unit of measurement <br> 

** Data Collection Method by dataset <br>
* [Hybrid: Human, Automatic] <br>

** Labeling Method by dataset <br>
* [Hybrid: Human, Automated] <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** [Number of data items in training set, descriptive information about the data indicating (i) the modalities (e.g,, text, images), (ii) nature of the content (e.g., personal data, copyright protected content, machine generated data such as Internet of Things or synthetic data) and (iii) its linguistic characteristics. If applicable, what specific sensor type was used for Data Collection] <br>
**Dataset License(s):** [Name, Link applicable to dataset license or applicable Jira ticket or NVBug. Write none if not applicable (N/A). This is an internal-only field.] <br>

### Testing Dataset:

**Link:** [Link or name to dataset used for evaluating the model.  Share Nspect IDs.  Nspect IDs will be internal-only.]  <br>

Data Collection Method by dataset:  <br>
* [Hybrid: Human, Automatic] <br>

Labeling Method by dataset:  <br>
* [Hybrid: Human, Automatic] <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** [Number of data items in training set, descriptive information about the data indicating (i) the modalities (e.g,, text, images), (ii) nature of the content (e.g., personal data, copyright protected content, machine generated data such as Internet of Things or synthetic data) and (iii) its linguistic characteristics. If applicable, what specific sensor type was used for Data Collection] <br>

**Dataset License(s):** [Name, Link applicable to dataset license or applicable Jira ticket or NVBug. Write none if not applicable (N/A).  Jira tikets are not required if Creating a Synthetic Dataset with No Third-Party Tools/Dependencies OR Creating a Real Dataset with No Personal Data and No Confidential/Third-Party Data. This is an internal-only field.] This is an internal-only field.] <br>


### Evaluation Dataset:

**Link:** [Link or name to dataset used for evaluating the model.  Share Nspect IDs.  Nspect IDs will be internal-only.]  <br>

**Benchmark Score <br>

Data Collection Method by dataset:  <br>
* [Hybrid: Human, Automatic] <br>

Labeling Method by dataset:  <br>
* [Hybrid: Human, Automatic] <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** [Number of data items in training set, descriptive information about the data indicating (i) the modalities (e.g,, text, images), (ii) nature of the content (e.g., personal data, copyright protected content, machine generated data such as Internet of Things or synthetic data) and (iii) its linguistic characteristics. If applicable, what specific sensor type was used for Data Collection] <br>

**Dataset License(s):** [Name, Link applicable to dataset license or applicable Jira ticket or NVBug. Write none if not applicable (N/A).  Jira tikets are not required if Creating a Synthetic Dataset with No Third-Party Tools/Dependencies OR Creating a Real Dataset with No Personal Data and No Confidential/Third-Party Data. This is an internal-only field.] <br>

*Insert Quantitative Evaluation Benchmarks Here (Language Model Reference https://docs.google.com/spreadsheets/d/1SwyHhBTFQJTLZ4tuc-JdDcjA7wy68DyWdTehGNUgJ6M/edit?gid=721511647#gid=721511647 for large language models) <br>

# Inference:
**Test Hardware:** <br>  
* NVIDA A100 or newer.  <br>

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br> 

Please make sure you have proper rights and permissions for all input image and video content; if image or video includes people, personal health information, or intellectual property, the image or video generated will not blur or maintain proportions of image subjects included. <br>

Users are responsible for model inputs and outputs. Users are responsible for ensuring safe integration of this model, including implementing guardrails as well as other safety mechanisms, prior to deployment. <br> 

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).  <br>

