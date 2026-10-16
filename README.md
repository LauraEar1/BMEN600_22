# The more the merrier: Does the inclusion of additional physiological measures of sleep quality help or harm accuracy of EEG-based automated sleep-pathology classification? 

 Cyclic alternating pattern (CAP) is a periodic EEG activity that occurs during sleep. Abnormal CAP rates have been used in sleep science as a measure of instable sleep and linked to a number of sleep disorders. The measurement of CAP rate historically has been performed by trained clinicians who review patients EEG data in 30 second epochs and systematically score each section as N1, N2, N3 REM or wake stages. The time spent in these stages is then used to calculate the CAP rate. The analysis of this overnight polysomnography (PSG) record is tedious and time consuming. AI models have been utilized to automate the CAP-scoring process however struggle to produce reliable and clinically acceptable levels of accuracy in their sleep pathology predictions. 

 Our project aims to investigate the effect of additional physiological data, often simultaneously recorded in PSG sessions, on the classification accuracy and reliability of EEG-based models. While more data points are typically beneficial, the noisy nature of EMG, EOG and ECG recordings introduce uncertainty to the value of including these measures. While they have the potential to improve the paradigm, they also introduce the need for additional computational power, preprocessing and training. We aim to see if the potential benefits of improved classifier accuracy and user trust associated with accurate classifications outweigh the computational efforts of including these measures. 

## Project Description
This project will be a comparative analysis of three classifiers trained on PSG recordings from the CAP Sleep Database with varying levels of detail extracted from the recordings. These classifiers will attempt to classify PSG recordings as belonging to a participant who was a healthy control, had nocturnal frontal lobe epilepsy (NFLE), REM sleep behavior disorder (RBD), or periodic leg movements(PLM). 
* Classifier 1 will be trained solely on EEG data and CAP annotations.
* Classifier 2 will be trained on EEG data, CAP annotations, EOG, and EMG.
* Classifier 3 will be trained on EEG, CAP annotations, EOG, EMG, ECG and cardiovascular events.

## Research Question

To what extent do various physiological measures collected during PSG influence the efficacy of sleep pathology detection by AI-Classifier models as compared to AIs performing isolated CAP-score analysis.

### Bigest Uncertainty
Our biggest uncertainty is in the quality of all physiological recordings through the dataset. Representative sample of all groups have been tested and passed quality checks with varying degrees. Initial visualization indicates that comprehensive preprocessing will be nessecary. 

### Biomedical Problem
Our investigation and subsequent software development aims to address clinician distrust towards automated sleep pathology diagnosis. We will do this by creating an AI model that can accurately predict sleep pathologies at a clinically acceptable level, improving on current EEG- and CAP-Rate based models. As AI classification accuracy and clinician trust increase our software can begin to replace clinicians in review of PSG recordings, reducing clinician workload significantly and freeing up time for patient interaction and other daily tasks. 

## Dataset
 [CAP Sleep Database](https://physionet.org/content/capslpdb/1.0.0/)

Credit to: B Kemp, AH Zwinderman, B Tuk, HAC Kamphuisen, JJL Oberyé. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. IEEE-BME 47(9):1185-1194 (2000).

## Team Contributions & Licensing
This project was created in a collaborative work environment in order to fulfil the course requirements of BMEN 600 at the University of Calgary. Project formation and software development was complete by team members Laura Earl, Jawad Darwish, Anushka Basu, Selinam Fiadzo. 
