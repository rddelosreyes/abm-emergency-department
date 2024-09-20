This code accompanies the following paper:

**Delos Reyes, R., Capurro, D., & Geard, N. (2024). _Modelling patient trajectories in emergency department simulations using retrospective patient cohorts_. Computers in Biology and Medicine. [https://doi.org/10.1016/j.compbiomed.2024.109147](https://doi.org/10.1016/j.compbiomed.2024.109147)**

![graphical-abstract_10-1016-j-compbiomed-2024-109147](https://github.com/user-attachments/assets/96f86c0a-ee55-42d2-b004-b5bb3adf73a2)

---

**Setting up the environment**
1. Download [anaconda](https://docs.anaconda.com/)
2. Run
    `conda env create --name edabm -f edabm.yaml`
3. Run
    `conda activate edabm`

**Getting the required data**
1. Download the following datasets (they require credentialed access which can be requested at the provided websites)
    - [MIMIC-IV v2.2](https://physionet.org/content/mimiciv/2.2/)
    - [MIMIC-IV-ED v2.2](https://physionet.org/content/mimic-iv-ed/2.2/)
2. Store the unzipped datasets inside the data folder
    - data/ed
    - data/hosp
    - data/icu

**Preprocessing the data**

Open the following Jupyter notebooks in the following order and run all cells:
1. `generate_patient_data.ipynb`&nbsp;&nbsp;\# To exclude patient records with invalid values
2. `generate_event_logs.ipynb`&nbsp;&nbsp;\# To convert the records to event logs
3. `generate_model_parameters.ipynb`&nbsp;&nbsp;\# To generate the parameters needed to run the ED simulation model
4. `generate_mci_frequency.ipynb`&nbsp;&nbsp;\# To generate modified parameters for experiment 3 (mass casualty incident scenarios)

**Running the experiments**
1. Run
    `chmod u+x ./run.sh`
2. Run
    `./run.sh`

**Generating the figures**

Open the following Jupyter notebooks and run all cells:
   1. `plot_results_exp1.ipynb`
   2. `plot_results_exp2.ipynb`
   3. `plot_results_exp3.ipynb`
   4. `plot_results_supplementary.ipynb`
