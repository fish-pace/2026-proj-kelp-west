## Project: Kelp Health 

## Objective: Assess how variation in environmental drivers may affect kelp forest health on the west coast of North America

## Collaborators

| Name                | Role                |
|---------------------|---------------------|
| Participant 1       | Tena Dhayalan       |
| Participant 2       | Annie Hansen        |
| Participant 3       | Christina Asante    |

## Planning

* Initial idea: "To evaluate whether hyperspectral optical classification can be used to characterize states of kelp forest canopies in PACE OCI data along the California coast."
* Ideation jam board: https://docs.google.com/document/d/1Kx6FJGEp2w1XMeiL8nQI49V98aJvUqvLbATHDY1HyOI/edit?usp=sharing
* Slack channel: proj-kelp-west
* Project google drive: https://drive.google.com/drive/folders/1WxudYbTDOLmh2qQZroQ3ScTcqr5b9CER?usp=drive_link
* Final presentation: Add link

## Background
Kelp forests are critical coastal ecosystems that support biodiversity and fisheries, yet their spatial extent and health can change rapidly due to environmental stressors. Monitoring these changes, particularly in the wake of recent marine heatwaves along the west coast, is of particular concern

## Goals
* Use in situ algae counts to determine kelp forests and kelp species of interest, then characterize abundances
* Use PACE data in combination with in situ data to establish thresholds of PACE-derived indices to map kelp signals
* Use physical driver data (SST/CUTI) to establish relationships between kelp abundance and environmental drivers

## Datasets
* PISCO https://search.dataone.org/view/doi:10.6085/AA/PISCO_kelpforest.1.11
* PACE Level 2 OCI
* MUR SST
* CUTI https://mjacox.com/upwelling-indices/

## Workflow/Roadmap
``` mermaid
flowchart LR
    A["Identify variables of interest for kelp forest health"] -- <br> --> B("Find datasets that match up spatially and temporally with PACE OCI")
    B --> C["Rrs and wavelength"] & n1["SST"] & n2["Upwelling Indices"]
    n1 --> n3["How do temperature and nutrients affect kelp forest health over time?"]
    C --> n4["QWIP analysis"]
    n2 --> n3
    n4 --> n3

    A@{ shape: rounded}
    C@{ shape: rounded}
    n1@{ shape: rounded}
    n2@{ shape: rounded}
```
## Results/Findings
* Channel Islands kelp forests have large datasets available
* Kelp species with taller canopies ( Macrocystis pyrifera) might be easier to detect with PACE than shorter species (Laminaria farlowii) that tend to form an understory
* QWIP Type III may not be able to capture kelp senescence
* Normalized difference vegetation Index (NVDI) more promising than Floating Algae Index (FAI) in identifying kelp-specific signals
* Physical drivers not 1:1 to changes in kelp health, may need a time lag or a more advanced model to accomplish this

## Lessons Learned
* Merging datasets across time and space is hard!
* How to: Python, Jupyter Notebooks, GitHub, and PACE :)

## References
Hu, C. (2009). A novel ocean color index to detect floating algae in the global oceans. Remote Sensing of Environment, 113(10), 2118-2129.
Jacox MG, CA Edwards, EL Hazen, SJ Bograd (2018) Coastal upwelling revisited: Ekman, Bakun, and improved upwelling indices for the U.S. west coast, J. Geophysical Research, 123(10), 7332-7350.
Jensen, J. R., Estes, J. E., & Tinney, L. (1980). Remote sensing techniques for kelp surveys. Photogrammetric Engineering and Remote Sensing, 46(6), 743-755.
Muzhoffar DAF, Sakuno Y, Taniguchi N, Hamada K, Shimabukuro H, Hori M. Automatic Detection of Floating Macroalgae via Adaptive Thresholding Using Sentinel-2 Satellite Data with 10 m Spatial Resolution. Remote Sensing. 2023; 15(8):2039. https://doi.org/10.3390/rs15082039

