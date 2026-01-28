# fpyy_proj_template

Template for starting Fish-PACE project repos. 

This repo is an example of how teams can structure their project repositories and format their project README.md file, but feel free to adapt as suits your needs.

**Folder Structure**

* `contributor_folders` (optional) Each contributor can make a folder here and 
push their work here during the week. This will allow everyone to see each others work but prevent any merge conflicts.
* `final_notebooks` When the team develops shared final notebooks, they 
can be shared here. Make sure to communicate so that you limit merge conflicts.
* `scripts` Shared scripts or functions can be added here.
* `data` Shared dataset can be shared here. Note, do not put large datasets on GitHub. Speak to the organizers if you 
need to share large datasets. 

## Project Name

## One-line Description

## Planning

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

## Goals

## Datasets

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

## Lessons Learned

## References

