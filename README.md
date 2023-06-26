# Deep Learning for Multiphase Segmentation of X-ray Tomography of Gas Diffusion Layers 
High-resolution X-ray computed tomography (micro-CT) has been widely used to characterise fluid flow in porous media for different applications, including in gas diffusion layers (GDLs) in fuel cells. In this study, we examine the performance of 2D and 3D U-Net deep learning models for multiphase segmentation of unfiltered X-ray tomograms of GDLs with different percentages of hydrophobic polytetrafluoroethylene (PTFE). The data is obtained by micro-CT imaging of GDLs after brine injection. We train deep learning models on base-case data prepared by the 3D Weka segmentation method and test them on the unfiltered unseen datasets. Our assessments highlight the effectiveness of the 2D and 3D U-Net models with test IoU values of 0.901 and 0.916 and f1-scores of 0.947 and 0.954, respectively. Most importantly, the U-Net models outperform conventional 3D trainable Weka and watershed segmentation based on various visual examinations. Lastly, flow simulation studies reveal segmentation errors associated with trainable Weka and watershed segmentation lead to significant errors in the calculated porous media properties, such as absolute permeability. Our findings show 43, 14, 14, and 3.9% deviations in computed permeabilities for GDLs coated by 5, 20, 40, and 60 w% of PTFE, respectively, compared to images segmented by the 3D Weka segmentation method.
![image](https://github.com/MehdiMahdaviara/2D-3D-UNet-for-GDL-segmentation/assets/99279360/a932b046-ddcc-40f7-b5ac-f03dbe7ebeb6)
