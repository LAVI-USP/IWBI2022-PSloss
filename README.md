======

This repository contains the training and testing codes for the paper "[Imposing noise correlation fidelity on digital breast tomosynthesis restoration through deep learning techniques]()", submitted to the IWBI 2022 conference.

## Abstract:

Digital breast tomosynthesis (DBT) is an important imaging modality for breast cancer screening. The morphology of breast masses and the shape of the microcalcifications are important factors to detect and determine the malignancy of breast cancer. Recently, convolutional neural networks (CNNs) have been used for denoising in medical imaging and have shown potential to improve the performance of radiologists. However, they can impose noise spatial correlation in the restoration process. Noise correlation can negatively impact radiologists' performance, creating image signals that can resemble breast lesions. In this work, we propose a deep CNN that restores low-dose DBT projections by partially filtering out the noise, but imposes fidelity of the noise correlation between the original and restored images, avoiding the creation of signals that may resemble signs of breast cancer. The combination of a loss function that calculates the difference in the power spectra (PS) of the input and output images and another one that seeks image visual perception is proposed. We compared the performance of the proposed neural network with traditional denoising methods that do not consider the noise correlation in the restoration process and found superior results in terms of PS for our approach.


## Some results:

Soon

## Reference:

If you use the codes, we will be very grateful if you refer to this [paper]():

> Soon

## Acknowledgments:

This work was supported by the São Paulo Research Foundation ([FAPESP](http://www.fapesp.br/) grant 2021/12673-6) and by the National Council for Scientific and Technological Development ([CNPq](http://www.cnpq.br/)) and by the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior ([CAPES](https://www.gov.br/capes/pt-br) - finance code 001). We would like to thank the contribution of our lab members and the [Instituto de Radiologia (inRad) do Hospital das Clínicas da Faculdade de Medicina da Universidade de São Paulo (FMUSP-HC)](http://www.hc.fm.usp.br) for providing the DBT images.


---
Laboratory of Computer Vision ([Lavi](http://iris.sel.eesc.usp.br/lavi/))  
Department of Electrical and Computer Engineering  
São Carlos School of Engineering, University of São Paulo  
São Carlos - Brazil

AI-based X-ray Imaging System ([AXIS](https://wang-axis.github.io))  
Department of Biomedical Engineering  
Rensselaer Polytechnic Institute  
Troy - USA
