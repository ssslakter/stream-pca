# robust-pca
Optimization project with ADMM and robust PCA


## main paper 
Online Robust Principal Component Analysis with Change Point Detection (https://arxiv.org/abs/1702.05698 )



+ in model.py, we assumed that for all the frame, the background does not change, take mean and vary the lighting condition -> not optimal

+ model_full_rpca.py . assumed background does not change, use ADMM to find obtimal separation -> require big amount of memory as we process it all together

+ model_online.py -> use SGD to perform the separation online, assumed the background does not change or change very slowly (input next frame -> output new background, background basis)

+ model_online_dynamic.py -> similar to model_online.py, but only do SGD on last N frames

+ TODO: Impelement "Change Point Detection in Online Moving Window RPCA with Hypothesis Testing"