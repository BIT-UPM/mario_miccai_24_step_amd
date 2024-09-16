# mario_miccai_2024_step_amd
The study introduces a novel approach for classifying and
predicting the progression of Age-related Macular Degeneration (AMD)
using Optical Coherence Tomography (OCT) images and Multiple In-
stance Learning (MIL). AMD is a leading cause of vision impairment
worldwide, making effective monitoring and treatment essential, partic-
ularly with anti-VEGF therapy. However, the increasing number of pa-
tients and the frequency of follow-up visits pose challenges for healthcare
systems.This approach addresses two key tasks: (1) classifying changes
between consecutive 2D OCT B-scans and (2) predicting disease pro-
gression within a 3-month period. For task 1, the model incorporates
contextual information from adjacent B-scans and applies bidirectional
cross-attention to learn time-dependent features. For task 2, a MIL-based
architecture is used to identify the most significant slices within an OCT
volume.The results demonstrated the effectiveness of the proposed meth-
ods. In task 1, the model achieved a mean score of 0.7488 across all
evaluation metrics. For task 2, the mean score was 0.4478, reflecting
the complexity of disease progression prediction. This approach offers
improvements over baseline models and contributes to developing au-
tomated tools for AMD management, potentially easing the burden on
ophthalmology services and improving personalized patient care.
