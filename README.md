<div align="center">
  <h1>Dual-level Adaptation for Multi-Object Tracking: Building Test-Time Calibration from Experience and Intuition</h1>
</div>

<div align="center">
  Wen Guo<sup>1</sup>,&nbsp;&nbsp; Pengfei Zhao<sup>1</sup>,&nbsp;&nbsp; Zongmeng Wang<sup>2</sup>,&nbsp;&nbsp;
  Yufan Hu<sup>3</sup><sup>*</sup>,&nbsp;&nbsp; Junyu Gao<sup>4</sup><br>
  <sup>1</sup>Shandong Technology and Business University, &nbsp;&nbsp;
  <sup>2</sup>Inner Mongolia University,<br>
  <sup>3</sup>University of Science and Technology Beijing, &nbsp;&nbsp;
  <sup>4</sup>Institute of Automation, Chinese Academy of Sciences, <br>
  <sup>*</sup>Corresponding Author.
</div>


## Highlight
⭐ Our paper has been accepted by CVPR2026 !

## Abstract
Multiple Object Tracking (MOT) has long been a fundamental task in computer vision, with broad applications in various real-world scenarios.
However, due to distribution shifts in appearance, motion pattern, and catagory between the training and testing data, model performance degrades considerably during online inference in MOT.
Test-Time Adaptation (TTA) has emerged as a promising paradigm to alleviate such distribution shifts.
However, existing TTA methods often fail to deliver satisfactory results in MOT, as they primarily focus solely on frame-level adaptation while neglecting temporal consistency and identity association across frames and videos.
Inspired by human decision-making process, this paper propose a Test-time Calibration from Experience and Intuition (TCEI) framework.
In this framework, the Intuitive system utilizes transient memory to recall recently observed objects for rapid predictions, while the Experiential system leverages the accumulated experience from prior test videos to reassess and calibrate these intuitive predictions.
Furthermore, both confident and uncertain objects during online testing are exploited as historical priors and reflective cases, respectively, enabling the model to adapt to the testing environment and alleviate performance degradation.
Extensive experiments demonstrate that the proposed TCEI framework consistently achieves superior performance across multiple benchmark datasets and significantly enhances the model's adaptability under distribution shifts.

