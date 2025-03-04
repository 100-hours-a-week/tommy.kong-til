---
title: Deep Residual Learning for Image Recognition
authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
year: 2015
publisher : arXiv
---
#Summary
Deeper neural networks are more difﬁcult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers—8× deeper than VGG nets [41] but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classiﬁcation task. We also present analysis on CIFAR-10 with 100 and 1000 layers.
#서지
[Go to Zotero](zotero://select/items/@heDeepResidualLearning2015)

#번역
심층 신경망은 훈련하기가 더 어렵습니다. 저희는 이전에 사용했던 것보다 훨씬 더 심층적인 네트워크를 쉽게 훈련할 수 있는 잔여 학습 프레임워크를 제시합니다. 참조되지 않은 함수를 학습하는 대신 레이어 입력을 참조하여 잔차 함수를 학습하는 것으로 레이어를 명시적으로 재구성합니다. 이러한 잔류 네트워크가 더 쉽게 최적화할 수 있으며, 상당히 깊어진 심도에서 정확도를 높일 수 있음을 보여주는 포괄적인 경험적 증거를 제공합니다. 이미지넷 데이터 세트에서는 최대 152개의 층으로 구성된 잔여 네트워크를 평가했는데, 이는 VGG 네트워크[41]보다 8배 더 깊지만 복잡도는 여전히 낮습니다. 이러한 잔여 네트워크의 앙상블은 ImageNet 테스트 세트에서 3.57%의 오류를 달성했습니다. 이 결과는 ILSVRC 2015 분류 과제에서 1위를 차지했습니다. 또한 100개와 1000개의 레이어가 있는 CIFAR-10에 대한 분석 결과도 제시합니다.