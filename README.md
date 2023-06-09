# flant5-finder

We are use Flan-T5 to classify prompts and not prompts. The positive class (110460 prompts) are in ![here](https://github.com/XueFuzhao/InstructionWild/tree/main/data_v2) and for the negative class (3,848,330 non-prompts), ![reddit](https://huggingface.co/datasets/webis/tldr-17) dataset was used form Huggingface.

Flan-T5 is trained in 3 different ways to try to combat the imbalanced dataset (majority class is around 35 times bigger than minority). 
- The first method was to not change anything but to train the classifier normally. 
- The second method was to upweight the minority class by increasing loss weights. The new loss weights are 35 for minority and 1 for majority
- The third method was to downsample the majority class and only take 20% of it as well as increasing its loss weights. The new loss weights are 1 for minority and 7 for majority