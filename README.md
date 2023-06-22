# flant5-finder

We are use Flan-T5 to classify prompts and non-prompts. The positive class (110460 prompts) are in ![here](https://github.com/XueFuzhao/InstructionWild/tree/main/data_v2) and for the negative class (3,848,330 non-prompts), ![reddit](https://huggingface.co/datasets/webis/tldr-17) dataset was used from Huggingface.

3 Flan-T5 models are trained in 3 different ways to try to combat the imbalanced dataset (majority class is around 35 times bigger than minority). 
- The first method was to not change anything but to train the classifier normally. 
- The second method was to upweight the minority class by increasing loss weights. The new loss weights are 35 for minority and 1 for majority
- The third method was to downsample the majority class and only take 20% of it as well as increasing its loss weights. The new loss weights are 1 for minority and 7 for majority

To run the code:

1. Download the data from ![here](https://github.com/XueFuzhao/InstructionWild)

2. Put the data in the same folder as the code

3. Install dependencies in requirements.txt
    ```bash
    pip install -r requirements.txt
    ```
4. Run train.py file. There are 5 arguments for the file

    ```
    --imsol: Method to solve data imbalance. 'none' for no change, 'up-weight' for upweighting, 'down-sample' for downsampling. Default is 'none'

    --split: Test split percentage. Default is 0.1

    --epochs: Number of epochs. Default is 3

    --hftoken: Huggingface token for uploading model. 

    --hfname: Huggingface model name. Default is 'flant5-finder'

    --wname: Name of w&b experiment. Default is 'flant5-finder'
    ```