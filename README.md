# pytorch-multiGPU-template

This project is for freshman to easily use multiGPU to train their models. 

Users just need to modify the TODO parts of the codes.

In the following, I will introduce the TODO parts in detailed, freshman only need to modify it to their own processing code.


# TODO
## 1. models.py
* Defining models. Containing rewrite `forward processing` and `initializing model parameters`.\
* Rewriting the `generate_MyModel` function is for model generation in training model part.

## 2. util.datasets.py
* Building dataset, which could contain some data preprocessing such as data augment.

## 3. util.decay.py
If user need to train their model with layer decaying, they need to rewrite this part.
* `num_layers` definition. I make an example in the codes, if the model has a module named blocks.
* Users need to rewrite the `get_layer_id_for_model` to set id for each layer of the training model. When setting each layer's layer decay to according to this id.

## 4. engine_pretrain.py & engine_finetuning.py
This two .py file both contain `train_one_epoch` and `evaluate` function, but one is used in pretraing and anthor is used in finetuning.
* Users need to rewrite the the loss calculation part.
* Users need to rewrite the `evalutie` function. \

In this part, I recommand users to read the codes of class `SmoothedValue` and `MetricLogger` in `util.misc.py`. After reading, users could understand how to save the ouputed result, which is more easily for users to know how to display the result.

## 5. main_pretrain.py
If users need to do pre-training mission, they could firstly trained their models in this part.
* Defining the hyper-parameters which users will use in `get_args_parser` function.
* Rewriting the dataset loading part.
* Defining the model hyper-parameters and creating model.
* Setting optimizer. Optimizer AdamW have already been setted.
* Rewriting the result ouputing part.

## 6. main_finetune.py
If users need to do finetuning mission, the could run this .py file.
* Defining the hyper-parameters which users will use in `get_args_parser` function.
* Rewriting the dataset loading part.
* Defining the model hyper-parameters and creating model.
* Rewriting the pretrained model loading part. In this part, users need to be aware of three parts: 1. The final layer of the pretrained model whether need to be depected; 2. Checking the finetuning parts whether they is missing. 3. Initilizing the final layer's parameter.
* User could choose whether to set the `required_grad` of non-finetuning layer to False. This part I have alread written and commented in the code following the part 3.
* Setting optimizer. Optimizer AdamW have already been setted.
* Setting loss function.
* Rewriting the result ouputing part.

## 7. Running Programe
* Using multiGPU: 
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main_pretrain.py
* Using one GPU: CUDA_VISIBLE_DEVICES=0 python main_pretrain.py
