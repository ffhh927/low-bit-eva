from resnet20 import ResNet
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, LlamaForCausalLM, LlamaTokenizer
import os

if __name__=='__main__':

    #net = ResNet()
    
    #print(net)
    #total = sum(p.numel() for p in net.parameters())
    #print("Total params: %.2fM" % (total / 1e6))
    

    # 加载RoBERTa-large预训练模型和tokenizer
    #model_name = "roberta-large"
    #modelpath = os.path.join('./', model_name)
    #tokenizer = RobertaTokenizer.from_pretrained(modelpath)
    #model = RobertaForSequenceClassification.from_pretrained(modelpath)
    
    
    model_path = "Llama-7b"
    tokenizer = LlamaTokenizer.from_pretrained(model_path, local_files_only=True)
    model = LlamaForCausalLM.from_pretrained(model_path, local_files_only=True)
    print(model)
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total / 1e6))