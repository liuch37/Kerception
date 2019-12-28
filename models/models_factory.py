import tensorflow as tf
from models import resnet
from models import leNet
from models import resnet_KNN
from models import leNet_KNN

def get_model(model_name, num_classes=10, keep_prob=1.0, **kwargs):
    if model_name.lower()=="lenetknn":
        return leNet_KNN.LeNet5KNN(num_classes=num_classes,
                        keep_prob=keep_prob,
                        **kwargs)
    elif model_name.lower()=='lenet':
        return leNet.LeNet5(num_classes=num_classes,
                         keep_prob=keep_prob)
    elif model_name.lower()=='resnet18':
        return resnet.ResNet18(num_classes=num_classes)
    elif model_name.lower()=='resnet34':
        return resnet.ResNet34(num_classes=num_classes)
    elif model_name.lower()=='resnet101':
        return resnet.ResNet101(num_classes=num_classes)
    elif model_name.lower()=="resnet18knn":
        return resnet_KNN.ResNet18(num_classes=num_classes, kernel_fn=kwargs["kernel_fn"])
    elif model_name.lower()=="resnet101knn":
        return resnet_KNN.ResNet101(num_classes=num_classes, kernel_fn=kwargs["kernel_fn"])
    elif model_name.lower()=='lenetkcnn':
        return leNet_KNN.LeNet5KCNN(num_classes=num_classes, keep_prob=keep_prob,**kwargs)
    elif model_name.lower()=='resnet101kcnn':
        return resnet_KNN.ResNet101KCNN(num_classes=num_classes, keep_prob=keep_prob,**kwargs)
    else:
        raise ValueError("Unknown model name {}".format(model_name)) 

    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
