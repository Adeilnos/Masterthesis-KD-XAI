import torch.nn.functional as F
import torch.nn as nn


#response based
def knowledge_distillation_loss(student_output,teacher_output,labels,label_smoothing=0.0,alpha=0.95,T=3.5):
    """Compute the KD-loss between student_output, teacher_output and the correct labels. 

    Args:
        student_output (Tensor): Logits (unnormalized output) of the student model
        teacher_output (Tensor): Logits (unnormalized output) of the teacher model
        labels (Tensor): Tensor of integers between [0,C]
        alpha (float, optional): alpha value, as described in the paper. Defaults to 0.95.
        T (float, optional): temperature value for the softmax, as described in the paper. Defaults to 3.5.
    """

    kd_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_output/T,dim=1),
                        F.softmax(teacher_output/T,dim=1)) * (alpha * T * T)
    cr_loss = F.cross_entropy(student_output,labels,label_smoothing=label_smoothing) * (1. - alpha)
    return kd_loss, cr_loss, kd_loss + cr_loss

#online feature based
def feature_based_distillation_loss(student_output,teacher_output,student_layer_features,teacher_layer_features,labels,label_smoothing=0.0,alpha=0.95,beta=0.05):
    """Compute the feature-based knowledge distillation loss.
    
        Args:
            student_output (torch.Tensor): Logits or raw predictions from the student model.
            teacher_output (torch.Tensor): Logits or raw predictions from the teacher model.
            student_layer_features (torch.Tensor): Features from a specific layer of the student model.
            teacher_layer_features (torch.Tensor): Features from the corresponding layer of the teacher model.
            labels (torch.LongTensor): True labels for the input samples.
            label_smoothing (float, optional): Label smoothing parameter. Default is 0.0.
            alpha (float, optional): Weight for the knowledge distillation loss. Default is 0.95.
    
        Returns:
            tuple: A tuple containing the knowledge distillation loss, student's cross-entropy loss, teacher's cross-entropy loss, total loss, and true total loss.    
    """
    #apply average pooling if layers arent the same size
    if teacher_layer_features.shape[1] > student_layer_features.shape[1]:
        teacher_layer_features = F.adaptive_avg_pool1d(teacher_layer_features, student_layer_features.shape[1])
    
    if student_layer_features.shape[1] > teacher_layer_features.shape[1]:
        student_layer_features = F.adaptive_avg_pool1d(student_layer_features, teacher_layer_features.shape[1])
        
        
    
    kd_loss = F.mse_loss(student_layer_features,teacher_layer_features)
    
    # kd_loss_output = F.cross_entropy(student_output,teacher_output,label_smoothing=label_smoothing)

    cr_loss_student = F.cross_entropy(student_output,labels,label_smoothing=label_smoothing)
    
    cr_loss_teacher = F.cross_entropy(teacher_output,labels,label_smoothing=label_smoothing)
    
    total_loss = beta * kd_loss + alpha * (1.-beta) * cr_loss_student + (1. - alpha) * (1. - beta) * cr_loss_teacher
    
    # total_loss = alpha * cr_loss_student + (1. - alpha) * kd_loss + beta * cr_loss_teacher

    
    true_total_loss = kd_loss + cr_loss_student + cr_loss_teacher
    
    return kd_loss, cr_loss_student, cr_loss_teacher, total_loss, true_total_loss


#offline feature based
def offline_feature_based_distillation_loss(student_output,teacher_output,student_layer_features,teacher_layer_features,labels,label_smoothing=0.0,alpha=0.95,beta=0.05):
    """Compute the feature-based knowledge distillation loss.
    
        Args:
            student_output (torch.Tensor): Logits or raw predictions from the student model.
            teacher_output (torch.Tensor): Logits or raw predictions from the teacher model.
            student_layer_features (torch.Tensor): Features from a specific layer of the student model.
            teacher_layer_features (torch.Tensor): Features from the corresponding layer of the teacher model.
            labels (torch.LongTensor): True labels for the input samples.
            label_smoothing (float, optional): Label smoothing parameter. Default is 0.0.
            alpha (float, optional): Weight for the knowledge distillation loss. Default is 0.95.
    
        Returns:
            tuple: A tuple containing the knowledge distillation loss, student's cross-entropy loss, teacher's cross-entropy loss, total loss, and true total loss.    
    """
    #apply average pooling if layers arent the same size
    if teacher_layer_features.shape[1] > student_layer_features.shape[1]:
        teacher_layer_features = F.adaptive_avg_pool1d(teacher_layer_features, student_layer_features.shape[1])
    
    if student_layer_features.shape[1] > teacher_layer_features.shape[1]:
        student_layer_features = F.adaptive_avg_pool1d(student_layer_features, teacher_layer_features.shape[1])
        
        
    
    kd_loss = F.mse_loss(student_layer_features,teacher_layer_features)
    

    cr_loss_student = F.cross_entropy(student_output,labels,label_smoothing=label_smoothing)
    
    cr_loss_teacher = F.cross_entropy(teacher_output,labels,label_smoothing=label_smoothing)
    
    total_loss = (1. - alpha) * kd_loss + alpha * cr_loss_student
    
    true_total_loss = kd_loss + cr_loss_student + cr_loss_teacher
    
    return kd_loss, cr_loss_student, cr_loss_teacher, total_loss, true_total_loss












