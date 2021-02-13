import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from covidxdataset import COVIDxDataset
from metric import accuracy
from util import print_stats, print_summary, select_optimizer, MetricTracker
from loss import csa_loss


def mm_pair_train(device, batch_size, a_m, i_m, cls, trainloader, optimizer, optimizer2, epoch, writer):
    a_m.train()
    i_m.train()
    cls.train()
    ce_loss = nn.CrossEntropyLoss(reduction='mean')
    alpha = 1

    metric_ftns = ['loss', 'correct', 'nums', 'accuracy']
    image_metrics = MetricTracker(*[m for m in metric_ftns], writer=writer, mode='train')
    audio_metrics = MetricTracker(*[m for m in metric_ftns], writer=writer, mode='train')
    image_metrics.reset()
    audio_metrics.reset()
    i_confusion_matrix = torch.zeros(2, 2)
    a_confusion_matrix = torch.zeros(2, 2)

    for batch_idx, (audio, a_label, img, i_label, p_img) in enumerate(trainloader):
        
        
        audio, img = audio.to(device), img.to(device)
        a_label, i_label = a_label.to(device), i_label.to(device)
        p_img = p_img.to(device)
        i_output, i_feature = i_m(img)
        a_output, a_feature = a_m(audio)
        _, pair_i_feature = i_m(p_img)

        i_ce  = ce_loss(i_output, i_label)
        a_ce  = ce_loss(a_output, a_label)
        csa = csa_loss(a_feature, i_feature.detach(), (a_label == i_label).float())
                                                    
        loss = i_ce + 0.2 * a_ce + alpha * csa
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, a_feature2 = a_m(audio)
        concat_output = cls(torch.cat([pair_i_feature, a_feature2], 1))
        
        concat_ce = ce_loss(concat_output, a_label)
        optimizer2.zero_grad()
        concat_ce.backward()
        optimizer2.step()
        
        num_samples = batch_idx * batch_size + 1
        # image
        i_correct, i_nums, i_acc = accuracy(i_output, i_label)
        image_metrics.update_all_metrics({'correct': i_correct, 'nums': i_nums, 'loss': loss.item(), 'accuracy': i_acc},
                                         writer_step=(epoch - 1) * len(trainloader) + batch_idx)
        # audio
        a_correct, a_nums, a_acc = accuracy(a_output, a_label)
        audio_metrics.update_all_metrics({'correct': a_correct, 'nums': a_nums, 'loss': loss.item(), 'accuracy': a_acc},
                                         writer_step=(epoch - 1) * len(trainloader) + batch_idx)
        _, preds = torch.max(a_output, 1)
        for t, p in zip(a_label.cpu().view(-1), preds.cpu().view(-1)):
            a_confusion_matrix[t.long(), p.long()] += 1
        print_stats(epoch, batch_size, num_samples, trainloader, image_metrics, mode="Image", acc=i_acc)
        print_stats(epoch, batch_size, num_samples, trainloader, audio_metrics, mode="Audio", acc=a_acc)
    num_samples += len(a_output) - 1

    print_summary(epoch, num_samples, image_metrics, mode="Training Image")
    print_summary(epoch, num_samples, audio_metrics, mode="Training Audio")
    print('A_Confusion Matrix\n{}\n'.format(a_confusion_matrix.cpu().numpy()))
    
    return audio_metrics

def mm_pair_valid(device, batch_size, a_m, i_m, cls, testloader, epoch, writer):
    a_m.eval()
    i_m.eval()
    cls.eval()
    criterion = nn.CrossEntropyLoss(reduction='mean')

    metric_ftns = ['loss', 'correct', 'nums', 'accuracy']
    val_metrics = MetricTracker(*[m for m in metric_ftns], writer=writer, mode='val')
    val_metrics.reset()
    confusion_matrix = torch.zeros(2, 2)
    with torch.no_grad():
        for batch_idx, (audio, img, label) in enumerate(testloader):
            
            audio, img = audio.to(device), img.to(device)
            label = label.to(device)
            i_output, i_feature = i_m(img)
            a_output, a_feature = a_m(audio)
            concat_output = cls(torch.cat([i_feature, a_feature], 1))

            loss = criterion(concat_output, label)

            correct, nums, acc = accuracy(concat_output, label)
            num_samples = batch_idx * batch_size + 1
            _, preds = torch.max(concat_output, 1)
            for t, p in zip(label.cpu().view(-1), preds.cpu().view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            val_metrics.update_all_metrics({'correct': correct, 'nums': nums, 'loss': loss.item(), 'accuracy': acc},
                                           writer_step=(epoch - 1) * len(testloader) + batch_idx)
    
    num_samples += len(label) - 1
    print_summary(epoch, num_samples, val_metrics, mode="Validation")

    print('Confusion Matrix\n{}'.format(confusion_matrix.cpu().numpy()))
    return val_metrics, confusion_matrix

def mm_train(device, batch_size, a_m, i_m, trainloader, optimizer, epoch, writer):
    a_m.train()
    i_m.train()
    weight = torch.tensor([0.1, 0.9]).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=weight, reduction='mean')
    alpha = 1

    metric_ftns = ['loss', 'correct', 'nums', 'accuracy']
    image_metrics = MetricTracker(*[m for m in metric_ftns], writer=writer, mode='train')
    audio_metrics = MetricTracker(*[m for m in metric_ftns], writer=writer, mode='train')
    image_metrics.reset()
    audio_metrics.reset()
    i_confusion_matrix = torch.zeros(2, 2)
    a_confusion_matrix = torch.zeros(2, 2)

    for batch_idx, (audio, a_label, img, i_label) in enumerate(trainloader):
        
        
        audio, img = audio.to(device), img.to(device)
        a_label, i_label = a_label.to(device), i_label.to(device)
        i_output, i_feature = i_m(img)
        a_output, a_feature = a_m(audio)

        i_ce  = ce_loss(i_output, i_label)
        a_ce  = ce_loss(a_output, a_label)
        csa = csa_loss(a_feature, i_feature.detach(), (a_label == i_label).float())
                                                    
        loss = i_ce + 0.4 * a_ce + alpha * csa
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        num_samples = batch_idx * batch_size + 1
        # image
        i_correct, i_nums, i_acc = accuracy(i_output, i_label)
        image_metrics.update_all_metrics({'correct': i_correct, 'nums': i_nums, 'loss': loss.item(), 'accuracy': i_acc},
                                         writer_step=(epoch - 1) * len(trainloader) + batch_idx)
        # audio
        a_correct, a_nums, a_acc = accuracy(a_output, a_label)
        audio_metrics.update_all_metrics({'correct': a_correct, 'nums': a_nums, 'loss': loss.item(), 'accuracy': a_acc},
                                         writer_step=(epoch - 1) * len(trainloader) + batch_idx)
        _, preds = torch.max(a_output, 1)
        for t, p in zip(a_label.cpu().view(-1), preds.cpu().view(-1)):
            a_confusion_matrix[t.long(), p.long()] += 1
        print_stats(epoch, batch_size, num_samples, trainloader, image_metrics, mode="Image", acc=i_acc)
        print_stats(epoch, batch_size, num_samples, trainloader, audio_metrics, mode="Audio", acc=a_acc)
    num_samples += len(a_output) - 1

    print_summary(epoch, num_samples, image_metrics, mode="Training Image")
    print_summary(epoch, num_samples, audio_metrics, mode="Training Audio")
    print('A_Confusion Matrix\n{}\n'.format(a_confusion_matrix.cpu().numpy()))
    
    return audio_metrics


def train(device, batch_size, model, trainloader, optimizer, epoch, writer):
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='mean')

    metric_ftns = ['loss', 'correct', 'nums', 'accuracy']
    train_metrics = MetricTracker(*[m for m in metric_ftns], writer=writer, mode='train')
    train_metrics.reset()
    confusion_matrix = torch.zeros(2, 2)

    for batch_idx, input_tensors in enumerate(trainloader):
        optimizer.zero_grad()
        input_data, target = input_tensors
        input_data = input_data.to(device)
        target = target.to(device)

        output = model(input_data)

        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        correct, nums, acc = accuracy(output, target)
        num_samples = batch_idx * batch_size + 1
        _, preds = torch.max(output, 1)
        for t, p in zip(target.cpu().view(-1), preds.cpu().view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        train_metrics.update_all_metrics({'correct': correct, 'nums': nums, 'loss': loss.item(), 'accuracy': acc},
                                         writer_step=(epoch - 1) * len(trainloader) + batch_idx)
        print_stats(epoch, batch_size, num_samples, trainloader, train_metrics)
    num_samples += len(target) - 1

    print_summary(epoch, num_samples, train_metrics, mode="Training")
    print('Confusion Matrix\n{}\n'.format(confusion_matrix.cpu().numpy()))
    return train_metrics


def validation(device, batch_size, classes, model, testloader, epoch, writer):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='mean')

    metric_ftns = ['loss', 'correct', 'nums', 'accuracy']
    val_metrics = MetricTracker(*[m for m in metric_ftns], writer=writer, mode='val')
    val_metrics.reset()
    confusion_matrix = torch.zeros(classes, classes)
    with torch.no_grad():
        for batch_idx, input_tensors in enumerate(testloader):

            input_data, target = input_tensors
            input_data = input_data.to(device)
            target = target.to(device)

            output,_ = model(input_data)

            loss = criterion(output, target)

            correct, nums, acc = accuracy(output, target)
            num_samples = batch_idx * batch_size + 1
            _, preds = torch.max(output, 1)
            for t, p in zip(target.cpu().view(-1), preds.cpu().view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            val_metrics.update_all_metrics({'correct': correct, 'nums': nums, 'loss': loss.item(), 'accuracy': acc},
                                           writer_step=(epoch - 1) * len(testloader) + batch_idx)
    
    num_samples += len(target) - 1
    print_summary(epoch, num_samples, val_metrics, mode="Validation")

    print('Confusion Matrix\n{}'.format(confusion_matrix.cpu().numpy()))
    return val_metrics, confusion_matrix
