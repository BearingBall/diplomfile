import random
import cv2 as cv2
import dataset as data
import matplotlib.pyplot as plt
import attackMethods as am
import pickle
import torch
import numpy as np

'''
This model needed to look at dataset and check correckness of dataset parsing and model working.

"q" = exit
"space" = get another random pitcure from dataset

'''

def lookDataset(model, dataset):
    model.eval()
    model = model.float()

    flag = True

    while (flag):
        k = random.randint(0,len(dataset))
        image, labels = dataset[k]

        for l in labels:
            image = cv2.rectangle(image, ( int(l["bbox"][0]), int(l["bbox"][1]) ), ( int(l["bbox"][0])+int(l["bbox"][2]),int(l["bbox"][1])+int(l["bbox"][3]) ), (255,0,0), 1)

        prediction = model([data.ImToTen(image)])

        for l in prediction[0]['boxes']:
            image = cv2.rectangle(image, ( int(l[0]), int(l[1]) ), ( int(l[0])+int(l[2]),int(l[1])+int(l[3]) ), (0,0,255), 1)

        cv2.imshow(str(k), image)

        while (True):
            key = cv2.waitKey(1)
            if key == 32:
                cv2.destroyAllWindows() 
                break

            if key & 0xFF == ord('q'):
                flag = False
                break
  
    cv2.destroyAllWindows() 


def ShowPatch(patch):
    plt.imshow(patch.permute(1, 2, 0).detach().numpy())

def ShowImageWithPatch(image, label, patch, device):
    image = data.ImToTen(image)

    for l in label:
        if l["category_id"] == 1:
            image = am.setPatch(image, patch, l['bbox'], 0.2, device) 

    plt.imshow(image.permute(1, 2, 0).detach().numpy())

def SavePatch(patch, name):
    with open(name+ '.pickle', 'wb') as f:
        pickle.dump( data.TenToIm(patch), f)

def LoadPatch(name):
    with open(name + '.pickle', 'rb') as f:
        return data.ImToTen(pickle.load(f))


def TrainOneImage(model, loss, image, label, epoches = 30, GradRate = 0.05, TVCoeff = 0.0001):
    device = torch.device("cpu")

    model.eval()
    model = model.float().to(device)

    patch = am.generatePatch()
    patch = patch.to(device)

    #personIndex = []

    for epoch in range(epoches):
        
        patch.requires_grad = True
        attackedImage = data.ImToTen(image)

        attackedImage = attackedImage.to(device)
        attackedImage.requires_grad = True

        wasPerson = False

        for l in label:
            if l["category_id"] == 1:
                wasPerson = True
                attackedImage = am.setPatch(attackedImage, patch, l['bbox'], 0.2, device) 

        if wasPerson:
            #personIndex.append(i)

            cost = loss(label, model([attackedImage])[0]) + TVCoeff * am.TV(patch)
            grad = torch.autograd.grad(cost, patch, retain_graph=False, create_graph=False,  allow_unused=True)[0]
            patch.requires_grad = False
            if grad != None:
                patch = patch - GradRate*grad.sign()

        print("ep:", epoch, " loss:", cost)
    
    return patch

def CheckPatch(patch, image, label, model, device, loss):
    image = data.ImToTen(image)
    clearPred = model([image])[0]

    attackedImage = image

    for l in label:
            if l["category_id"] == 1:
                wasPerson = True
                attackedImage = am.setPatch(attackedImage, patch, l['bbox'], 0.2, device)

    attackedPred = model([attackedImage])[0]

    print("Clear loss:", loss(label, clearPred), "Attacked loss:", loss(label, attackedPred) )

def labelization(image, label, threshold):
    for i in range(len(label["labels"])):
        if label["scores"][i] < threshold:
            continue
        if label["labels"][i] == 1:
            image = cv2.rectangle(image, (int(label['boxes'][i][0]), int(label['boxes'][i][1]) ), ( int(label['boxes'][i][0])+int(label['boxes'][i][2]),int(label['boxes'][i][1])+int(label['boxes'][i][3])), (255,0,0), 1)
    return image

def testPatch(patch, image, label, model, device, loss, threshold = 0.0):
    clearPred = model([data.ImToTen(image).to(device)])[0]
    attackedImage = data.ImToTen(image)
    attackedImage = attackedImage.to(device)
    for l in label:
        attackedImage = am.setPatch(attackedImage, patch, l, 0.2, device) 

    attackedPred = model([attackedImage])[0]

    image_before = labelization(cv2.cvtColor(data.TenToIm((data.ImToTen(image)).cpu()), cv2.COLOR_RGB2BGR), clearPred, threshold)
    image_after = labelization(cv2.cvtColor(data.TenToIm(attackedImage.cpu()), cv2.COLOR_RGB2BGR), attackedPred, threshold)

    cv2.imshow("image", np.concatenate((image_before, image_after), axis=1)/255)

    flag = True
    while (flag):
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            flag = False
            break
  
    cv2.destroyAllWindows() 

