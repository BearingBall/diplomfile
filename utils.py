import random
import cv2 as cv2
import dataset as data
import matplotlib.pyplot as plt
import attackMethods as am
import pickle
import torch

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