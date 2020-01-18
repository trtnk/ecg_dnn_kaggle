import torch
from tqdm import tqdm

# train and validate network
def train_model(net, dataloader_dict, criterion, optimizer, num_epochs):
    # initial setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    net.to(device)
    torch.backends.cudnn.benchmark = True

    # epoch loop
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("-----------------------")
        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()
            epoch_loss = 0.0
            epoch_corrects = 0

            if (epoch == 0) and (phase == "train"):
                continue

            for inputs, labels in tqdm(dataloader_dict[phase]):
                inputs = inputs.float()
                labels = labels.long()
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # back propagation if phase == "train"
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)
                epoch_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects / len(dataloader_dict[phase].dataset)

            print("")
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

def test_model(net, dataloader):
    correct = 0
    total = 0
    predicted_labels = []
    correct_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.float()
            labels = labels.long()
            outputs = net(inputs)

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += torch.sum(preds == labels.data).item()

            predicted_labels.extend(preds.tolist())
            correct_labels.extend(labels.tolist())

    acc = correct/total*100
    print(f"Accuracy: {acc}")
    return predicted_labels, correct_labels

# Reference:
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
def test_model_each_class(net, dataloader, class_num):
    class_correct = list(0. for i in range(class_num))
    class_total = list(0. for i in range(class_num))

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.float()
            labels = labels.long()
            outputs = net(inputs)
            _, preds = torch.max(outputs, 1)
            c = (preds == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(class_num):
        print("Accuracy of class {} : {:.4f}".format(i, 100 * class_correct[i] / class_total[i]))
    return
