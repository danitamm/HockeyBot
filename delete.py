import torch

def get_accuracies(output, target, topk=(1,5)):
	batch_size = target.shape[0]
	_, preds = torch.topk(output, max(topk), dim=1)
	target = target.unsqueeze(-1).expand(preds.shape)
	compare = preds.eq(target)
	accs = [compare[:,:k,...].sum().float().item()/batch_size for k in topk]
	return accs

def cls_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    print(correct)
    print(correct.shape)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k / batch_size)
    return res

output = torch.load('tempoutput.pt')
target = torch.load('tempy.pt')


target = torch.tensor([928, 1383])

print(get_accuracies(output, target))

# print(cls_accuracy(output, target))


# def get_accuracy(pred, target):
# 	_, pred = torch.max(pred, 1)
# 	correct = (pred==target).sum().item()
# 	return float(correct)/target.shape[0]