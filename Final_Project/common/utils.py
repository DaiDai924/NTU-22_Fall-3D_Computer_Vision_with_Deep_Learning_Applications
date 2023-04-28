import torch
import numpy as np
import hashlib
from torch.autograd import Variable
import os
    
def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2 ** 32 - 1) * (max_value - min_value)) + min_value


def mpjpe_cal(predicted, target, device, loss_fn, norm=2):
    assert predicted.shape == target.shape

    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    if loss_fn == "mpjpe":
        return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1, p=norm))
    
    elif loss_fn == "joint":
        pred_joint, gt_joint = torch.tensor([]).to(device), torch.tensor([]).to(device)
        for i in range(0, 17):

            pred_joint = torch.cat(
                (pred_joint, (predicted[:, :, i] - predicted[:, :, 0]).reshape(predicted.shape[0], predicted.shape[1], 1, 3)), 2
            )
            gt_joint = torch.cat(
                (gt_joint, (target[:, :, i] - target[:, :, 0]).reshape(target.shape[0], target.shape[1], 1, 3)), 2
            )

        return torch.mean(torch.norm(pred_joint - gt_joint, dim=len(gt_joint.shape) - 1, p=norm))

    elif loss_fn == "bone":
        pred_bones, gt_bones = torch.tensor([]).to(device), torch.tensor([]).to(device)
        for conn in connections:
            pred_bones = torch.cat(
                (pred_bones, (predicted[:, :, conn[0]] - predicted[:, :, conn[1]]).reshape(predicted.shape[0], predicted.shape[1], 1, 3)), 2
            )
            gt_bones = torch.cat(
                (gt_bones, (target[:, :, conn[0]] - target[:, :, conn[1]]).reshape(target.shape[0], target.shape[1], 1, 3)), 2
            )

        return torch.mean(torch.norm(pred_bones - gt_bones, dim=len(gt_bones.shape) - 1, p=norm))

    # Combine both joint and bone loss functions
    elif loss_fn == "joint_bone":

        # Joint
        # pred_joint, gt_joint = torch.tensor([]).to(device), torch.tensor([]).to(device)
        # for i in range(0, 17):
        #     pred_joint = torch.cat((pred_joint, predicted[:, :, i] - predicted[:, :, 0]), 0)
        #     gt_joint = torch.cat((gt_joint, target[:, :, i] - target[:, :, 0]), 0)
        
        # pred_joint = torch.reshape(pred_joint, predicted.shape)
        # gt_joint = torch.reshape(gt_joint, target.shape)
        
        pred_joint, gt_joint = torch.tensor([]).to(device), torch.tensor([]).to(device)
        for i in range(0, 17):

            pred_joint = torch.cat(
                (pred_joint, (predicted[:, :, i] - predicted[:, :, 0]).reshape(predicted.shape[0], predicted.shape[1], 1, 3)), 2
            )
            gt_joint = torch.cat(
                (gt_joint, (target[:, :, i] - target[:, :, 0]).reshape(target.shape[0], target.shape[1], 1, 3)), 2
            )
        
        loss_joint = torch.mean(torch.norm(pred_joint - gt_joint, dim=len(gt_joint.shape) - 1, p=norm))

        # Bone
        pred_bones, gt_bones = torch.tensor([]).to(device), torch.tensor([]).to(device)
        for conn in connections:
            pred_bones = torch.cat(
                (pred_bones, (predicted[:, :, conn[0]] - predicted[:, :, conn[1]]).reshape(predicted.shape[0], predicted.shape[1], 1, 3)), 2
            )
            gt_bones = torch.cat(
                (gt_bones, (target[:, :, conn[0]] - target[:, :, conn[1]]).reshape(target.shape[0], target.shape[1], 1, 3)), 2
            )
        # pred_bones, gt_bones = torch.tensor([]).to(device), torch.tensor([]).to(device)
        # for conn in connections:
        #     pred_bones = torch.cat((pred_bones, predicted[:, :, conn[0]] - predicted[:, :, conn[1]]), 0)
        #     gt_bones = torch.cat((gt_bones, target[:, :, conn[0]] - target[:, :, conn[1]]), 0)

        # pred_bones = torch.reshape(pred_bones, (predicted.shape[0], predicted.shape[1], -1, 3))
        # gt_bones = torch.reshape(gt_bones, (target.shape[0], target.shape[1], -1, 3))

        loss_bone = torch.mean(torch.norm(pred_bones - gt_bones, dim=len(gt_bones.shape) - 1, p=norm))

        return (loss_joint + loss_bone)
    
    elif loss_fn == "allpairs":
        pred_all, gt_all = torch.tensor([]).to(device), torch.tensor([]).to(device)
        for i in range(0, 17):
            for j in range(i + 1, 17):
                pred_all = torch.cat(
                    (pred_all, (predicted[:, :, i] - predicted[:, :, j]).reshape(predicted.shape[0], predicted.shape[1], 1, 3)), 2
                )
                gt_all = torch.cat(
                    (gt_all, (target[:, :, i] - target[:, :, j]).reshape(target.shape[0], target.shape[1], 1, 3)), 2
                )
                # pred_all = torch.cat((pred_all, predicted[:, :, i] - predicted[:, :, j]), 0)
                # gt_all = torch.cat((gt_all, target[:, :, i] - target[:, :, j]), 0)

        # pred_all = torch.reshape(pred_all, (predicted.shape[0], predicted.shape[1], -1, 3))
        # gt_all = torch.reshape(gt_all, (target.shape[0], target.shape[1], -1, 3))

        return torch.mean(torch.norm(pred_all - gt_all, dim=len(gt_all.shape) - 1, p=norm))

    elif loss_fn == "mpjpe_allpairs":
        
        # mpjpe
        loss_mpjpe = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1, p=2))

        # all bone pairs
        pred_all, gt_all = torch.tensor([]).to(device), torch.tensor([]).to(device)
        for i in range(0, 17):
            for j in range(i + 1, 17):
                pred_all = torch.cat(
                    (pred_all, (predicted[:, :, i] - predicted[:, :, j]).reshape(predicted.shape[0], predicted.shape[1], 1, 3)), 2
                )
                gt_all = torch.cat(
                    (gt_all, (target[:, :, i] - target[:, :, j]).reshape(target.shape[0], target.shape[1], 1, 3)), 2
                )
        #         pred_all = torch.cat((pred_all, predicted[:, :, i] - predicted[:, :, j]), 0)
        #         gt_all = torch.cat((gt_all, target[:, :, i] - target[:, :, j]), 0)

        # pred_all = torch.reshape(pred_all, (predicted.shape[0], predicted.shape[1], -1, 3))
        # gt_all = torch.reshape(gt_all, (target.shape[0], target.shape[1], -1, 3))

        loss_allpairs = torch.mean(torch.norm(pred_all - gt_all, dim=len(gt_all.shape) - 1, p=1))

        return (loss_mpjpe + loss_allpairs)

def test_calculation(predicted, target, action, error_sum, data_type, subject):
    error_sum = mpjpe_by_action_p1(predicted, target, action, error_sum)
    error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum)

    return error_sum


def mpjpe_by_action_p1(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=len(target.shape) - 2)

    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]

        action_error_sum[action_name]['p1'].update(torch.mean(dist).item()*num, num)
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]

            action_error_sum[action_name]['p1'].update(dist[i].item(), 1)
            
    return action_error_sum

def mpjpe_by_action_p2(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    pred = predicted.detach().cpu().numpy().reshape(-1, predicted.shape[-2], predicted.shape[-1])
    gt = target.detach().cpu().numpy().reshape(-1, target.shape[-2], target.shape[-1])
    dist = p_mpjpe(pred, gt)

    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]
        action_error_sum[action_name]['p2'].update(np.mean(dist) * num, num)
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]
            action_error_sum[action_name]['p2'].update(np.mean(dist), 1)
            
    return action_error_sum


def p_mpjpe(predicted, target):
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY 
    t = muX - a * np.matmul(muY, R)

    predicted_aligned = a * np.matmul(predicted, R) + t

    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=len(target.shape) - 2)


def define_actions( action ):

  actions = ["Directions","Discussion","Eating","Greeting",
           "Phoning","Photo","Posing","Purchases",
           "Sitting","SittingDown","Smoking","Waiting",
           "WalkDog","Walking","WalkTogether"]

  if action == "All" or action == "all" or action == '*':
    return actions

  if not action in actions:
    raise( ValueError, "Unrecognized action: %s" % action )

  return [action]


def define_error_list(actions):
    error_sum = {}
    error_sum.update({actions[i]: 
        {'p1':AccumLoss(), 'p2':AccumLoss()} 
        for i in range(len(actions))})
    return error_sum


class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
        

def get_varialbe(split, target):
    num = len(target)
    var = []
    if split == 'train':
        for i in range(num):
            temp = Variable(target[i], requires_grad=False).contiguous().type(torch.cuda.FloatTensor)
            var.append(temp)
    else:
        for i in range(num):
            temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor)
            var.append(temp)

    return var


def print_error(data_type, action_error_sum, is_train):
    mean_error_p1, mean_error_p2 = print_error_action(action_error_sum, is_train)

    return mean_error_p1, mean_error_p2


def print_error_action(action_error_sum, is_train):
    mean_error_each = {'p1': 0.0, 'p2': 0.0}
    mean_error_all  = {'p1': AccumLoss(), 'p2': AccumLoss()}

    if is_train == 0:
        print("{0:=^12} {1:=^10} {2:=^8}".format("Action", "p#1 mm", "p#2 mm"))


    for action, value in action_error_sum.items():
        if is_train == 0:
            print("{0:<12} ".format(action), end="")

        mean_error_each['p1'] = action_error_sum[action]['p1'].avg * 1000.0
        mean_error_all['p1'].update(mean_error_each['p1'], 1)

        mean_error_each['p2'] = action_error_sum[action]['p2'].avg * 1000.0
        mean_error_all['p2'].update(mean_error_each['p2'], 1)

        if is_train == 0:
            print("{0:>6.2f} {1:>10.2f}".format(mean_error_each['p1'], mean_error_each['p2']))

    if is_train == 0:
        print("{0:<12} {1:>6.2f} {2:>10.2f}".format("Average", mean_error_all['p1'].avg, \
                mean_error_all['p2'].avg))
    
    return mean_error_all['p1'].avg, mean_error_all['p2'].avg


def save_model(previous_name, save_dir, epoch, data_threshold, model):
    if os.path.exists(previous_name):
        os.remove(previous_name)

    torch.save(model.state_dict(), '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100))

    previous_name = '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100)

    return previous_name
    

def save_model_epoch(save_dir, epoch, model):
    torch.save(model.state_dict(), '%s/epoch_%d.pth' % (save_dir, epoch))
