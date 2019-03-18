import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets, transforms


# USE_CUDA = True
#
#
# class Mnist:
#     def __init__(self, batch_size):
#         dataset_transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])
#
#         train_dataset = datasets.MNIST('../data', train=True, download=True, transform=dataset_transform)
#         test_dataset = datasets.MNIST('../data', train=False, download=True, transform=dataset_transform)
#
#         self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
#

# # define a class
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

def EntryNodeOfLoop(pHead):
    # write code here
    head = ListNode(0)
    head.next = pHead
    fast = head
    slow = head
    # while True:
    #     if fast and fast.next:
    #         fast = fast.next.next
    #         slow = slow.next
    #     else:
    #         return None
    #     if fast == slow:
    #         fast = head
    #         break
    fast = fast.next
    while fast:
        print(fast.val)
        fast = fast.next


    while fast != slow:
        fast = fast.next
        slow = slow.next
    return fast


p = ListNode(1)
p.next = ListNode(2)
p.next.next = ListNode(3)
p.next.next.next = ListNode(4)
p.next.next.next.next = ListNode(5)
p.next.next.next.next.next = ListNode(6)

EntryNodeOfLoop(p)
