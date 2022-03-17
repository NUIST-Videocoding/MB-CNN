import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image
from scipy.signal import convolve2d
import h5py
from pylab import *


def default_loader(path):
    return Image.open(path).convert('L')


def LocalNormalization(patch, P=3, Q=3, C=1):
    kernel = np.ones((P, Q)) / (P * Q)
    patch_mean = convolve2d(patch, kernel, boundary='symm', mode='same')
    patch_sm = convolve2d(np.square(patch), kernel, boundary='symm', mode='same')
    patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C
    patch_ln = torch.from_numpy((patch - patch_mean) / patch_std).float().unsqueeze(0)
    return patch_ln


def OverlappingCropPatches(im, patch_size=32, stride=32):
    w, h = im.size
    centerx = int(w/2) + 1
    centery = int(h/2) + 1
    vecenter = np.array([centerx, centery])
    patches_dis = ()
    distance = ()
    entropy = ()
    for i in range(0, h - stride, stride):
        for j in range(0, w - stride, stride):
            imm = im.crop((j, i, j + patch_size, i + patch_size))
            patch = to_tensor(imm)
            patch_dis = LocalNormalization(patch[0].numpy())
            x1 = j + 1
            y1 = i + 1
            vec1 = np.array([x1, y1])
            x2 = j + 1
            y2 = i + 32
            vec2 = np.array([x2, y2])
            x3 = j + 32
            y3 = i + 1
            vec3 = np.array([x3, y3])
            x4 = j + 32
            y4 = i + 32
            vec4 = np.array([x4, y4])
            x_cen = int(x4 / 2) + 1
            y_cen = int(y4 / 2) + 1
            vec_cen = np.array([x_cen, y_cen])
            dist1 = np.linalg.norm(vec1 - vecenter)
            dist2 = np.linalg.norm(vec2 - vecenter)
            dist3 = np.linalg.norm(vec3 - vecenter)
            dist4 = np.linalg.norm(vec4 - vecenter)
            dist5 = np.linalg.norm(vec_cen - vecenter)
            dist = torch.tensor([[[dist1, dist2, dist3, dist4, dist5]]])

            patches_dis = patches_dis + (patch_dis,)
            distance = distance + (dist,)
            entropyy = ComputerEntropy(imm)
            entropy = entropy + (entropyy,)

    return patches_dis, distance, entropy


def ComputerEntropy(im):
    tmp = []
    for i in range(256):
        tmp.append(0)
    val = 0
    k = 0
    res = 0
    img = np.array(im)
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[val] = float(tmp[val]+1)
            k = float(k+1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i]/k)
    for i in range(len(tmp)):
        if(tmp[i] == 0):
            res = res
        else:
            res = float(res-tmp[i]*(np.log2(tmp[i])))
    res = torch.tensor([[[res]]])
    return res


def OverlappingCropPatches_gradient(im, patch_size=32, stride=32):
    w, h = im.size
    patches_gradient = ()
    for i in range(0, h - stride, stride):
        for j in range(0, w - stride, stride):
            patch = to_tensor(im.crop((j, i, j + patch_size, i + patch_size)))
            patches_gradient = patches_gradient + (patch,)
    return patches_gradient


class IQADataset(Dataset):
    def __init__(self, conf, exp_id=0, status='train', loader=default_loader):
        self.loader = loader
        im_dir = conf['LIVE']['im_dir']
        self.patch_size = conf['patch_size']
        self.stride = conf['stride']
        datainfo = conf['LIVE']['datainfo']

        Info = h5py.File(datainfo)
        index = Info['index'][:, int(exp_id) % 1000]
        ref_ids = Info['ref_ids'][0, :]
        ref_ids = ref_ids[0:10]

        test_ratio = conf['test_ratio']
        train_ratio = conf['train_ratio']
        trainindex = index[:int(train_ratio * len(index))]
        testindex = index[int((1 - test_ratio) * len(index)):]
        train_index, val_index, test_index = [], [], []
        for i in range(len(ref_ids)):
            train_index.append(i) if (ref_ids[i] in trainindex) else \
                test_index.append(i) if (ref_ids[i] in testindex) else \
                    val_index.append(i)
        if status == 'train':
            self.index = train_index
            print("# Train Images: {}".format(len(self.index)))
            print('Ref Index:')
            print(trainindex)
        if status == 'test':
            self.index = test_index
            print("# Test Images: {}".format(len(self.index)))
            print('Ref Index:')
            print(testindex)
        if status == 'val':
            self.index = val_index
            print("# Val Images: {}".format(len(self.index)))

        self.mos = Info['subjective_scores'][0, self.index]
        self.mos_std = Info['subjective_scoresSTD'][0, self.index]

        #损失图
        im_names = [Info[Info['im_names'][0, :][i]].value.tobytes() \
                        [::2].decode() for i in self.index]
        #梯度图
        im_names_gradient = [Info[Info['im_names1'][0, :][i]].value.tobytes() \
                         [::2].decode() for i in self.index]

        self.patches_dis = ()
        self.patches_gradient = ()
        self.distance = ()
        self.entropy = ()
        self.label = []
        self.label_std = []
        for idx in range(len(self.index)):
            im = self.loader(os.path.join(im_dir, im_names[idx]))
            im_gradient = self.loader(os.path.join(im_dir, im_names_gradient[idx]))

            patches_dis, distance, entropy = OverlappingCropPatches(im, self.patch_size,self.stride)
            patches_gradient = OverlappingCropPatches_gradient(im_gradient, self.patch_size, self.stride)
            if status == 'train':
                self.patches_dis = self.patches_dis + patches_dis
                self.patches_gradient = self.patches_gradient + patches_gradient
                self.distance = self.distance + distance
                self.entropy = self.entropy + entropy
                for i in range(len(patches_dis)):
                    self.label.append(self.mos[idx])
                    self.label_std.append(self.mos_std[idx])

            else:
                self.patches_dis = self.patches_dis + (torch.stack(patches_dis),)
                self.patches_gradient = self.patches_gradient + (torch.stack(patches_gradient),)
                self.distance = self.distance + (torch.stack(distance),)
                self.entropy = self.entropy + (torch.stack(entropy),)
                self.label.append(self.mos[idx])
                self.label_std.append(self.mos_std[idx])

    def __len__(self):
        return len(self.patches_dis)

    def __getitem__(self, idx):
        return ((self.patches_dis[idx], self.patches_gradient[idx], self.distance[idx], self.entropy[idx]),
                (torch.Tensor([self.label[idx], ]),
                 torch.Tensor([self.label_std[idx], ])))




































