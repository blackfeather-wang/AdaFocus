import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


def visualize(images, action_sequence, confidence_sequence, correct_sequence, image_size, patch_size_sequence, target, save_path):
    images = _image_restore(images)
    target = target.cpu().item()
    batch_size = images.size(0) # actually is time sequence length
    maximum_length = len(confidence_sequence)
    patch_coordinate_sequence = []
    patches = []
    
    # ft = ImageFont.truetype("./ops/arialuni.ttf", 20)

    for i in range(0, maximum_length):
        if i < maximum_length: 
            action = torch.floor(action_sequence[i] * (image_size - patch_size_sequence[i])).int()
            max_coord = (image_size - patch_size_sequence[i] - 1)*torch.ones_like(action)
            patch_coordinate_sequence.append(torch.where(action >= image_size - patch_size_sequence[i] - 1, max_coord, action).int())

        # y,x = 0,0
        # confidence = confidence_sequence[0][i]
        # if correct_sequence[0][i]==True:
        #     patch_color, is_correct = (0,255,0), True
        # else:
        #     patch_color, is_correct = (255,0,0), False

        # label = target[i]
        # img_draw.text((x+4,y-6), '%d - ' % (label) + class_dict[label].split(',')[0], fill=(0,0,0))

        # y += 20
        # img_draw.polygon([(x,y), (x, y+image_size-1), (x+image_size-1, y+image_size-1), (x+image_size-1,y)], outline=patch_color)
        # img_draw.polygon([(x,y), (x, y+20), (x+80, y+20), (x+80,y)], outline=patch_color, fill=patch_color)

        # if confidence >=0.9995:
        #     img_draw.text((x+1,y-4), '%d: 100' % (1) + '%', fill=(0,0,0))
        # else:
        #     img_draw.text((x+1,y-4), '%d: %.1f' % (1, 100*confidence) + '%', fill=(0,0,0))

    for patch_step in range(0, maximum_length):
        per_patch = (255*images[patch_step]).cpu().numpy().astype(np.uint8)
        white_top = 255*np.ones((3,20,image_size)).astype(np.uint8)
        per_patch = np.concatenate((white_top,per_patch), axis=1)
        per_patch = Image.fromarray(per_patch.transpose([1,2,0])).convert('RGB')
        img_draw = ImageDraw.Draw(per_patch)
        if patch_step == 0:
            label = target
            img_draw.text((4,6), '%d - ' % (label) + class_dict[label].split(',')[0], fill=(0,0,0))
        is_correct = correct_sequence[patch_step]
        if correct_sequence[patch_step]==True:
            patch_color, is_correct = (0,255,0), True
        else:
            patch_color = (255,0,0)
        # if is_correct == False:
        if True:
            coords = patch_coordinate_sequence[patch_step]
            y,x = coords[0],coords[1]
            confidence = confidence_sequence[patch_step]

            y += 20
            img_draw.polygon([(x,y), (x, y+patch_size_sequence[patch_step]), (x+patch_size_sequence[patch_step], y+patch_size_sequence[patch_step]), (x+patch_size_sequence[patch_step],y)], outline=patch_color)
            img_draw.polygon([(x,y), (x, y+20), (x+80, y+20), (x+80,y)], outline=patch_color,fill=patch_color)

            if confidence >= 0.9995:
                img_draw.text((x+1,y+4), '%d: 100' % (patch_step+1) + '%', fill=(0,0,0))
            else:
                img_draw.text((x+1,y+4), '%d: %.1f' % (patch_step+1, 100*confidence) + '%', fill=(0,0,0))

        per_patch = np.array(per_patch).transpose([2,0,1])
        patches.append(per_patch.reshape(1, per_patch.shape[0], per_patch.shape[1], per_patch.shape[2]))

    visualized_images = np.concatenate(patches, axis=0)
    save_images(visualized_images, save_path)



def _image_restore(images):
    mask_mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).expand(images.size(0), 3, images.size(2), images.size(3)).cuda()
    mask_std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).expand(images.size(0), 3, images.size(2), images.size(3)).cuda()
    return images.mul(mask_std) + mask_mean


def save_images(images, save_path):
    if isinstance(images.flatten()[0], np.floating):
        images = (255.99*images).astype('uint8')

    n_samples = images.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1
    nh, nw = rows, int(n_samples/rows)

    images = images.transpose(0,2,3,1)
    h, w = images[0].shape[:2]
    visualized_image = np.zeros((h*nh, w*nw, 3))

    for n, img in enumerate(images):
        j = int(n/nw)
        i = n%nw
        visualized_image[j*h:j*h+h, i*w:i*w+w] = img

    visualized_image = Image.fromarray(visualized_image.astype('uint8'))
    visualized_image.save(save_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_multi_hot(test_y, classes, assumes_starts_zero=True):
    bs = test_y.shape[0]
    label_cnt = 0

    # TODO ranking labels: (-1,-1,4,5,3,7)->(4,4,2,1,0,3)
    if not assumes_starts_zero:
        for label_val in torch.unique(test_y):
            if label_val >= 0:
                test_y[test_y == label_val] = label_cnt
                label_cnt += 1

    gt = torch.zeros(bs, classes + 1)  # TODO(yue) +1 for -1 in multi-label case
    for i in range(test_y.shape[1]):
        gt[torch.LongTensor(range(bs)), test_y[:, i]] = 1  # TODO(yue) see?

    return gt[:, :classes]


def cal_map(output, old_test_y):
    batch_size = output.size(0)
    num_classes = output.size(1)
    ap = torch.zeros(num_classes)
    test_y = old_test_y.clone()

    gt = get_multi_hot(test_y, num_classes, False)

    probs = F.softmax(output, dim=1)

    rg = torch.range(1, batch_size).float()
    for k in range(num_classes):
        scores = probs[:, k]
        targets = gt[:, k]
        _, sortind = torch.sort(scores, 0, True)
        truth = targets[sortind]
        tp = truth.float().cumsum(0)
        precision = tp.div(rg)

        ap[k] = precision[truth.byte()].sum() / max(float(truth.sum()), 1)
    return ap.mean() * 100, ap * 100

def cal_reward(confidence, confidence_last, patch_size_list, penalty=0.5):
    reward = confidence - confidence_last
    reward = reward - penalty*(patch_size_list/100.)**2
    return reward

class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        out = '\t'.join(entries)
        print(out)
        return out + '\n'

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class Recorder:
    def __init__(self, larger_is_better=True):
        self.history = []
        self.larger_is_better = larger_is_better
        self.best_at = None
        self.best_val = None

    def is_better_than(self, x, y):
        if self.larger_is_better:
            return x > y
        else:
            return x < y

    def update(self, val):
        self.history.append(val)
        if len(self.history) == 1 or self.is_better_than(val, self.best_val):
            self.best_val = val
            self.best_at = len(self.history) - 1

    def is_current_best(self):
        return self.best_at == len(self.history) - 1


def get_mobv2_new_sd(old_sd, reverse=False):
    # TODO(DEBUG)
    pairs = [
        ["conv.0.0", "conv.0"],
        ["conv.0.1", "conv.1"],
        ["conv.1.0", "conv.3"],
        ["conv.1.1", "conv.4"],
        ["conv.2", "conv.6"],
        ["conv.3", "conv.7"],
        ["conv.1", "conv.3"],
        # ["conv.2", "conv.4"],
    ]

    old_keys_del = []
    new_keys_add = []
    if reverse:
        for old_key in old_sd.keys():
            if "features.0" not in old_key:
                for pair in pairs:
                    if pair[1] in old_key:
                        if pair[1] in ["conv.4", "conv.3"]:
                            idx = int(pair[1].split(".")[1])
                            if "features.1." in old_key:
                                new_key = old_key.replace(pair[1], "conv.%d" % (idx - 2))
                            else:
                                new_key = old_key.replace(pair[1], "conv.1.%d" % (idx - 3))
                        else:
                            new_key = old_key.replace(pair[1], pair[0])
                        old_keys_del.append(old_key)
                        new_keys_add.append(new_key)
                        break
    else:
        for old_key in old_sd.keys():
            if "features.0" not in old_key:
                for pair in pairs:
                    if pair[0] in old_key:
                        if pair[0] == "conv.2" and "features.1." in old_key:
                            new_key = old_key.replace(pair[0], "conv.4")
                        else:
                            new_key = old_key.replace(pair[0], pair[1])
                        old_keys_del.append(old_key)
                        new_keys_add.append(new_key)
                        break
    state_dict_2d_new = {}
    for key in old_sd:
        if key not in old_keys_del:
            state_dict_2d_new[key] = old_sd[key]

    for i in range(len(old_keys_del)):
        state_dict_2d_new[new_keys_add[i]] = old_sd[old_keys_del[i]]

    return state_dict_2d_new

class_dict = {
0: 'Applying sunscreen',
1: 'Archery',
2: 'Arm wrestling',
3: 'Assembling bicycle',
4: 'BMX',
5: 'Baking cookies',
6: 'Ballet',
7: 'Bathing dog',
8: 'Baton twirling',
9: 'Beach soccer',
10: 'Beer pong',
11: 'Belly dance',
12: 'Blow-drying hair',
13: 'Blowing leaves',
14: 'Braiding hair',
15: 'Breakdancing',
16: 'Brushing hair',
17: 'Brushing teeth',
18: 'Building sandcastles',
19: 'Bullfighting',
20: 'Bungee jumping',
21: 'Calf roping',
22: 'Camel ride',
23: 'Canoeing',
24: 'Capoeira',
25: 'Carving jack-o-lanterns',
26: 'Changing car wheel',
27: 'Cheerleading',
28: 'Chopping wood',
29: 'Clean and jerk',
30: 'Cleaning shoes',
31: 'Cleaning sink',
32: 'Cleaning windows',
33: 'Clipping cat claws',
34: 'Cricket',
35: 'Croquet',
36: 'Cumbia',
37: 'Curling',
38: 'Cutting the grass',
39: 'Decorating the Christmas tree',
40: 'Disc dog',
41: 'Discus throw',
42: 'Dodgeball',
43: 'Doing a powerbomb',
44: 'Doing crunches',
45: 'Doing fencing',
46: 'Doing karate',
47: 'Doing kickboxing',
48: 'Doing motocross',
49: 'Doing nails',
50: 'Doing step aerobics',
51: 'Drinking beer',
52: 'Drinking coffee',
53: 'Drum corps',
54: 'Elliptical trainer',
55: 'Fixing bicycle',
56: 'Fixing the roof',
57: 'Fun sliding down',
58: 'Futsal',
59: 'Gargling mouthwash',
60: 'Getting a haircut',
61: 'Getting a piercing',
62: 'Getting a tattoo',
63: 'Grooming dog',
64: 'Grooming horse',
65: 'Hammer throw',
66: 'Hand car wash',
67: 'Hand washing clothes',
68: 'Hanging wallpaper',
69: 'Having an ice cream',
70: 'High jump',
71: 'Hitting a pinata',
72: 'Hopscotch',
73: 'Horseback riding',
74: 'Hula hoop',
75: 'Hurling',
76: 'Ice fishing',
77: 'Installing carpet',
78: 'Ironing clothes',
79: 'Javelin throw',
80: 'Kayaking',
81: 'Kite flying',
82: 'Kneeling',
83: 'Knitting',
84: 'Laying tile',
85: 'Layup drill in basketball',
86: 'Long jump',
87: 'Longboarding',
88: 'Making a cake',
89: 'Making a lemonade',
90: 'Making a sandwich',
91: 'Making an omelette',
92: 'Mixing drinks',
93: 'Mooping floor',
94: 'Mowing the lawn',
95: 'Paintball',
96: 'Painting',
97: 'Painting fence',
98: 'Painting furniture',
99: 'Peeling potatoes',
100: 'Ping-pong',
101: 'Plastering',
102: 'Plataform diving',
103: 'Playing accordion',
104: 'Playing badminton',
105: 'Playing bagpipes',
106: 'Playing beach volleyball',
107: 'Playing blackjack',
108: 'Playing congas',
109: 'Playing drums',
110: 'Playing field hockey',
111: 'Playing flauta',
112: 'Playing guitarra',
113: 'Playing harmonica',
114: 'Playing ice hockey',
115: 'Playing kickball',
116: 'Playing lacrosse',
117: 'Playing piano',
118: 'Playing polo',
119: 'Playing pool',
120: 'Playing racquetball',
121: 'Playing rubik cube',
122: 'Playing saxophone',
123: 'Playing squash',
124: 'Playing ten pins',
125: 'Playing violin',
126: 'Playing water polo',
127: 'Pole vault',
128: 'Polishing forniture',
129: 'Polishing shoes',
130: 'Powerbocking',
131: 'Preparing pasta',
132: 'Preparing salad',
133: 'Putting in contact lenses',
134: 'Putting on makeup',
135: 'Putting on shoes',
136: 'Rafting',
137: 'Raking leaves',
138: 'Removing curlers',
139: 'Removing ice from car',
140: 'Riding bumper cars',
141: 'River tubing',
142: 'Rock climbing',
143: 'Rock-paper-scissors',
144: 'Rollerblading',
145: 'Roof shingle removal',
146: 'Rope skipping',
147: 'Running a marathon',
148: 'Sailing',
149: 'Scuba diving',
150: 'Sharpening knives',
151: 'Shaving',
152: 'Shaving legs',
153: 'Shot put',
154: 'Shoveling snow',
155: 'Shuffleboard',
156: 'Skateboarding',
157: 'Skiing',
158: 'Slacklining',
159: 'Smoking a cigarette',
160: 'Smoking hookah',
161: 'Snatch',
162: 'Snow tubing',
163: 'Snowboarding',
164: 'Spinning',
165: 'Spread mulch',
166: 'Springboard diving',
167: 'Starting a campfire',
168: 'Sumo',
169: 'Surfing',
170: 'Swimming',
171: 'Swinging at the playground',
172: 'Table soccer',
173: 'Tai chi',
174: 'Tango',
175: 'Tennis serve with ball bouncing',
176: 'Throwing darts',
177: 'Trimming branches or hedges',
178: 'Triple jump',
179: 'Tug of war',
180: 'Tumbling',
181: 'Using parallel bars',
182: 'Using the balance beam',
183: 'Using the monkey bar',
184: 'Using the pommel horse',
185: 'Using the rowing machine',
186: 'Using uneven bars',
187: 'Vacuuming floor',
188: 'Volleyball',
189: 'Wakeboarding',
190: 'Walking the dog',
191: 'Washing dishes',
192: 'Washing face',
193: 'Washing hands',
194: 'Waterskiing',
195: 'Waxing skis',
196: 'Welding',
197: 'Windsurfing',
198: 'Wrapping presents',
199: 'Zumba'
}