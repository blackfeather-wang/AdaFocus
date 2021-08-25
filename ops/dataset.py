import torch.utils.data as data
import torch

from PIL import Image
import os
import numpy as np
from numpy.random import randint


class VideoRecord(object):
    def __init__(self, row):
        self._data = row
        self._labels = torch.tensor([-1, -1, -1])
        labels = sorted(list(set([int(x) for x in self._data[2:]])))
        for i, l in enumerate(labels):
            self._labels[i] = l

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        if self._labels[-2] > -1:
            if self._labels[-1] > -1:
                return self._labels[torch.randperm(self._labels.shape[0])]
            else:
                if torch.rand(1) > 0.5:
                    return self._labels[[0,1,2]]
                else:
                    return self.label[[1,0,2]]
        else:
            return self._labels


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, twice_sample=False,
                 dataset=None, partial_fcvid_eval=False, partial_ratio=None,
                 ada_reso_skip=False, reso_list=None, random_crop=False, center_crop=False, ada_crop_list=None,
                 rescale_to=224, policy_input_offset=None, save_meta=False):

        self.root_path = root_path

        self.list_file = \
            ".".join(list_file.split(".")[:-1]) + "." + list_file.split(".")[-1]  # TODO
        self.num_segments = num_segments
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation

        # TODO(yue)
        self.dataset = dataset
        self.partial_fcvid_eval = partial_fcvid_eval
        self.partial_ratio = partial_ratio
        self.ada_reso_skip = ada_reso_skip
        self.reso_list = reso_list
        self.random_crop = random_crop
        self.center_crop = center_crop
        self.ada_crop_list = ada_crop_list
        self.rescale_to = rescale_to
        self.policy_input_offset = policy_input_offset
        self.save_meta = save_meta

        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        self._parse_list()

    def _load_image(self, directory, idx):
        try:
            return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
        except Exception:
            print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
            return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]

    def _parse_list(self):
        # check the frame number is large >3:
        splitter = "," if self.dataset in ["actnet", "fcvid"] else " "
        if self.dataset == "kinetics":
            splitter = ";"
        tmp = [x.strip().split(splitter) for x in open(self.list_file)]

        if any(len(items) >= 3 for items in tmp) and self.dataset == "minik":
            tmp = [[splitter.join(x[:-2]), x[-2], x[-1]] for x in tmp]

        if self.dataset == "kinetics":
            tmp = [[x[0], x[-2], x[-1]] for x in tmp]

        if not self.test_mode or self.remove_missing:
            tmp = [item for item in tmp if int(item[1]) >= 3]

        if self.partial_fcvid_eval and self.dataset == "fcvid":
            tmp = tmp[:int(len(tmp) * self.partial_ratio)]

        self.video_list = [VideoRecord(item) for item in tmp]

        if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            for v in self.video_list:
                v._data[1] = int(v._data[1]) / 2
        print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:  # normal sample
            average_duration = record.num_frames // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(randint(record.num_frames, size=self.num_segments))
            else:
                offsets = np.array(
                    list(range(record.num_frames)) + [record.num_frames - 1] * (self.num_segments - record.num_frames))
            return offsets + 1

    def _get_val_indices(self, record):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            if record.num_frames > self.num_segments:
                tick = record.num_frames / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.array(
                    list(range(record.num_frames)) + [record.num_frames - 1] * (self.num_segments - record.num_frames))
            return offsets + 1

    def _get_test_indices(self, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        elif self.twice_sample:
            tick = record.num_frames / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
                               [int(tick * x) for x in range(self.num_segments)])

            return offsets + 1
        else:
            tick = record.num_frames / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder
        if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            file_name = self.image_tmpl.format(int(record.path), 'x', 1)
            full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
        else:
            file_name = self.image_tmpl.format(1)
            full_path = os.path.join(self.root_path, record.path, file_name)

        err_cnt = 0
        while not os.path.exists(full_path):
            err_cnt += 1
            if err_cnt > 3:
                exit("Sth wrong with the dataloader to get items. Check your data path. Exit...")
            print('################## Not Found:', os.path.join(self.root_path, record.path, file_name))
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]
            if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
                file_name = self.image_tmpl.format(int(record.path), 'x', 1)
                full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
            else:
                file_name = self.image_tmpl.format(1)
                full_path = os.path.join(self.root_path, record.path, file_name)

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            images.extend(self._load_image(record.path, int(seg_ind)))

        process_data = self.transform(images)
        if self.ada_reso_skip:
            return_items = [process_data]
            if self.random_crop:
                rescaled = [self.random_crop_proc(process_data, (x, x)) for x in self.reso_list[1:]]
            elif self.center_crop:
                rescaled = [self.center_crop_proc(process_data, (x, x)) for x in self.reso_list[1:]]
            else:
                rescaled = [self.rescale_proc(process_data, (x, x)) for x in self.reso_list[1:]]
            return_items = return_items + rescaled
            if self.save_meta:
                return_items = return_items + [record.path] + [indices]  # [torch.tensor(indices)]
            return_items = return_items + [record.label]

            return tuple(return_items)
        else:
            if self.rescale_to == 224:
                rescaled = process_data
            else:
                x = self.rescale_to
                if self.random_crop:
                    rescaled = self.random_crop_proc(process_data, (x, x))
                elif self.center_crop:
                    rescaled = self.center_crop_proc(process_data, (x, x))
                else:
                    rescaled = self.rescale_proc(process_data, (x, x))

            return rescaled, record.label

    # TODO(yue)
    # (NC, H, W)->(NC, H', W')
    def rescale_proc(self, input_data, size):
        return torch.nn.functional.interpolate(input_data.unsqueeze(1), size=size, mode='nearest').squeeze(1)

    def center_crop_proc(self, input_data, size):
        h = input_data.shape[1] // 2
        w = input_data.shape[2] // 2
        return input_data[:, h - size[0] // 2:h + size[0] // 2, w - size[1] // 2:w + size[1] // 2]

    def random_crop_proc(self, input_data, size):
        H = input_data.shape[1]
        W = input_data.shape[2]
        input_data_nchw = input_data.view(-1, 3, H, W)
        batchsize = input_data_nchw.shape[0]
        return_list = []
        hs0 = np.random.randint(0, H - size[0], batchsize)
        ws0 = np.random.randint(0, W - size[1], batchsize)
        for i in range(batchsize):
            return_list.append(input_data_nchw[i, :, hs0[i]:hs0[i] + size[0], ws0[i]:ws0[i] + size[1]])
        return torch.stack(return_list).view(batchsize * 3, size[0], size[1])

    def __len__(self):
        return len(self.video_list)