# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch.utils.data as data

from PIL import Image
import os
import numpy as np
from numpy.random import randint


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file, num_segments_glancer=8,
                 num_segments_focuser=8, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, twice_sample=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments_glancer = num_segments_glancer
        self.num_segments_focuser = num_segments_focuser
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        # print('self.test_mode:', self.test_mode)
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
        elif self.modality == 'Flow':
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':  # ucf
                x_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('x', idx))).convert(
                    'L')
                y_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('y', idx))).convert(
                    'L')
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':  # something v1 flow
                x_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'x', idx))).convert('L')
                y_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'y', idx))).convert('L')
            else:
                try:
                    # idx_skip = 1 + (idx-1)*5
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert(
                        'RGB')
                except Exception:
                    print('error loading flow file:',
                          os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')
                # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
                flow_x, flow_y, _ = flow.split()
                x_img = flow_x.convert('L')
                y_img = flow_y.convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        # check the frame number is large >3:
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        if not self.test_mode or self.remove_missing:
            tmp = [item for item in tmp if int(item[1]) >= 3]
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
        ### For glancer
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments_glancer
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments_glancer)]
            offsets_glancer = np.array(offsets) + 1
        else:  # normal sample
            # print('normal sample')
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments_glancer # RGB self.new_length=1
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments_glancer)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments_glancer)
            elif record.num_frames > self.num_segments_glancer:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments_glancer))
            else:
                offsets = np.zeros((self.num_segments_glancer,))
            offsets_glancer = offsets + 1

        ### For focuser
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments_focuser
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments_focuser)]
            offsets_focuser = np.array(offsets) + 1
        else:  # normal sample
            # print('normal sample')
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments_focuser # RGB self.new_length=1
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments_focuser)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments_focuser)
            elif record.num_frames > self.num_segments_focuser:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments_focuser))
            else:
                offsets = np.zeros((self.num_segments_focuser,))
            offsets_focuser = offsets + 1

        return offsets_glancer, offsets_focuser

    def _get_val_indices(self, record):

        ### For glancer
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments_glancer
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments_glancer)]
            offsets_glancer = np.array(offsets) + 1
        else:
            if record.num_frames > self.num_segments_glancer + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments_glancer)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments_glancer)])
            else:
                offsets = np.zeros((self.num_segments_glancer,))
            offsets_glancer = offsets + 1

        ### For focuser
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments_focuser
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments_focuser)]
            offsets_focuser = np.array(offsets) + 1
        else:
            if record.num_frames > self.num_segments_focuser + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments_focuser)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments_focuser)])
            else:
                offsets = np.zeros((self.num_segments_focuser,))
            offsets_focuser = offsets + 1

        return offsets_glancer, offsets_focuser

    def _get_test_indices(self, record):

        ### For glancer
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments_glancer
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments_glancer)]
            offsets_glancer = np.array(offsets) + 1
        elif self.twice_sample:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments_glancer)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments_glancer)] +
                               [int(tick * x) for x in range(self.num_segments_glancer)])

            offsets_glancer = offsets + 1
        else:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments_glancer)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments_glancer)])
            offsets_glancer = offsets + 1

        ### For glancer
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments_focuser
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in
                            range(self.num_segments_focuser)]
            offsets_focuser = np.array(offsets) + 1
        elif self.twice_sample:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments_focuser)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments_focuser)] +
                               [int(tick * x) for x in range(self.num_segments_focuser)])

            offsets_focuser = offsets + 1
        else:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments_focuser)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments_focuser)])
            offsets_focuser = offsets + 1

        return offsets_glancer, offsets_focuser

    def __getitem__(self, index):
        # print('get item')
        record = self.video_list[index]
        # check this is a legit video folder

        if self.image_tmpl == 'flow_{}_{:05d}.jpg':
            file_name = self.image_tmpl.format('x', 1)
            full_path = os.path.join(self.root_path, record.path, file_name)
        elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            file_name = self.image_tmpl.format(int(record.path), 'x', 1)
            full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
        else:
            file_name = self.image_tmpl.format(1)
            full_path = os.path.join(self.root_path, record.path, file_name)

        while not os.path.exists(full_path):
            print('################## Not Found:', os.path.join(self.root_path, record.path, file_name))
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':
                file_name = self.image_tmpl.format('x', 1)
                full_path = os.path.join(self.root_path, record.path, file_name)
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
                file_name = self.image_tmpl.format(int(record.path), 'x', 1)
                full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
            else:
                file_name = self.image_tmpl.format(1)
                full_path = os.path.join(self.root_path, record.path, file_name)

        # print('record:', record)
        if not self.test_mode:
            # print('test model False')
            segment_indices_glancer, segment_indices_focuser = self._sample_indices(record) \
                if self.random_shift else self._get_val_indices(record)
        else:
            # print('test model True')
            segment_indices_glancer, segment_indices_focuser = self._get_test_indices(record)

        return self.get(record, segment_indices_glancer, segment_indices_focuser)

    def get(self, record, indices_glancer, indices_focuser):

        images_glancer = list()
        for seg_ind in indices_glancer:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images_glancer.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data_glancer = self.transform(images_glancer)

        images_focuser = list()
        for seg_ind in indices_focuser:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images_focuser.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data_focuser = self.transform(images_focuser)
        return process_data_glancer, process_data_focuser, record.label

    def __len__(self):
        return len(self.video_list)
