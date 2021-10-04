from os.path import join as ospj

def return_actnet(data_dir):
    filename_categories = ospj(data_dir, 'classInd.txt')
    root_data = data_dir + "/frames"
    filename_imglist_train = ospj(data_dir, 'actnet_train_split.txt')
    filename_imglist_val = ospj(data_dir, 'actnet_val_split.txt')
    prefix = 'image_{:05d}.jpg'

    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_fcvid(data_dir):
    filename_categories = ospj(data_dir, 'classInd.txt')
    root_data = data_dir + "/frames"
    filename_imglist_train = ospj(data_dir, 'fcvid_train_split.txt')
    filename_imglist_val = ospj(data_dir, 'fcvid_val_split.txt')
    prefix = 'image_{:05d}.jpg'

    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_minik(data_dir):
    filename_categories = ospj(data_dir, 'minik_classInd.txt')
    root_data = data_dir + "/frames"
    filename_imglist_train = ospj(data_dir, 'mini_train_videofolder.txt')
    filename_imglist_val = ospj(data_dir, 'mini_val_videofolder.txt')
    prefix = 'image_{:05d}.jpg'

    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset, data_dir):
    dict_single = {'actnet': return_actnet, 'fcvid': return_fcvid, 'minik': return_minik}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](data_dir)
    else:
        raise ValueError('Unknown dataset ' + dataset)

    if isinstance(file_categories, str):
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix
