import os


def return_somethingv1(modality, root_dataset):
    filename_categories = 'something-something-v1/category.txt'
    if modality == 'RGB':
        root_data = os.path.join(root_dataset, 'something-something-v1/20bn-something-something-v1')
        filename_imglist_train = 'something-something-v1/train_videofolder.txt'
        filename_imglist_val = 'something-something-v1/val_videofolder.txt'
        prefix = '{:05d}.jpg'
    elif modality == 'Flow':
        root_data = os.path.join(root_dataset, 'something-something-v1/20bn-something-something-v1-flow')
        filename_imglist_train = 'something-something-v1/train_videofolder_flow.txt'
        filename_imglist_val = 'something-something-v1/val_videofolder_flow.txt'
        prefix = '{:06d}-{}_{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv2(modality, root_dataset):
    filename_categories = 'something-something-v2/category.txt'
    if modality == 'RGB':
        root_data = os.path.join(root_dataset, 'something-something-v2/20bn-something-something-v2-frames')
        filename_imglist_train = 'something-something-v2/train_videofolder.txt'
        filename_imglist_val = 'something-something-v2/val_videofolder.txt'
        prefix = '{:06d}.jpg'
    elif modality == 'Flow':
        root_data = os.path.join(root_dataset, 'something-something-v2/20bn-something-something-v2-flow')
        filename_imglist_train = 'something-something-v2/train_videofolder_flow.txt'
        filename_imglist_val = 'something-something-v2/val_videofolder_flow.txt'
        prefix = '{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset, modality, root_dataset):
    dict_single = {'somethingv1': return_somethingv1, 'somethingv2': return_somethingv2}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality, root_dataset)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(root_dataset, file_imglist_train)
    file_imglist_val = os.path.join(root_dataset, file_imglist_val)
    if isinstance(file_categories, str):
        file_categories = os.path.join(root_dataset, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix