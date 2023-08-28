import torch
import os
import numpy as np
from datasets.crowd import Crowd
from models.vgg_c import vgg19_trans
import argparse
import math
import glob

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='/workspace/jhu_Train_Val_Test',
                        help='training data directory')
    # parser.add_argument('--save-dir', default='/workspace/Code/models/UCF.pth',
    #                     help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    datasets = Crowd(os.path.join(args.data_dir, 'test'), 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=4, pin_memory=False)

    device = torch.device('cuda')

    if torch.cuda.is_available():
        print("GPU is available")
    else:
        print("GPU is not available")
    model = vgg19_trans()
    model.to(device)
    model.eval()
    
    best_mse = np.inf
    best_mae = np.inf
    best_model =''
    desired_pattern = "/workspace/model/0823-072924/best_model_*.pth"
    matching_files = glob.glob(desired_pattern)

    for file_path in matching_files:
        epoch_minus = []
        model.load_state_dict(torch.load(file_path, device))
        for inputs, count, name in dataloader:
            inputs = inputs.to(device)
            b, c, h, w = inputs.shape
            h, w = int(h), int(w)
            assert b == 1, 'the batch size should equal to 1 in validation mode'
            input_list = []
            if h >= 3584 or w >= 3584:
                h_stride = int(math.ceil(1.0 * h / 3584))
                w_stride = int(math.ceil(1.0 * w / 3584))
                h_step = h // h_stride
                w_step = w // w_stride
                for i in range(h_stride):
                    for j in range(w_stride):
                        h_start = i * h_step
                        if i != h_stride - 1:
                            h_end = (i + 1) * h_step
                        else:
                            h_end = h
                        w_start = j * w_step
                        if j != w_stride - 1:
                            w_end = (j + 1) * w_step
                        else:
                            w_end = w
                        input_list.append(inputs[:, :, h_start:h_end, w_start:w_end])
                        # print(000000,input_list)
                with torch.set_grad_enabled(False):
                    pre_count = 0.0
                    for idx, input in enumerate(input_list):
                        output = model(input)[0]
                        pre_count += torch.sum(output)
                        # print(1100,pre_count)
                res = count[0].item() - pre_count.item()
                epoch_minus.append(res)
                # print(21100,res)
            else:
                with torch.set_grad_enabled(False):
                    outputs = model(inputs)[0]
                    res = count[0].item() - torch.sum(outputs).item()
                    epoch_minus.append(res)
                    # print(21100,res)

        epoch_minus = np.array(epoch_minus)
        mse = np.sqrt(np.mean(np.square(epoch_minus)))
        mae = np.mean(np.abs(epoch_minus))
        log_str = 'Model: {}'.format(file_path)
        print(log_str)
        log_str = 'mae {}, mse {}'.format(mae, mse)
        print(log_str)
        if (2.0 * mse + mae) < (2.0 * best_mse + best_mae):
            best_mse = mse
            best_mae = mae
            best_model = file_path
            log_str = 'Best mae: {}, Best mse: {}'.format(mae, mse)
            print(log_str)
            log_str = 'New best model: {}'.format(best_model)
            print(log_str)
    log_str = 'Final best model: {}'.format(best_model)    
    print(log_str)