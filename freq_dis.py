import datetime
import numpy as np
import time
import torch
import utils
import model   
import torch.backends.cudnn as cudnn

from attack import *
from engine import *
from pathlib import Path
from base_args import get_args
from datasets import build_dataset
from optim_factory import create_optimizer
from utils import get_model, sel_criterion, load_checkpoint
from utils import NativeScalerWithGradNormCount as NativeScaler


feat_result_input = []
feat_result_output = []
feat_result_input_std = []
feat_result_output_std = []
grad_result = []


def get_features_hook(module, data_input, data_output):
    feat_result_input.append(data_input)
    feat_result_output.append(data_output)

def get_features_hook_std(module, data_input, data_output):
    feat_result_input_std.append(data_input)
    feat_result_output_std.append(data_output)

def tensor2cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    return tensor

############################################################
def seed_initial(seed=0):
    seed += utils.get_rank()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args): 
    ### Configuration
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    seed_initial(seed=args.seed)
    ####################################### Get the model
    model_clip = get_model(args)
    model_adv = get_model(args)
    if args.resume:
        checkpoint_model = load_checkpoint(model_clip, args)
        utils.load_state_dict(model_clip, checkpoint_model, prefix=args.model_prefix)
        utils.load_state_dict(model_adv, checkpoint_model, prefix=args.model_prefix)

    model_clip.to(device)
    model_adv.to(device)
    patch_size = model_clip.img_encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size
    print("------------------------------------------------------")

    
    ############################################## Get the test dataloader
    
    valset = build_dataset(is_train=False, args=args)
    sampler_val = torch.utils.data.SequentialSampler(valset)
    
    if valset is not None:
        dataloader_val = torch.utils.data.DataLoader(
            valset, sampler=sampler_val, batch_size=int(1.0 * args.batch_size),
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
    else:
        dataloader_val = None
    
    attack = FGSM_REG(model_adv, 8./255., 2./255., min_val=0, max_val=1, max_iters=4)


    with torch.no_grad():
        if torch.cuda.is_available():
            model_adv.cuda()
            model_clip.cuda()

    #########################################################################################
        # register handler
        relu_index = 0
        model_adv.img_encoder.blocks_cas[-1].relu.register_forward_hook(get_features_hook)
        model_clip.img_encoder.blocks_cas[-1].relu.register_forward_hook(get_features_hook_std)

    ###########################################################################################
        statis_results_robust = 0.
        statis_results_std = 0.
        magnitude_robust = 0.
        magnitude_std = 0.
        batch_idx = 0
        count_samples = 0
        for data_list, label in tqdm(dataloader_val):
            data, bool_masked_pos = data_list
            label = label.to(device, non_blocking=True)
            data = data.to(device, non_blocking=True)
            bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)   
            bool_unmasked_pos = torch.zeros_like(bool_masked_pos)
            bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)          
            bool_unmasked_pos = bool_unmasked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)      
            
            adv_data1 = attack.perturb_fim(data, label, bool_unmasked_pos, 'mean', False, 1.)
            
            # clear feature blobs
            feat_result_input.clear()
            feat_result_output.clear()
            feat_result_input_std.clear()
            feat_result_output_std.clear()


            output1 = model_adv(adv_data1, bool_masked_pos, label, _eval=True)
            output3 = model_clip(data, bool_masked_pos, label, _eval=True)
            output_x_1 = output1['out_x']
            output_x_3 = output3['out_x']
            
            pred1 = torch.max(output_x_1, dim=1)[1]
            pred3 = torch.max(output_x_3, dim=1)[1]

            idx = np.where(label.cpu().numpy() == np.array([0]*data.shape[0]))[0]
            idx = torch.tensor(idx)
            count_samples += len(idx)

            test_std = 0.
            test_robust = 0.
            if len(idx) > 0:
                feat1 = feat_result_input_std[0]
                feat2 = feat_result_output_std[0]
                feat_in = feat1[0][idx]
                feat_out = feat2[idx]
                if len(feat_out.shape) == 3:
                    N, L, C = feat_out.shape
                    # feat_out = feat_out.view(N, C, L)
                    feat_out = torch.mean(feat_out, dim=-1)
                N, C = feat_out.shape
                max_value = torch.max(feat_out, dim=1, keepdim=True)[0]
                threshold = 0.8 * max_value
                mask = feat_out > threshold.expand(N, C)
                count_activate = torch.sum(mask, dim=0).view(C)
                feat_mean_magnitude = torch.sum(feat_out, dim=0).view(C)
                for k in range(C):
                    if feat_mean_magnitude[k] != 0:
                        feat_mean_magnitude[k] = feat_mean_magnitude[k] / count_activate[k].float()
                count_activate = count_activate.cpu().numpy()
                feat_mean_magnitude = feat_mean_magnitude.cpu().numpy()
                if batch_idx == 0:
                    statis_results_std = count_activate
                    magnitude_std = feat_mean_magnitude
                else:
                    statis_results_std = statis_results_std + count_activate
                    magnitude_std = (magnitude_std + feat_mean_magnitude) / 2

            # print(statis_results_std)
            if len(idx) > 0:
                feat1 = feat_result_input[0]
                feat2 = feat_result_output[0]
                feat_in = feat1[0][idx]
                feat_out = feat2[idx]
                if len(feat_out.shape) == 3:
                    N, L, C = feat_out.shape
                    # feat_out = feat_out.view(N, C, L)
                    feat_out = torch.mean(feat_out, dim=-1)
                N, C = feat_out.shape
                max_value = torch.max(feat_out, dim=1, keepdim=True)[0]
                threshold = 0.8 * max_value
                mask = feat_out > threshold.expand(N, C)
                count_activate = torch.sum(mask, dim=0).view(C)
                feat_mean_magnitude = torch.sum(feat_out, dim=0).view(C)
                for k in range(C):
                    if feat_mean_magnitude[k] != 0:
                        feat_mean_magnitude[k] = feat_mean_magnitude[k] / count_activate[k].float()
                count_activate = count_activate.cpu().numpy()
                feat_mean_magnitude = feat_mean_magnitude.cpu().numpy()
                if batch_idx == 0:
                    statis_results_robust = count_activate
                    magnitude_robust = feat_mean_magnitude
                else:
                    statis_results_robust = (statis_results_robust + count_activate)
                    magnitude_robust = (magnitude_robust + feat_mean_magnitude) / 2
            batch_idx += 1

    #################################################################################

        print('Count Samples', count_samples)
        statis_results_robust = np.array(statis_results_robust)
        statis_results_std = np.array(statis_results_std)
        res = np.concatenate([statis_results_robust, statis_results_std], axis=0)
        if os.path.exists('./Frequency') == False:
            os.makedirs('./Frequency')
        np.save('./Frequency/cifar10_std_class0.npy', res)

        magnitude_results_robust = np.array(magnitude_robust)
        magnitude_results_std = np.array(magnitude_std)
        # res = np.concatenate([magnitude_results_robust, magnitude_results_std], axis=0)
        # if os.path.exists('./Magnitude') == False:
        #     os.makedirs('./Magnitude')
        # np.save('./Magnitude/cifar10_std_class0.npy', res)


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
