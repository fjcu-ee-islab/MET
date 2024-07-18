import argparse
import cv2
import numpy as np
import torch

'''
from grad_cam_visualize.pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
'''
from grad_cam_visualize.pytorch_grad_cam import GradCAM
from grad_cam_visualize.pytorch_grad_cam import GuidedBackpropReLUModel
from grad_cam_visualize.pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from grad_cam_visualize.pytorch_grad_cam.ablation_layer import AblationLayerVit


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=6, width=6):
    #result = tensor[:, 1:, :].reshape(tensor.size(0),
    #                                  height, width, tensor.size(2))
    result = tensor[:, :, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def mevt_gradcam(model, all_params, input_res, patch_image_counter, ts_counter):
    """ (XXXX) python vit_gradcam.py --image-path <path_to_image>
    (OOOO) python vit_example.py --use-cuda
    Example usage of using cam-methods on a VIT network.

    """
    
    '''
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}
    '''
    methods = {"gradcam": GradCAM}

    #if args.method not in list(methods.keys()):
        #raise Exception(f"method should be one of {list(methods.keys())}")

    #model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
    #model.eval()

    #if args.use_cuda: model = model.cuda()
    
    #target_layers = [model.backbone.proc_memory_blocks[-1].cross_attention.layer_norm_att]
    #target_layers = [model.backbone.memory_self_att_blocks[-1].latent_attentions[-1].layer_norm_att]

    target_layers = [model.backbone.proc_embs_block.layer_norm]

    
    cam = methods['gradcam'](model=model,
                               target_layers=target_layers,
                               reshape_transform=reshape_transform)

    #rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    #rgb_img = cv2.resize(rgb_img, (224, 224))
    #rgb_img = np.float32(rgb_img) / 255
    #input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
    #                                std=[0.5, 0.5, 0.5])
    
    input_tensor = input_res
    
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    targets = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets)
    
    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]
    
    height, width = input_tensor['pixels'].shape[-2], 1
    
    blank_image = np.zeros((height, width,3), np.float32)
    # blank_image[:,0:width//2] = (255,0,0)      # (B, G, R)
    # blank_image[:,width//2:width] = (0,255,0)
    rgb_img = np.float32(blank_image) / 255
    
    #cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cam_image = show_cam_on_image(blank_image, grayscale_cam)
    
    cv2.imwrite(f'./visual/Patch_visualization/patch_heatmap/patch_'+ str(patch_image_counter)+ '_test_cam_' + str(ts_counter) +'.jpg', cam_image)
