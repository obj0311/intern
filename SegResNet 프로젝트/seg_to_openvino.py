import logging
import os
import os.path
import sys
import tempfile
from glob import glob
from datetime import datetime
import time
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing
from torch.nn.functional import interpolate
from scipy.ndimage import zoom
from monai.data.meta_tensor import MetaTensor

import monai
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Spacingd,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
)
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.utils import set_determinism
from sklearn.preprocessing import Normalizer 
from matplotlib import cm 
import openvino as ov
from openvino.runtime import get_version


def main(data_dir, checkpoint_file, target_label,save_folder):


    # typeofprocess = 'NoThresh_labelMap'           # Label map default (no thresholding and output uses argmax) --> 확률이 낮은 곳들도 정답 영역으로 분류함
    #typeofprocess = 'NoThresh_probabiltyMap'         # probability map (no thresholding and output uses probability)
    #typeofprocess = 'Thresh_probabiltyMap' # Thresholding probability and output is probability map
    typeofprocess = 'Thresh_labelMap'      # Thresholding probability and output is label map

    #monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    torch.cuda.empty_cache()
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # Set deterministic training for reproducibility
    set_determinism(seed=0)    
    
    ###################################################################
    # 0. set parameters
    ###################################################################
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    NUM_OF_LABELS = 6
    BATCH_SIZE = 1
    MODEL_NAME = 'SEGRESNET'
    INFERENCE_ROI_SIZE = 128
   

    ###################################################################
    # 1. setup database
    ###################################################################

    # create a temporary directory and 40 random image, mask pairs
    images = sorted(glob(os.path.join(data_dir, "*_image.nii.gz")))

    test_count = (int)(len(images))
    test_files = [{"image": img} for img in zip(images[:test_count])]
    
    ###################################################################
    # 2. set transformations
    ###################################################################

    # define transforms for image and segmentation
    test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys="image"),
            Spacingd(
                keys=["image"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            #ScaleIntensityd(keys=["image"]),
            EnsureTyped(keys=["image"]),
        ]
    )
    
    ###################################################################
    # 3. load database
    ###################################################################

    # create a test data loader
    test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=2*BATCH_SIZE, pin_memory=True)

    ###################################################################
    # 4. set model and convert to OpenVINO model
    ###################################################################
    
    model = monai.networks.nets.SegResNet(
                blocks_down=[1, 2, 2, 4],
                blocks_up=[1, 1, 1],
                init_filters=16,
                in_channels=1,
                out_channels=NUM_OF_LABELS,
                dropout_prob=0.2,
            )


    model.to('cpu')
    model.load_state_dict(torch.load(checkpoint_file))
    model.eval()
    
    
    ##################################################
    #OpenVINO 모델로 변환
    ##################################################
    example_input = torch.randn(1, 1, 128, 128, 128).cpu()  # (batch, channel, D, H, W) 예시
    ov_model = ov.convert_model(model, example_input=[example_input],input=(1,1,128,128,128))
    ov.save_model(ov_model, 'seg.xml')

    core = ov.Core()
    network = core.read_model(model = 'seg.xml', weights = 'seg.bin')
    compiled_model = core.compile_model(model = network, device_name = 'CPU')
    output_layer = next(iter(compiled_model.outputs))

    targetIndex = -1
    
    if targetIndex > 0 :
        post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    else:
        post_trans           = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(argmax=True, to_onehot=NUM_OF_LABELS)])
        post_trans_sigmoid   = Compose([EnsureType(), Activations(sigmoid=True)])
        post_trans_argmax    = AsDiscrete(argmax=True, to_onehot=NUM_OF_LABELS)
        #post_trans_max       = AsDiscrete(argmax=False, to_onehot=NUM_OF_LABELS)

    ###################################################################
    # 5. execute inference and save images  
    ###################################################################

    with torch.no_grad():
        index = 0
        for test_data in test_loader:
            #if volume_size > 128:
            test_inputs = interpolate(test_data["image"], size=[INFERENCE_ROI_SIZE, INFERENCE_ROI_SIZE, INFERENCE_ROI_SIZE]).to('cpu')
            
            ## 추가한 부분 #######################################
            example_input = test_inputs
            ov_result = compiled_model([example_input])[output_layer]                                                                                                                                                                                                                                                                                                                                                  #out_max11 = torch.max(test_outputs)

            
            tensor = torch.from_numpy(ov_result)
            test_outputs = MetaTensor(tensor)
            # ov_outputs_= [post_trans(i) for i in decollate_batch(meta_tensor)]
            # ov_outputs = torch.stack(ov_outputs_)
            #####################################################
            if typeofprocess=='Thresh_probabiltyMap' or typeofprocess=='Thresh_labelMap' :

                thresh_background  = nn.Threshold(1.0,0)
                thresh_fluid       = nn.Threshold(0.8,0) # everything below thresh.value will be 0
                thresh_face        = nn.Threshold(0.3,0)
                thresh_body        = nn.Threshold(0.5,0)
                thresh_ucord       = nn.Threshold(0.1,0)
                thresh_limb        = nn.Threshold(0.5,0)
                thresh_uterus      = nn.Threshold(0.5,0)
                thresh_placenta    = nn.Threshold(0.5,0) 

                test_outputs_probability = torch.zeros(1,NUM_OF_LABELS,INFERENCE_ROI_SIZE,INFERENCE_ROI_SIZE,INFERENCE_ROI_SIZE)
                test_outputs_probability_thresh = torch.zeros(1,NUM_OF_LABELS,INFERENCE_ROI_SIZE,INFERENCE_ROI_SIZE,INFERENCE_ROI_SIZE)
                test_output_background   = torch.zeros(INFERENCE_ROI_SIZE,INFERENCE_ROI_SIZE,INFERENCE_ROI_SIZE)

                #probability: 

                test_outputs_ = [post_trans_sigmoid(i) for i in decollate_batch(test_outputs)] # probability
                test_outputs = torch.stack(test_outputs_)
                #test_outputs = thresh_fluid(test_outputs_)
                #test_output_fluid_numpy = test_output_fluid.cpu().detach().numpy()
                #test_output_fluid__ = post_trans2(test_output_fluid_numpy)

                if NUM_OF_LABELS==8:
                    test_output_fluid      = test_outputs[0,1,:,:,:]
                    test_output_face       = test_outputs[0,2,:,:,:]
                    test_output_body       = test_outputs[0,3,:,:,:]
                    test_output_ucord      = test_outputs[0,4,:,:,:]
                    test_output_limb       = test_outputs[0,5,:,:,:]
                    test_output_uterus     = test_outputs[0,6,:,:,:]
                    test_output_placenta   = test_outputs[0,7,:,:,:]

                elif NUM_OF_LABELS==6: 
                    test_output_fluid      = test_outputs[0,1,:,:,:]
                    test_output_face       = test_outputs[0,2,:,:,:]
                    test_output_ucord      = test_outputs[0,3,:,:,:]
                    test_output_uterus     = test_outputs[0,4,:,:,:]
                    test_output_placenta   = test_outputs[0,5,:,:,:]
                                
                ## Histogram
                #test_output_fluid_ = test_outputs[0,0,:,:,:]
                #hist = torch.histc(test_output_fluid_, bins=2, min=0, max=1)
                #hist_numpy = hist.cpu().detach().numpy()
                #bins = 2
                #x = range(2)
                #plt.bar(x,hist_numpy,align='center',color=['blue'])
                #plt.xlabel('Bins')
                #plt.ylabel('Frequency')
                #plt.show()

                # background
                test_output_background_    = thresh_background(test_output_background)

                if NUM_OF_LABELS==8:
                    # fluid 
                    test_output_fluid_         = thresh_fluid(test_output_fluid)
                    # face                    
                    test_output_face_          = thresh_face(test_output_face)
                    # body                    
                    test_output_body_          = thresh_body(test_output_body)
                    # ucord                   
                    test_output_ucord_         = thresh_ucord(test_output_ucord)
                    # limb                     
                    test_output_limb_          = thresh_limb(test_output_limb)
                    # uterus                  
                    test_output_uterus_        = thresh_uterus(test_output_uterus)
                    # placenta                
                    test_output_placenta_      = thresh_placenta(test_output_placenta)

                elif NUM_OF_LABELS==6: 
                    # fluid 
                    test_output_fluid_         = thresh_fluid(test_output_fluid)
                    # face                    
                    test_output_face_          = thresh_face(test_output_face)
                    # ucord                   
                    test_output_ucord_         = thresh_ucord(test_output_ucord)
                    # uterus                  
                    test_output_uterus_        = thresh_uterus(test_output_uterus)
                    # placenta                
                    test_output_placenta_      = thresh_placenta(test_output_placenta)
                
                # assign back after thresholding: 
                if NUM_OF_LABELS==8:
                    test_outputs_probability[0,0,:,:,:]  =  test_output_background
                    test_outputs_probability[0,1,:,:,:]  =  test_output_fluid_   
                    test_outputs_probability[0,2,:,:,:]  =  test_output_face_    
                    test_outputs_probability[0,3,:,:,:]  =  test_output_body_    
                    test_outputs_probability[0,4,:,:,:]  =  test_output_ucord_   
                    test_outputs_probability[0,5,:,:,:]  =  test_output_limb_ 
                    test_outputs_probability[0,6,:,:,:]  =  test_output_uterus_
                    test_outputs_probability[0,7,:,:,:]  =  test_output_placenta_

                elif NUM_OF_LABELS==6: 
                    test_outputs_probability[0,0,:,:,:]  =  test_output_background
                    test_outputs_probability[0,1,:,:,:]  =  test_output_fluid_   
                    test_outputs_probability[0,2,:,:,:]  =  test_output_face_    
                    test_outputs_probability[0,3,:,:,:]  =  test_output_ucord_   
                    test_outputs_probability[0,4,:,:,:]  =  test_output_uterus_
                    test_outputs_probability[0,5,:,:,:]  =  test_output_placenta_

                test_outputs_numpy  =  test_outputs_probability.cpu().detach().numpy()

                ## If you wanna have 0's or 1's: 
                test_outputs__ = [post_trans_argmax(i) for i in (test_outputs_numpy)]
                test_outputs_indexes = np.stack(test_outputs__)
                test_outputs_indexes = torch.from_numpy(test_outputs_indexes).to('cpu')

                test_outputs_probability_thresh = torch.zeros(1,NUM_OF_LABELS,INFERENCE_ROI_SIZE, INFERENCE_ROI_SIZE, INFERENCE_ROI_SIZE)
                # convert only 1's to probability  (comment this if you want to have 1's and 0's) 
                for label_index in range(2, NUM_OF_LABELS+1):  
                    test_outputs_probabilities_single = torch.where(test_outputs_indexes[0][label_index-1] ==1, test_outputs_[0][label_index-1],0)
                    test_outputs_probability_thresh[0][label_index-1] = test_outputs_probabilities_single
                
                ################################
                c_bin = np.fromfile("C:/Users/MinheeJang/source/repos/segresnet-OpenVINO/portraitVue-Enhancement-OpenVINO/prob_thresh_c1.bin", dtype=np.float32).reshape(128, 128, 128)
                py_prob = test_outputs_probability[0, 1].cpu().numpy()  # 1번 채널(ex: fluid, 첨부 코드 구조와 일치)
                diff = np.abs(c_bin - py_prob)
                print("최대 차이:", diff.max())
                print("평균 차이:", diff.mean())
                import matplotlib.pyplot as plt
                plt.subplot(1,2,1); plt.imshow(py_prob[64,:,:]); plt.title("Py(MONAI) prob")
                plt.subplot(1,2,2); plt.imshow(c_bin[64,:,:]); plt.title("C++ prob")
                plt.show()
                exit()
                ###################################


                                          
            if typeofprocess=='NoThresh_labelMap':
                ##  1's and 0's:
                test_outputs_= [post_trans(i) for i in decollate_batch(test_outputs)]
                test_outputs = torch.stack(test_outputs_)

                

            if typeofprocess=='NoThresh_probabiltyMap':
                test_outputs_ = [post_trans_sigmoid(i) for i in decollate_batch(test_outputs)] # probability
                test_outputs = torch.stack(test_outputs_)
                #initialize: 
                test_outputs_probabilities = torch.zeros(1,NUM_OF_LABELS,INFERENCE_ROI_SIZE, INFERENCE_ROI_SIZE, INFERENCE_ROI_SIZE)
                test_outputs__ = [post_trans_argmax(i) for i in decollate_batch(test_outputs)] # probability
                test_outputs_indexes = torch.stack(test_outputs__)  # get 1's and 0' 
                # convert only 1's to probability   
                for label_index in range(2, NUM_OF_LABELS+1):  
                    test_outputs_probabilities_single                   = torch.where(test_outputs_indexes[0][label_index-1] ==1, test_outputs_[0][label_index-1],0)
                    test_outputs_probabilities[0][label_index-1] = test_outputs_probabilities_single
            
            targetName = target_label
            basename = os.path.basename(images[index]).split('.nii.gz')[0].replace('_image', '_' + targetName)
            
            #if volume_size == 128:
            volume_array = np.zeros((INFERENCE_ROI_SIZE, INFERENCE_ROI_SIZE, INFERENCE_ROI_SIZE), dtype=np.float32)
            #elif volume_size > 128:
            #     volume_array = np.zeros((INFERENCE_ROI_SIZE*2, INFERENCE_ROI_SIZE*2, INFERENCE_ROI_SIZE*2), dtype=np.float32)
            
            if targetIndex > 0 :
                volume_array = np.array(test_outputs[0][targetIndex-1].detach().cpu()) * targetIndex
            else :
                for label_index in range(2, NUM_OF_LABELS+1):                
                    if typeofprocess == 'Thresh_labelMap':
                            arr = np.array(test_outputs_indexes[0][label_index-1].detach().cpu()) * label_index   # numpy not tenor 
                            volume_array = volume_array + arr

                    elif typeofprocess == 'Thresh_probabiltyMap':
                            arr = np.array(test_outputs_probability_thresh[0][label_index-1].detach().cpu()) * label_index   # numpy not tenor 
                            volume_array = volume_array + arr

                    elif typeofprocess == 'NoThresh_probabiltyMap':
                            arr = np.array(test_outputs_probabilities[0][label_index-1].detach().cpu()) * label_index
                            volume_array = volume_array + arr

                    else:
                            arr = np.array(test_outputs[0][label_index-1].detach().cpu()) * label_index
                            volume_array = volume_array + arr



            save_nifti_filename = os.path.join(save_folder, f"{basename}.nii.gz")
            

            nib.save(nib.Nifti1Image(volume_array, np.eye(4)), save_nifti_filename)
            print(save_nifti_filename)
                

            index = index + 1
            f = time.time()
            a = time.time()


if __name__ == "__main__":
    #test_fodler = "D:/project_fetus_segmentation/pythonProject/input/test_newData/1st_input/"
    test_folder     = "./test_image/"
    checkpoint_file = "./1st.pth"
    target_label    = "AllLabels6"
    save_folder     = "./ov_test/"
    main(test_folder, checkpoint_file, target_label, save_folder)