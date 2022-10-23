import torch
import cv2
# Mobile
from Saved_Models.mobileGRU.front_IR import front_IR_v3 as mobile_front_IR_v3
from Saved_Models.mobileGRU.front_depth import front_depth_v1 as mobile_front_depth_v1
from Saved_Models.mobileGRU.top_IR import top_IR_v1 as mobile_top_IR_v1
from Saved_Models.mobileGRU.top_depth import top_depth_v1 as mobile_top_depth_v1
# ResNeXt
from Saved_Models.ResNextGRU.front_IR import front_IR_v2 as resnext_front_IR_v2
from Saved_Models.ResNextGRU.front_depth import front_depth_v12 as resnext_front_depth_v12
from Saved_Models.ResNextGRU.top_IR import top_IR_v2 as resnext_top_IR_v2
from Saved_Models.ResNextGRU.top_depth import top_depth_v2 as resnext_top_depth_v2
# ConvGRU
from Saved_Models.ConvGRUv4.front_IR import front_IR_v1 as convgru_front_IR_v1
from Saved_Models.ConvGRUv4.front_depth import front_depth_v1 as convgru_front_depth_v1
from Saved_Models.ConvGRUv4.top_IR import top_IR_v2 as convgru_top_IR_v2
from Saved_Models.ConvGRUv4.top_depth import top_depth_v13 as convgru_top_depth_v13
from Demo.dataset_demo import DAD_demo
import numpy as np
import os
import torchvision
from tqdm import tqdm


def run(make_scores=False, val_folder='val01', rec_folder='rec1', model_head='ConvGRUv4'):
    if make_scores:
        if model_head.lower() == 'convgruv4':
            front_IR_model = convgru_front_IR_v1.ConvGRUv4()
            front_IR_model.load_state_dict(torch.load('C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Saved_Models/ConvGRUv4/front_IR/front_IR_v1.pth'))
            front_IR_model.eval().cuda()
            front_depth_model = convgru_front_depth_v1.ConvGRUv4()
            front_depth_model.load_state_dict(torch.load('C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Saved_Models/ConvGRUv4/front_depth/front_depth_v1.pth'))
            front_depth_model.eval().cuda()
            top_IR_model = convgru_top_IR_v2.ConvGRUv4()
            top_IR_model.load_state_dict(torch.load('C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Saved_Models/ConvGRUv4/top_IR/top_IR_v2.pth'))
            top_IR_model.eval().cuda()
            top_depth_model = convgru_top_depth_v13.ConvGRUv4()
            top_depth_model.load_state_dict(torch.load('C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Saved_Models/ConvGRUv4/top_depth/top_depth_v13.pth'))
            top_depth_model.eval().cuda()
        elif model_head.lower() == 'mobilenetv2':
            front_IR_model = mobile_front_IR_v3.mobileGRU()
            front_IR_model.load_state_dict(torch.load('C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Saved_Models/mobileGRU/front_IR/front_IR_v3.pth'))
            front_IR_model.eval().cuda()
            front_depth_model = mobile_front_depth_v1.mobileGRU()
            front_depth_model.load_state_dict(torch.load('C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Saved_Models/mobileGRU/front_depth/front_depth_v1.pth'))
            front_depth_model.eval().cuda()
            top_IR_model = mobile_top_IR_v1.mobileGRU()
            top_IR_model.load_state_dict(torch.load('C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Saved_Models/mobileGRU/top_IR/top_IR_v1.pth'))
            top_IR_model.eval().cuda()
            top_depth_model = mobile_top_depth_v1.mobileGRU()
            top_depth_model.load_state_dict(torch.load('C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Saved_Models/mobileGRU/top_depth/top_depth_v1.pth'))
            top_depth_model.eval().cuda()
        elif model_head.lower() == 'resnext':
            front_IR_model = resnext_front_IR_v2.ResNextGRU()
            front_IR_model.load_state_dict(torch.load('C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Saved_Models/ResNextGRU/front_IR/front_IR_v2.pth'))
            front_IR_model.eval().cuda()
            front_depth_model = resnext_front_depth_v12.ResNextGRU()
            front_depth_model.load_state_dict(torch.load('C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Saved_Models/ResNextGRU/front_depth/front_depth_v12.pth'))
            front_depth_model.eval().cuda()
            top_IR_model = resnext_top_IR_v2.ResNextGRU()
            top_IR_model.load_state_dict(torch.load('C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Saved_Models/ResNextGRU/top_IR/top_IR_v2.pth'))
            top_IR_model.eval().cuda()
            top_depth_model = resnext_top_depth_v2.ResNextGRU()
            top_depth_model.load_state_dict(torch.load('C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Saved_Models/ResNextGRU/top_depth/top_depth_v2.pth'))
            top_depth_model.eval().cuda()


        # Transforms
        normal_eval_trans = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(size=(165, 194)),
                                                            torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Resize(size=(160, 160)),
                                                            torchvision.transforms.Normalize([0.], [1.])])

        front_IR_test_ds = DAD_demo(sample_duration=16, val_folder=val_folder, rec_folder=rec_folder, view='front_IR', spatial_transform=normal_eval_trans)
        front_depth_test_ds = DAD_demo(sample_duration=16, val_folder=val_folder, rec_folder=rec_folder, view='front_depth', spatial_transform=normal_eval_trans)
        top_IR_test_ds = DAD_demo(sample_duration=16, val_folder=val_folder, rec_folder=rec_folder, view='top_IR', spatial_transform=normal_eval_trans)
        top_depth_test_ds = DAD_demo(sample_duration=16, val_folder=val_folder, rec_folder=rec_folder, view='top_depth', spatial_transform=normal_eval_trans)

        front_IR_test_dl = torch.utils.data.DataLoader(front_IR_test_ds, batch_size=32, num_workers=1, pin_memory=True)
        front_depth_test_dl = torch.utils.data.DataLoader(front_depth_test_ds, batch_size=32, num_workers=1, pin_memory=True)
        top_IR_test_dl = torch.utils.data.DataLoader(top_IR_test_ds, batch_size=32, num_workers=1, pin_memory=True)
        top_depth_test_dl = torch.utils.data.DataLoader(top_depth_test_ds, batch_size=32, num_workers=1, pin_memory=True)

        front_IR_thresh = torch.tensor([-0.9005]).cuda()
        front_depth_thresh = torch.tensor([-0.7038]).cuda()
        top_IR_thresh = torch.tensor([-1.3789]).cuda()
        top_depth_thresh = torch.tensor([-1.0242]).cuda()

        # For average
        thresh = ((front_IR_thresh + front_depth_thresh) / 2 + (top_depth_thresh + top_IR_thresh) / 2) / 2

        # Ground Truth
        total_labels = np.empty(shape=(0,))

        # Total Predictions
        total_pred_front_IR = np.empty(shape=(0,))
        total_pred_front_depth = np.empty(shape=(0,))
        total_pred_top_IR = np.empty(shape=(0,))
        total_pred_top_depth = np.empty(shape=(0,))

        # Total Scores
        scores_front_IR = np.empty(shape=(0,))
        scores_front_depth = np.empty(shape=(0,))
        scores_top_IR = np.empty(shape=(0,))
        scores_top_depth = np.empty(shape=(0,))

        for data1, data2, data3, data4 in tqdm(zip(front_IR_test_dl, front_depth_test_dl, top_IR_test_dl, top_depth_test_dl), total=len(front_IR_test_dl), desc='Calculating Scores '):
            inputs_front_IR, labels = data1
            inputs_front_depth, _ = data2
            inputs_top_IR, _ = data3
            inputs_top_depth, _ = data4
            inputs_front_IR, inputs_front_depth, inputs_top_IR, inputs_top_depth = inputs_front_IR.cuda(), inputs_front_depth.cuda(), inputs_top_IR.cuda(), inputs_top_depth.cuda()
            labels = labels.cuda()
            with torch.no_grad():
                outputs_front_IR = front_IR_model(inputs_front_IR)
                outputs_front_depth = front_depth_model(inputs_front_depth)
                outputs_top_IR = top_IR_model(inputs_top_IR)
                outputs_top_depth = top_depth_model(inputs_top_depth)
                pred_front_IR = (outputs_front_IR > front_IR_thresh).float()
                total_pred_front_IR = np.concatenate((total_pred_front_IR, pred_front_IR.detach().cpu()))
                scores_front_IR = np.concatenate((scores_front_IR, outputs_front_IR.detach().cpu()))
                pred_front_depth = (outputs_front_depth > front_depth_thresh).float()
                scores_front_depth = np.concatenate((scores_front_depth, outputs_front_depth.detach().cpu()))
                total_pred_front_depth = np.concatenate((total_pred_front_depth, pred_front_depth.detach().cpu()))
                pred_top_IR = (outputs_top_IR > top_IR_thresh).float()
                scores_top_IR = np.concatenate((scores_top_IR, outputs_top_IR.detach().cpu()))
                total_pred_top_IR = np.concatenate((total_pred_top_IR, pred_top_IR.detach().cpu()))
                pred_top_depth = (outputs_top_depth > top_depth_thresh).float()
                scores_top_depth = np.concatenate((scores_top_depth, outputs_top_depth.detach().cpu()))
                total_pred_top_depth = np.concatenate((total_pred_top_depth, pred_top_depth.detach().cpu()))
                total_labels = np.concatenate((total_labels, labels.detach().cpu()))

        # Majority Voting
        total_sum = total_pred_front_IR + total_pred_front_depth + total_pred_top_depth + total_pred_top_IR
        total_sum = np.where(total_sum >= 3., 1., total_pred_top_IR)  # Change it to best camera
        save_path = 'C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Demo/' + model_head + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(save_path + 'Majority_' + val_folder + '_' + rec_folder + '.txt', 'w') as f:
            for item in total_sum:
                f.write(str(item) + '\n')
        # Decision on Average
        total_scores = ((scores_front_IR + scores_front_depth) / 2 + (scores_top_depth + scores_top_IR) / 2) / 2
        thresh = ((front_IR_thresh + front_depth_thresh) / 2 + (top_depth_thresh + top_IR_thresh) / 2) / 2
        total_pred = np.where(total_scores > thresh.item(), 1., 0.)
        with open(save_path + 'Average_' + val_folder + '_' + rec_folder + '.txt', 'w') as f:
            for item in total_pred:
                f.write(str(item) + '\n')
        with open(save_path + 'Ground_Truth_' + val_folder + '_' + rec_folder + '.txt', 'w') as f:
            for item in total_labels:
                f.write(str(item) + '\n')

    else:
        open_path = 'C:/Users/panta/Desktop/Driver Distraction Another Approach v4/Demo/' + model_head + '/'
        with open(open_path + 'Majority_' + val_folder + '_' + rec_folder + '.txt') as f:
            majority_pred = np.array(f.readlines())
        with open(open_path + 'Average_' + val_folder + '_' + rec_folder + '.txt') as f:
            average_pred = np.array(f.readlines())
        with open(open_path + 'Ground_Truth_' + val_folder + '_' + rec_folder + '.txt') as f:
            ground_truth = np.array(f.readlines())

        images_path = 'E:/DAD/' + val_folder + '/' + rec_folder + '/'
        os.chdir(images_path)
        frames_front_IR = [images_path + 'front_IR/img_' + str(i) + '.png' for i in range(10000)]
        frames_front_depth = [images_path + 'front_depth/img_' + str(i) + '.png' for i in range(10000)]
        frames_top_IR = [images_path + 'top_IR/img_' + str(i) + '.png' for i in range(10000)]
        frames_top_depth = [images_path + 'top_depth/img_' + str(i) + '.png' for i in range(10000)]

        font = cv2.FONT_HERSHEY_SIMPLEX

        # org
        org_maj = (620, 30)
        org_avg = (620, 60)
        org_gt = (620, 90)
        # fontScale
        fontScale = 0.7
        color = (250, 206, 135)

        # Line thickness of 2 px
        thickness = 2
        count = 0
        count2 = 0
        for img_front_IR, img_front_depth, img_top_IR, img_top_depth in zip(frames_front_IR, frames_front_depth, frames_top_IR, frames_top_depth):
            front_IR = cv2.imread(img_front_IR)
            front_IR = cv2.resize(front_IR, (720, 480))
            front_depth = cv2.imread(img_front_depth)
            front_depth = cv2.resize(front_depth, (720, 480))
            top_IR = cv2.imread(img_top_IR)
            top_IR = cv2.resize(top_IR, (720, 480))
            top_depth = cv2.imread(img_top_depth)
            top_depth = cv2.resize(top_depth, (720, 480))

            if count%16 == 0:
                if majority_pred[count2].replace('\n', '') == '1.0':
                    color_maj = (0, 0, 255)
                    text_maj = 'Anomaly'
                else:
                    color_maj = (0, 255, 0)
                    text_maj = 'Normal'
                if average_pred[count2].replace('\n', '') == '1.0':
                    color_avg = (0, 0, 255)
                    text_avg = 'Anomaly'
                else:
                    color_avg = (0, 255, 0)
                    text_avg = 'Normal'
                if ground_truth[count2].replace('\n', '') == '1.0':
                    color_gt = (0, 0, 255)
                    text_gt = 'Anomaly'
                else:
                    color_gt = (0, 255, 0)
                    text_gt = 'Normal'
                count2 = count2 + 1

            if 'text_maj' in locals() and 'text_avg' in locals() and 'text_gt' in locals():
                top_IR = cv2.putText(top_IR, 'Majority: ', (org_maj[0] - 100, org_maj[1]) , font, fontScale, color, thickness, cv2.LINE_AA)
                top_IR = cv2.putText(top_IR, text_maj, org_maj, font, fontScale, color_maj, thickness, cv2.LINE_AA)
                top_IR = cv2.putText(top_IR, 'Average: ', (org_avg[0] - 100, org_avg[1]), font, fontScale, color, thickness, cv2.LINE_AA)
                top_IR = cv2.putText(top_IR, text_avg, org_avg, font, fontScale, color_avg, thickness, cv2.LINE_AA)
                top_IR = cv2.putText(top_IR, 'Actual: ', (org_gt[0] - 100, org_gt[1]), font, fontScale, color, thickness, cv2.LINE_AA)
                top_IR = cv2.putText(top_IR, text_gt, org_gt, font, fontScale, color_gt, thickness, cv2.LINE_AA)

            ir = np.concatenate((front_IR, top_IR), axis=1)
            depth = np.concatenate((front_depth, top_depth), axis=1)

            combined = np.concatenate((ir, depth), axis=0)
            cv2.imshow('All cameras', combined)
            cv2.waitKey(50)
            count = count + 1


if __name__ == '__main__':
    # MobileNetV2  or  ConvGRUv4   or ResNext   val06, rec2
    run(make_scores=False, val_folder='val06', rec_folder='rec2', model_head='ConvGRUv4')

