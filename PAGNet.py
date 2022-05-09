import os
from copy import deepcopy

import numpy as np
from scipy import ndimage
import SimpleITK as sitk
from matplotlib.image import imread

import torch
import matplotlib.pyplot as plt

from ConfigInterpretor import ConfigInterpretor, BaseImageOutModel
from AttenMap import AttentionMap
from GradCAM import GradCAM
from ReformatAxis import ReformatAxis


class ECEClassification(BaseImageOutModel):
    def __init__(self):
        super(ECEClassification, self).__init__()
        self._image_preparer = ConfigInterpretor()

    def __GetMaxRoiSlice(self, mask):
        roi_area = np.sum(mask, axis=(0, 1))
        return int(np.argmax(roi_area))

    def __KeepLargest(self, mask):
        new_mask = np.zeros(mask.shape)
        label_im, nb_labels = ndimage.label(mask)
        max_volume = [(label_im == index).sum() for index in range(1, nb_labels + 1)]
        index = np.argmax(max_volume)
        new_mask[label_im == index + 1] = 1
        return new_mask

    def __GetCenter(self, mask):
        assert (np.ndim(mask) == 2)
        roi_row = np.sum(mask, axis=1)
        roi_column = np.sum(mask, axis=0)

        row = np.nonzero(roi_row)[0]
        column = np.nonzero(roi_column)[0]

        center = [row[0] + int((row[-1]-row[0])/2), column[0] + int((column[-1]-column[0])/2)]
        return center

    # def __GetCenter(self, roi):
    #     roi_row = []
    #     roi_column = []
    #     for row in range(roi.shape[0]):
    #         roi_row.append(np.sum(roi[row, ...]))
    #     for column in range(roi.shape[1]):
    #         roi_column.append(np.sum(roi[..., column]))
    #
    #     max_row = max(roi_row)
    #     max_column = max(roi_column)
    #     row_index = roi_row.index(max_row)
    #     column_index = roi_column.index(max_column)
    #
    #     column = np.argmax(roi[row_index])
    #     row = np.argmax(roi[..., column_index])
    #     center = [int(row + max_row // 2), int(column + max_column // 2)]
    #     # center = [int(column + max_column // 2), int(row + max_row // 2)]
    #     return center

    def __NormalizeZ(self, data):
        data -= np.mean(data)
        data /= np.std(data)
        return data

    def __FusionImage(self, gray_array, fusion_array, color_map='jet', alpha=0.3, is_show=False):
        '''
        To Fusion two 2D images.
        :param gray_array: The background
        :param fusion_array: The fore-ground
        :param is_show: Boolen. If set to True, to show the result; else to return the fusion image. (RGB).
        :return:
        '''
        if gray_array.ndim >= 3:
            print("Should input 2d image")
            return gray_array

        dpi = 100
        x, y = gray_array.shape
        w = y / dpi
        h = x / dpi

        fig = plt.figure()
        fig.set_size_inches(w, h)

        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        plt.imshow(gray_array, cmap='gray')
        plt.imshow(fusion_array, cmap=color_map, alpha=alpha)

        fig.subplots_adjust(bottom=0)
        fig.subplots_adjust(top=1)
        fig.subplots_adjust(right=1)
        fig.subplots_adjust(left=0)

        plt.axis('off')
        plt.savefig('temp.jpg', format='jpeg', aspect='normal', bbox_inches='tight', pad_inches=0.0)
        merge_array = imread('temp.jpg')
        os.remove('temp.jpg')

        if is_show:
            plt.show()
        plt.close(fig)

        return merge_array

    def ROI2Array(self, ROI, ref, is_cancer=False):
        if isinstance(ROI, str):
            ROI = sitk.ReadImage(ROI)
        raw_data = ref.Run(ROI)
        raw_data = self.__KeepLargest(raw_data.astype(int))
        if is_cancer:
            slice = self.__GetMaxRoiSlice(raw_data)
            center = self.__GetCenter(raw_data[..., slice])
            return raw_data, slice, center
        else:
            return raw_data

    def Image2Array(self, image, ref):
        if isinstance(image, str):
            image = sitk.ReadImage(image)
        raw_data = ref.Run(image)
        return raw_data.astype(np.float32)

    def ToTorch(self, image_list, roi_list, is_show=False, slice=None):
        from MeDIT.ArrayProcess import ExtractPatch
        ''' image_list = [t2, adc, dwi],  roi_list = [gland, cancer] '''
        ref = ReformatAxis()
        gland = self.ROI2Array(roi_list[0], ref)
        cancer, max_slice, center = self.ROI2Array(roi_list[1], ref, is_cancer=True)

        if slice == None:
            slice = max_slice

        gland_slice = np.around(self._config.CropDataShape(gland[..., slice], roi_list[0].GetSpacing(), center_point=center))
        cancer_slice = np.around(self._config.CropDataShape(cancer[..., slice], roi_list[1].GetSpacing(), center_point=center))

        input_list = []
        for image in image_list:
            data = self.Image2Array(image, ref)
            data, _ = ExtractPatch(data[..., slice], (280, 280), center_point=center)
            data = self.__NormalizeZ(data)
            input_list.append(data)

        attmap = AttentionMap()
        atten_map = attmap.FindRegion(gland_slice, cancer_slice)

        if is_show:
            plt.subplot(121)
            plt.axis('off')
            plt.imshow(input_list[0], cmap='gray')
            plt.contour(cancer_slice, colors='r')
            plt.contour(gland_slice, colors='y')
            plt.subplot(122)
            plt.axis('off')
            plt.imshow(atten_map, cmap='jet')
            plt.show()

        input_list = [torch.from_numpy(input_list[0][np.newaxis, np.newaxis, ...]),
                      torch.from_numpy(input_list[1][np.newaxis, np.newaxis, ...]),
                      torch.from_numpy(input_list[2][np.newaxis, np.newaxis, ...])]
        atten_map = torch.from_numpy(atten_map[np.newaxis, np.newaxis, ...])
        return input_list, atten_map, gland_slice, cancer_slice, center, slice

    def Run(self, image_list, roi_list, is_show_gcam=False, slice=None):
        ''' image_list = [t2, adc, dwi],  roi_list = [gland, cancer] '''

        input_list, atten_map, _, _, _, _ = self.ToTorch(image_list, roi_list, is_show=False, slice=slice)

        cv_pred_list = []
        grad_cam_list = []

        input_0, input_1, input_2 = input_list[0], input_list[1], input_list[2]
        if torch.cuda.is_available():
            input_0 = input_0.to(self.device)
            input_1 = input_1.to(self.device)
            input_2 = input_2.to(self.device)
            atten_map = atten_map.to(self.device)

        for cv_index, cv_weight in enumerate(self._weights_path):
            temp_model = deepcopy(self._model)
            # print(cv_weight.name)
            temp_model.load_state_dict(torch.load(str(cv_weight)))
            temp_model.eval()

            # if cv_index == 1:
            #     gcam = GradCAM(temp_model)
            #     model_pred = gcam.forward([input_0, input_1, input_2, atten_map])
            #     preds = model_pred[:, 0]
            #     target_class = [torch.tensor(0), torch.tensor(1)]
            #     for target in target_class:
            #         gcam.backward(ids=torch.tensor([[target]]).long().to(self.device))
            #         grad_cam = gcam.generate(target_layer='layer4', target_shape=self._config.GetShape())
            #         grad_cam_list.append(grad_cam[0, 0, ...].cpu().numpy())
            # else:
            #     model_pred = temp_model(input_0, input_1, input_2, atten_map)
            #     preds = model_pred[:, 0]
            with torch.no_grad():
                model_pred = temp_model(input_0, input_1, input_2, atten_map)
                preds = model_pred[:, 0]

            cv_pred_list.append((preds).cpu().data.numpy().squeeze())
            del temp_model

        mean_pred = np.mean(np.array(cv_pred_list))

        if is_show_gcam:
            merged_image_0 = self.__FusionImage(Normalize01(np.squeeze(input_list[0])),
                                                Normalize01(np.squeeze(grad_cam_list[0])), is_show=False)
            merged_image_1 = self.__FusionImage(Normalize01(np.squeeze(input_list[0])),
                                                Normalize01(np.squeeze(grad_cam_list[1])), is_show=False)
            plt.suptitle('Prediction: {:.3f}'.format(mean_pred))
            plt.subplot(121)
            plt.title('Grad CAM for ECE')
            plt.axis('off')
            plt.imshow(merged_image_0, cmap='jet')
            plt.subplot(122)
            plt.title('Grad CAM for non-ECE')
            plt.axis('off')
            plt.imshow(merged_image_1, cmap='jet')
            plt.show()
            plt.close()

        return mean_pred, grad_cam_list


if __name__ == '__main__':
    from MeDIT.UsualUse import *
    from pathlib import Path
    import pandas as pd
    from MeDIT.ArrayProcess import Crop2DArray

    segmentor = ECEClassification()
    attenmap = AttentionMap()
    segmentor.LoadConfigAndModel(r'/home/zhangyihong/Documents/ProstateECE/ModelConfig')
    root_folder = Path(r'/home/zhangyihong/Documents/ProstateECE/OriginalData/ProstateCancerECE_SUH')
    root = Path(r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500/AdcSlice')
    if not os.path.exists(r'/home/zhangyihong/Documents/ProstateECE/SUH_AllSlice/T2Slice'):
        os.mkdir(r'/home/zhangyihong/Documents/ProstateECE/SUH_AllSlice/T2Slice')
    if not os.path.exists(r'/home/zhangyihong/Documents/ProstateECE/SUH_AllSlice/AdcSlice'):
        os.mkdir(r'/home/zhangyihong/Documents/ProstateECE/SUH_AllSlice/AdcSlice')
    if not os.path.exists(r'/home/zhangyihong/Documents/ProstateECE/SUH_AllSlice/DwiSlice'):
        os.mkdir(r'/home/zhangyihong/Documents/ProstateECE/SUH_AllSlice/DwiSlice')
    if not os.path.exists(r'/home/zhangyihong/Documents/ProstateECE/SUH_AllSlice/DistanceMap'):
        os.mkdir(r'/home/zhangyihong/Documents/ProstateECE/SUH_AllSlice/DistanceMap')

    total_case_dict = {'CaseName': [], 'All Pred': []}
    for case in sorted(root_folder.iterdir()):
        case_dict = {'slice': [], "pred": []}
        if not os.path.isdir(case):
            print('{} is not a folder'.format(case.name))
            continue

        t2, _, _ = LoadImage(str(case / 't2_5x5.nii'))
        dwi, _, _ = LoadImage(str(case / 'dwi_Reg_5x5.nii'))
        adc, _, _ = LoadImage(str(case / 'adc_Reg_5x5.nii'))
        gland, _, _ = LoadImage(str(case / 'prostate_roi_5x5.nii.gz'))
        cancer, _, _ = LoadImage(str(case / 'pca_roi_5x5.nii.gz'))
        image_list = [t2, adc, dwi]
        roi_image_list = [gland, cancer]
        for slice in range(t2.GetSize()[-1]):
            input_list, atten_map, gland_slice, cancer_slice, center, _ = segmentor.ToTorch(image_list, roi_image_list, slice=slice)
            np.save(os.path.join(r'/home/zhangyihong/Documents/ProstateECE/SUH_AllSlice/T2Slice', '{}_-_slice{}.npy'.format(case.name, slice)), input_list[0])
            np.save(os.path.join(r'/home/zhangyihong/Documents/ProstateECE/SUH_AllSlice/AdcSlice', '{}_-_slice{}.npy'.format(case.name, slice)), input_list[1])
            np.save(os.path.join(r'/home/zhangyihong/Documents/ProstateECE/SUH_AllSlice/DwiSlice', '{}_-_slice{}.npy'.format(case.name, slice)), input_list[2])
            np.save(os.path.join(r'/home/zhangyihong/Documents/ProstateECE/SUH_AllSlice/DistanceMap', '{}_-_slice{}.npy'.format(case.name, slice)), atten_map)


            # if not os.path.exists(os.path.join(root, '{}_-_slice{}.npy'.format(case.name, slice))):
            #     continue
            # else:
            #     adc_data = np.load(os.path.join(root, '{}_-_slice{}.npy'.format(case.name, slice)))

        # plt.subplot(131)
        # plt.imshow(t2_data, cmap='gray')
        # plt.contour(cancer_slice, colors='r')
        # plt.contour(gland_slice, colors='g')
        # plt.subplot(132)
        # plt.imshow(input_list[0].numpy().squeeze(), cmap='gray')
        # plt.contour(cancer_slice, colors='r')
        # plt.contour(gland_slice, colors='g')
        # plt.subplot(133)
        # plt.imshow((input_list[0].numpy().squeeze() - t2_data), cmap='gray')
        # plt.show()
        # print((input_list[1].numpy().squeeze() == adc_data))

    #         try:
    #             mean_pred, _ = segmentor.Run(image_list, roi_image_list, is_show_gcam=False, slice=slice)
    #             case_dict['slice'].append(slice)
    #             case_dict['pred'].append(mean_pred)
    #             print('{} \t {} \t {:.3f}'.format(case.name, slice, mean_pred))
    #         except Exception as e:
    #             print(case.name, e)
    # #
    #
    #     total_case_dict['CaseName'].append(case.name)
    #     total_case_dict['All Pred'].append(case_dict)
    # df = pd.DataFrame(total_case_dict)
    # df.to_csv(r'/home/zhangyihong/Documents/ProstateECE/OriginalData/ProstateCancerECE_SUH/all_slice_preds.csv', index=False)



