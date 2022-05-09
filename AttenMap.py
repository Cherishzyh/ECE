from copy import deepcopy
import numpy as np
from scipy.ndimage.morphology import binary_erosion, binary_dilation


class AttentionMap():
    def __init__(self):
        super(AttentionMap, self).__init__()

    def BlurryEdge(self, roi, step=10):
        kernel = np.ones((3, 3))
        result = np.zeros_like(roi).astype(float)
        temp_roi = deepcopy(roi)
        for index in range(step):
            result += 1. / step * temp_roi.astype(float)
            temp_roi = binary_dilation(temp_roi, kernel)

        return result

    def ExtractEdge(self, roi, kernel=np.ones((3, 3))):
        return binary_dilation(roi.astype(bool), kernel, iterations=2).astype(int) - \
               binary_erosion(roi.astype(bool), kernel, iterations=2).astype(int)

    def DetectRegion(self, gland, cancer):
        kernel = np.ones((3, 3))
        gland_edge = self.ExtractEdge(gland, kernel)
        cancer_edge = self.ExtractEdge(cancer, kernel)

        cancer_out = cancer - gland
        cancer_out[cancer_out < 0] = 0
        # cancer_out[cancer_out == 255] = 0

        region = cancer_out + (cancer_edge * gland_edge)
        region[region > 1] = 1
        return region

    def DetectCloseRegion(self, gland, cancer, step=10, kernel=np.ones((3, 3))):
        gland_edge, cancer_edge = self.ExtractEdge(gland), self.ExtractEdge(cancer)
        assert((gland_edge * cancer_edge).sum() == 0)

        diff = np.zeros_like(gland)
        ratio = 0.
        for index in range(step):
            diff = gland_edge * cancer_edge
            if diff.sum() > 0:
                ratio = (step - index) / step
                break
            gland_edge = binary_dilation(gland_edge.astype(bool), kernel)
            cancer_edge = binary_dilation(cancer_edge.astype(bool), kernel)

        return diff, ratio

    def FindRegion(self, gland, cancer):
        # 寻找结合点
        step = 10
        region = self.DetectRegion(gland, cancer)
        if region.sum() >= 1:
            blurry = self.BlurryEdge(region, step)
        else:
            diff, ratio = self.DetectCloseRegion(gland, cancer)
            blurry = ratio * self.BlurryEdge(diff.astype(int), step)

        return blurry.astype(np.float32)


def Test():
    roi = np.zeros((256, 256))
    roi1 = np.zeros((256, 256))
    roi2 = np.zeros((256, 256))

    roi[50:150, 50:150] = 1
    roi1[60:100, 60:100] = 1
    roi2[40:100, 40:100] = 1

    am = AttentionMap()
    # roi = gland, roi1&roi2 = pca
    # blurry = am.FindRegion(roi, roi1)
    blurry = am.FindRegion(roi, roi2)

if __name__ == '__main__':
    Test()