//
// Copyright(c) Multimedia Signal Processing Group (MMSPG),
//              Ecole Polytechnique Fédérale de Lausanne (EPFL)
//              http://mmspg.epfl.ch
//              Zhou Wang
//              https://ece.uwaterloo.ca/~z70wang/
// All rights reserved.
// Author: Philippe Hanhart (philippe.hanhart@epfl.ch)
//
// Permission is hereby granted, without written agreement and without
// license or royalty fees, to use, copy, modify, and distribute the
// software provided and its documentation for research purpose only,
// provided that this copyright notice and the original authors' names
// appear on all copies and supporting documentation.
// The software provided may not be commercially distributed.
// In no event shall the Ecole Polytechnique Fédérale de Lausanne (EPFL)
// be liable to any party for direct, indirect, special, incidental, or
// consequential damages arising out of the use of the software and its
// documentation.
// The Ecole Polytechnique Fédérale de Lausanne (EPFL) specifically
// disclaims any warranties.
// The software provided hereunder is on an "as is" basis and the Ecole
// Polytechnique Fédérale de Lausanne (EPFL) has no obligation to provide
// maintenance, support, updates, enhancements, or modifications.
//

//
// This is an OpenCV implementation of the original Matlab implementation
// from Nikolay Ponomarenko available from http://live.ece.utexas.edu/research/quality/.
// Please refer to the following papers:
// - Z. Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli, "Image quality
//   assessment: from error visibility to structural similarity," IEEE
//   Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, April 2004.
//

#include "SSIM.hpp"

using namespace std;
using namespace cv;

const double SSIM::C1 = 6.5025;
const double SSIM::C2 = 58.5225;

SSIM::SSIM(int h, int w) : Metric(h, w) {
}

float SSIM::compute(const cv::Mat &original, const cv::Mat &processed) {
    cv::Scalar res = computeSSIM(original, processed);
    return float(res.val[0]);
}

float SSIM::compute(const cv::cuda::GpuMat &original, const cv::cuda::GpuMat &processed) {

    cv::Scalar res = computeSSIM(original, processed);
    return float(res.val[0]);
}


cv::Scalar SSIM::computeSSIM(const cv::Mat &img1, const cv::Mat &img2) {

    int ht = img1.rows;
    int wt = img1.cols;
    int w = wt - 10;
    int h = ht - 10;

    cv::Mat mu1(h, w, CV_32F), mu2(h, w, CV_32F);
    cv::Mat mu1_sq(h, w, CV_32F), mu2_sq(h, w, CV_32F), mu1_mu2(h, w, CV_32F);
    cv::Mat img1_sq(ht, wt, CV_32F), img2_sq(ht, wt, CV_32F), img1_img2(ht, wt, CV_32F);
    cv::Mat sigma1_sq(h, w, CV_32F), sigma2_sq(h, w, CV_32F), sigma12(h, w, CV_32F);
    cv::Mat tmp1(h, w, CV_32F), tmp2(h, w, CV_32F), tmp3(h, w, CV_32F);
    cv::Mat ssim_map(h, w, CV_32F), cs_map(h, w, CV_32F);

    // mu1 = filter2(window, img1, 'valid');
    applyGaussianBlur(img1, mu1, 11, 1.5);

    // mu2 = filter2(window, img2, 'valid');
    applyGaussianBlur(img2, mu2, 11, 1.5);

    // mu1_sq = mu1.*mu1;
    cv::multiply(mu1, mu1, mu1_sq);
    // mu2_sq = mu2.*mu2;
    cv::multiply(mu2, mu2, mu2_sq);
    // mu1_mu2 = mu1.*mu2;
    cv::multiply(mu1, mu2, mu1_mu2);

    cv::multiply(img1, img1, img1_sq);
    cv::multiply(img2, img2, img2_sq);
    cv::multiply(img1, img2, img1_img2);

    // sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
    applyGaussianBlur(img1_sq, sigma1_sq, 11, 1.5);
    sigma1_sq -= mu1_sq;

    // sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
    applyGaussianBlur(img2_sq, sigma2_sq, 11, 1.5);
    sigma2_sq -= mu2_sq;

    // sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;
    applyGaussianBlur(img1_img2, sigma12, 11, 1.5);
    sigma12 -= mu1_mu2;

    // cs_map = (2*sigma12 + C2)./(sigma1_sq + sigma2_sq + C2);
    tmp1 = 2 * sigma12 + C2;
    tmp2 = sigma1_sq + sigma2_sq + C2;
    cv::divide(tmp1, tmp2, cs_map);
    // ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
    tmp3 = 2 * mu1_mu2 + C1;
    cv::multiply(tmp1, tmp3, tmp1);
    tmp3 = mu1_sq + mu2_sq + C1;
    cv::multiply(tmp2, tmp3, tmp2);
    cv::divide(tmp1, tmp2, ssim_map);

    // mssim = mean2(ssim_map);
    double mssim = cv::mean(ssim_map).val[0];
    // mcs = mean2(cs_map);
    double mcs = cv::mean(cs_map).val[0];

    cv::Scalar res(mssim, mcs);

    return res;
}

cv::Scalar SSIM::computeSSIM(const cv::cuda::GpuMat &t1, const cv::cuda::GpuMat &t2) {

    cuda::Stream stream;

    cuda::split(t1, gvI1, stream);
    cuda::split(t2, gvI2, stream);
    Scalar mssim;

    Ptr<cuda::Filter> gauss = cuda::createGaussianFilter(gvI1[0].type(), -1, Size(11, 11), 1.5);

    for (size_t i = 0; i < size_t(t1.channels()); ++i) {
        cuda::multiply(gvI1[i], gvI1[i], gI1_2, 1, -1, stream);        // I1^2
        cuda::multiply(gvI2[i], gvI2[i], gI2_2, 1, -1, stream);        // I2^2
        cuda::multiply(gvI1[i], gvI2[i], gI1_I2, 1, -1, stream);       // I1 * I2

        gauss->apply(gvI1[i], gmu1, stream);
        gauss->apply(gvI2[i], gmu2, stream);

        cuda::multiply(gmu1, gmu1, gmu1_2, 1, -1, stream);
        cuda::multiply(gmu2, gmu2, gmu2_2, 1, -1, stream);
        cuda::multiply(gmu1, gmu2, gmu1_mu2, 1, -1, stream);

        gauss->apply(gI1_2, gsigma1_2, stream);
        cuda::subtract(gsigma1_2, gmu1_2, gsigma1_2, cuda::GpuMat(), -1, stream);
        //b.sigma1_2 -= b.mu1_2;  - This would result in an extra data transfer operation

        gauss->apply(gI2_2, gsigma2_2, stream);
        cuda::subtract(gsigma2_2, gmu2_2, gsigma2_2, cuda::GpuMat(), -1, stream);
        //b.sigma2_2 -= b.mu2_2;

        gauss->apply(gI1_I2, gsigma12, stream);
        cuda::subtract(gsigma12, gmu1_mu2, gsigma12, cuda::GpuMat(), -1, stream);
        //b.sigma12 -= b.mu1_mu2;

        //here too it would be an extra data transfer due to call of operator*(Scalar, Mat)
        cuda::multiply(gmu1_mu2, 2, t1, 1, -1, stream); //b.t1 = 2 * b.mu1_mu2 + C1;
        cuda::add(t1, C1, t1, cuda::GpuMat(), -1, stream);
        cuda::multiply(gsigma12, 2, t2, 1, -1, stream); //b.t2 = 2 * b.sigma12 + C2;
        cuda::add(t2, C2, t2, cuda::GpuMat(), -12, stream);

        cuda::multiply(t1, t2, gt3, 1, -1, stream);     // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

        cuda::add(gmu1_2, gmu2_2, t1, cuda::GpuMat(), -1, stream);
        cuda::add(t1, C1, t1, cuda::GpuMat(), -1, stream);

        cuda::add(gsigma1_2, gsigma2_2, t2, cuda::GpuMat(), -1, stream);
        cuda::add(t2, C2, t2, cuda::GpuMat(), -1, stream);


        cuda::multiply(t1, t2, t1, 1, -1, stream);     // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
        cuda::divide(gt3, t1, g_ssim_map, 1, -1, stream);      // ssim_map =  t3./t1;

        stream.waitForCompletion();

        Scalar s = cuda::sum(g_ssim_map, gbuf);
        mssim.val[i] = s.val[0] / (g_ssim_map.rows * g_ssim_map.cols);
    }

//    cv::Scalar res(mssim, 0);
    return mssim;
}

