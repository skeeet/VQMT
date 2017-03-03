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

/**************************************************************************

 Calculation of the Structural Similarity (SSIM) image quality measure.

**************************************************************************/

#ifndef SSIM_hpp
#define SSIM_hpp

#include "Metric.hpp"

class SSIM : protected Metric {
public:
    SSIM(int height, int width);

    // Compute the SSIM index of the processed image
    float compute(const cv::Mat &original, const cv::Mat &processed);

    float compute(const cv::cuda::GpuMat &original, const cv::cuda::GpuMat &processed);

protected:
    // Compute the SSIM index and mean of the contrast comparison function
    cv::Scalar computeSSIM(const cv::Mat &img1, const cv::Mat &img2);

    cv::Scalar computeSSIM(const cv::cuda::GpuMat &img1, const cv::cuda::GpuMat &img2);

private:
    static const double C1;
    static const double C2;


    cv::cuda::GpuMat gI1_2, gI2_2, gI1_I2;
    std::vector<cv::cuda::GpuMat> gvI1, gvI2;

    cv::cuda::GpuMat gmu1, gmu2;
    cv::cuda::GpuMat gmu1_2, gmu2_2, gmu1_mu2;

    cv::cuda::GpuMat gsigma1_2, gsigma2_2, gsigma12;
    cv::cuda::GpuMat gt3;

    cv::cuda::GpuMat g_ssim_map;

    cv::cuda::GpuMat gbuf;
};

#endif
