// File description
/*!
  Copyright \htmlonly &copy \endhtmlonly 2008-2011 Cranfield University
  \file SaliencyDetector.h
  \brief SaliencyDetector class header
  \author Ioannis Katramados
*/

// Pragmas
#pragma once

// Include Files
// #include "VisionerLibDefs.h"
// #include "VisionerLibTypes.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h> // For legacy support
#include "ImagePyramid.h"
#include "ImageArray.h"

#include <iostream> // standard C++ I/O FS for debugging
#include <string>

//! A class for detecting visual saliency
class SaliencyDetector
{
private:
  // Private variables
  int init_status;
  int image_depth;
  cv::Size image_size;
  uint pyramid_height;

protected:
public:
  // Public Variables
  ImagePyramid *pyramid, *pyramid_inv;
  ImageArray *image_3ch;
  IplImage *image_8u; // Compatibility interface
  IplImage *matrix_ratio, *matrix_ratio_inv, *matrix_min_ratio, *unit_matrix;
  IplImage *saliency_matrix;

  // Modern storage (for internal use)
  cv::Mat image_8u_mat;
  cv::Mat matrix_ratio_mat, matrix_ratio_inv_mat, matrix_min_ratio_mat, unit_matrix_mat;
  cv::Mat saliency_matrix_mat;
  IplImage *temp_image_32f; // Temporary IplImage wrapper

  // Constructor & Destructor
  SaliencyDetector();
  ~SaliencyDetector();

  // Public Function Prototypes
  // DoGoS_Saliency 是DoG_Saliency的改进版，主要是为了提高计算效率和准确性。
  void Create(IplImage *p_image_src, uint p_pyr_levels);
  void Release(void);
  int DIVoG_Saliency(IplImage *p_image_src, IplImage *p_image_dest = NULL, int p_pyr_levels = 3, bool p_filter = false, bool p_norm = false);
  int DoGoS_Saliency(IplImage *p_image_src, IplImage *p_image_dest = NULL, int p_pyr_levels = 3, bool p_filter = false, bool p_norm = false);
  int CheckImage(IplImage *p_image_src, int p_pyr_levels);
};