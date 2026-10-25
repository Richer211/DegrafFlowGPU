// File description
/*!
  Copyright \htmlonly &copy \endhtmlonly 2008-2011 Cranfield University
  \file SaliencyDetector.cpp
  \brief SaliencyDetector class implementation
  \author Ioannis Katramados
*/

// Include Files
#include "stdafx.h"
#include "SaliencyDetector.h"

// #include <iostream>	// new
//
// using namespace std;

//! Class constructor
SaliencyDetector::SaliencyDetector()
{
    // Initialise variables
    const int DEFAULT_IMAGE_WIDTH = 640;  // new
    const int DEFAULT_IMAGE_HEIGHT = 480; // new
    init_status = false;
    image_size.width = DEFAULT_IMAGE_WIDTH;
    image_size.height = DEFAULT_IMAGE_HEIGHT;
    image_depth = CV_8U;
    pyramid_height = 3;
}

//! Class destructor
SaliencyDetector::~SaliencyDetector()
{
}

//! Initialises a saliency detector
/*!
  \param p_image source image
  \param p_pyr_levels number of pyramid levels
*/
void SaliencyDetector::Create(IplImage *p_image, uint p_pyr_levels)
{
    // Local Variables
    cv::Mat image_32f_mat;

    // Get source image dimensions
    image_size = cv::Size(p_image->width, p_image->height);
    image_depth = p_image->depth & 0xFF;
    pyramid_height = p_pyr_levels;

    // Setup image templates using cv::Mat
    image_32f_mat = cv::Mat::zeros(image_size.height, image_size.width, CV_32FC1);
    image_8u_mat = cv::Mat::zeros(image_size.height, image_size.width, CV_8UC1);

    // Create IplImage wrappers for compatibility
    temp_image_32f = cvCreateImageHeader(cvSize(image_32f_mat.cols, image_32f_mat.rows), IPL_DEPTH_32F, 1);
    cvSetData(temp_image_32f, image_32f_mat.data, image_32f_mat.step);
    image_8u = cvCreateImageHeader(cvSize(image_8u_mat.cols, image_8u_mat.rows), IPL_DEPTH_8U, 1);
    cvSetData(image_8u, image_8u_mat.data, image_8u_mat.step);

    // Initialise pyramids
    pyramid = new ImagePyramid();
    pyramid->Create(temp_image_32f, pyramid_height);
    pyramid_inv = new ImagePyramid();
    pyramid_inv->Create(temp_image_32f, pyramid_height);
    // Initialise image arrays
    image_3ch = new ImageArray();
    image_3ch->InitArray(temp_image_32f, 3);
    // Initialise images using cv::Mat
    saliency_matrix_mat = cv::Mat::zeros(image_size.height, image_size.width, CV_32FC1);
    matrix_ratio_mat = cv::Mat::zeros(image_size.height, image_size.width, CV_32FC1);
    matrix_ratio_inv_mat = cv::Mat::zeros(image_size.height, image_size.width, CV_32FC1);
    matrix_min_ratio_mat = cv::Mat::zeros(image_size.height, image_size.width, CV_32FC1);
    unit_matrix_mat = cv::Mat::ones(image_size.height, image_size.width, CV_32FC1);

    // Create IplImage wrappers
    saliency_matrix = cvCreateImageHeader(cvSize(saliency_matrix_mat.cols, saliency_matrix_mat.rows), IPL_DEPTH_32F, 1);
    cvSetData(saliency_matrix, saliency_matrix_mat.data, saliency_matrix_mat.step);
    matrix_ratio = cvCreateImageHeader(cvSize(matrix_ratio_mat.cols, matrix_ratio_mat.rows), IPL_DEPTH_32F, 1);
    cvSetData(matrix_ratio, matrix_ratio_mat.data, matrix_ratio_mat.step);
    matrix_ratio_inv = cvCreateImageHeader(cvSize(matrix_ratio_inv_mat.cols, matrix_ratio_inv_mat.rows), IPL_DEPTH_32F, 1);
    cvSetData(matrix_ratio_inv, matrix_ratio_inv_mat.data, matrix_ratio_inv_mat.step);
    matrix_min_ratio = cvCreateImageHeader(cvSize(matrix_min_ratio_mat.cols, matrix_min_ratio_mat.rows), IPL_DEPTH_32F, 1);
    cvSetData(matrix_min_ratio, matrix_min_ratio_mat.data, matrix_min_ratio_mat.step);
    unit_matrix = cvCreateImageHeader(cvSize(unit_matrix_mat.cols, unit_matrix_mat.rows), IPL_DEPTH_32F, 1);
    cvSetData(unit_matrix, unit_matrix_mat.data, unit_matrix_mat.step);
    // Set initialisation flag
    init_status = true;
}

//! Releases saliency detector
void SaliencyDetector::Release(void)
{
    // Free memory
    if (init_status)
    {
        // Delete IplImage wrappers
        cvReleaseImageHeader(&image_8u);
        cvReleaseImageHeader(&saliency_matrix);
        cvReleaseImageHeader(&matrix_ratio);
        cvReleaseImageHeader(&matrix_ratio_inv);
        cvReleaseImageHeader(&matrix_min_ratio);
        cvReleaseImageHeader(&unit_matrix);
        cvReleaseImageHeader(&temp_image_32f);

        // Release pyramids and arrays
        pyramid->Release();
        pyramid_inv->Release();
        image_3ch->ReleaseArray();

        // Delete pyramid and array objects
        delete pyramid;
        delete pyramid_inv;
        delete image_3ch;

        // Reset initialisation flag
        init_status = false;
    }
}

//! Calculates per-pixel visual saliency using Division of Gaussians (DIVoG)
/*!
  \param p_image_src source image
  \param p_image_dest destination image
  \param p_pyr_levels pyramid height
  \param p_filter filter activation flag
  \param p_norm normalisation flag
  \return function status (0: failure, 1: success)
*/
int SaliencyDetector::DIVoG_Saliency(IplImage *p_image_src, IplImage *p_image_dest, int p_pyr_levels, bool p_filter, bool p_norm)
{
    // Local Variables
    cv::Scalar avg;

    // Check input
    if (CheckImage(p_image_src, p_pyr_levels))
    {
        // Convert to grayscale using cv::Mat
        cv::Mat src_mat = cv::cvarrToMat(p_image_src);
        cv::cvtColor(src_mat, image_8u_mat, cv::COLOR_RGB2GRAY);

        // Create a pyramid of resolutions. Shift image by 2^n to avoid division
        // by zero or any number in the range 0.0 - 1.0;
        pyramid->BuildPyramidUp(image_8u, p_pyr_levels, 1.0, static_cast<double>(1 << pyramid_height));
        pyramid_inv->BuildPyramidDown(pyramid->level_image[pyramid_height - 1]);

        // Calculate Minimum Ratio (MiR) matrix using cv::Mat operations
        cv::Mat level0_mat = cv::cvarrToMat(pyramid->level_image[0]);
        cv::Mat level_inv0_mat = cv::cvarrToMat(pyramid_inv->level_image[0]);

        cv::divide(level0_mat, level_inv0_mat, matrix_ratio_mat);
        cv::divide(level_inv0_mat, level0_mat, matrix_ratio_inv_mat);
        cv::min(matrix_ratio_mat, matrix_ratio_inv_mat, matrix_min_ratio_mat);

        // Derive salience by subtracting from unit matrix
        cv::subtract(unit_matrix_mat, matrix_min_ratio_mat, saliency_matrix_mat);
        saliency_matrix_mat.convertTo(image_8u_mat, CV_8UC1, 255.0);

        // Low-pass filter
        if (p_filter)
        {
            avg = cv::mean(image_8u_mat);
            cv::subtract(image_8u_mat, avg, image_8u_mat);
        }

        // Normalization to range 0-255
        if (p_norm)
        {
            cv::normalize(image_8u_mat, image_8u_mat, 0, 255, cv::NORM_MINMAX);
        }

        // Generate output if a destination image is given as function parameter
        if (p_image_dest != NULL)
        {
            cv::Mat dest_mat = cv::cvarrToMat(p_image_dest);
            if (p_image_src->nChannels == 1) // Process grayscale saliency matrix
            {
                image_8u_mat.copyTo(dest_mat);
            }
            else // Process colour saliency matrix
            {
                // Convert to colour
                cv::cvtColor(image_8u_mat, dest_mat, cv::COLOR_GRAY2RGB);
            }
        }
        return (true);
    }
    return (false);
}

//! Calculates per-pixel visual saliency using Difference of Gaussians (DoGoS)
/*!
  \param p_image_src source image
  \param p_image_dest destination image
  \param p_pyr_levels pyramid height
  \param p_filter filter activation flag
  \param p_norm normalisation flag
  \return function status (0: failure, 1: success)
*/
int SaliencyDetector::DoGoS_Saliency(IplImage *p_image_src, IplImage *p_image_dest, int p_pyr_levels, bool p_filter, bool p_norm)
{
    // Local Variables
    cv::Scalar avg;

    // Check input
    if (CheckImage(p_image_src, p_pyr_levels))
    {
        // Convert to grayscale using cv::Mat
        cv::Mat src_mat = cv::cvarrToMat(p_image_src);
        cv::cvtColor(src_mat, image_8u_mat, cv::COLOR_RGB2GRAY);

        // Create a pyramid of resolutions. Shift image by 2^n to avoid division
        // by zero or any number in the range 0.0 - 1.0;
        pyramid->BuildPyramidUp(image_8u, p_pyr_levels, 1.0, 1.0);
        pyramid_inv->BuildPyramidDown(pyramid->level_image[pyramid_height - 1]);

        // Calculate Minimum Ratio (MiR) matrix using cv::Mat operations
        cv::Mat level0_mat = cv::cvarrToMat(pyramid->level_image[0]);
        cv::Mat level_inv0_mat = cv::cvarrToMat(pyramid_inv->level_image[0]);

        // 确保类型一致，转换为float类型
        cv::Mat level0_f32, level_inv0_f32;
        level0_mat.convertTo(level0_f32, CV_32F);
        level_inv0_mat.convertTo(level_inv0_f32, CV_32F);
        // cv::absdiff:是做什么的？计算两张图的差值，cv::add:是做什么的？计算两张图的和，cv::divide:是做什么的？计算两张图的比值
        cv::absdiff(level0_f32, level_inv0_f32, matrix_ratio_mat); // 计算两张图的差值, 也就是两张图的差异性
        cv::add(level_inv0_f32, level0_f32, matrix_ratio_inv_mat); // 计算两张图的和, 也就是两张图的相似性
        cv::divide(matrix_ratio_mat, matrix_ratio_inv_mat, saliency_matrix_mat); // 计算两张图的比值, 也就是两张图的差异性除以相似性
        saliency_matrix_mat.convertTo(image_8u_mat, CV_8UC1, 255.0);

        // Low-pass filter
        if (p_filter)
        {
            avg = cv::mean(image_8u_mat);
            cv::subtract(image_8u_mat, avg, image_8u_mat);
        }

        // Normalization to range 0-255
        if (p_norm)
        {
            cv::normalize(image_8u_mat, image_8u_mat, 0, 255, cv::NORM_MINMAX);
        }

        // Generate output if a destination image is given as function parameter
        if (p_image_dest != NULL)
        {
            cv::Mat dest_mat = cv::cvarrToMat(p_image_dest);
            if (p_image_src->nChannels == 1) // Process grayscale saliency matrix
            {
                image_8u_mat.copyTo(dest_mat);
            }
            else // Process colour saliency matrix
            {
                // Convert to colour
                cv::cvtColor(image_8u_mat, dest_mat, cv::COLOR_GRAY2RGB);
            }
        }
        return (true);
    }
    return (false);
}

//! Checks an image is valid for processing
/*!
  \param p_image_src source image
  \param p_pyr_levels pyramid height
  \return function status (0: failure, 1: success)
*/
int SaliencyDetector::CheckImage(IplImage *p_image_src, int p_pyr_levels)
{
    // Check input
    if (p_image_src == NULL)
    {
        return (false);
    }

    // Detect changes to image properties
    if (init_status == true)
    {
        if (p_image_src->width != image_size.width || p_image_src->height != image_size.height ||
            (p_image_src->depth & 0xFF) != image_depth || p_pyr_levels != pyramid_height)
        {
            // Release memory
            Release();
        }
    }

    // Check initialisation status
    if (init_status == false)
    {
        Create(p_image_src, p_pyr_levels);
    }

    return (true);
}