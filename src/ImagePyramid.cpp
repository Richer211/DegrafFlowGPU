
// Include Files
#include "stdafx.h"
#include "ImagePyramid.h"

using namespace std;

//! Class constructor
ImagePyramid::ImagePyramid()
{
    const int DEFAULT_IMAGE_WIDTH = 640;  // new
    const int DEFAULT_IMAGE_HEIGHT = 480; // new
    init_status = false;
    pyramid_height = 0;
    image_size.width = DEFAULT_IMAGE_WIDTH;
    image_size.height = DEFAULT_IMAGE_HEIGHT;
    image_depth = CV_8U;
}

//! Class destructor
ImagePyramid::~ImagePyramid()
{
    Release();
}

//! Creates a pyramid of images
/*!
  \param p_image source image
  \param p_pyr_levels number of pyramid levels
  \todo malloc should be replaced with "new".
*/
void ImagePyramid::Create(IplImage *p_image, uint p_pyr_levels)
{
    // Local Variables
    int i;

    // Initialise variables
    pyramid_height = p_pyr_levels;

    // Allocate memory
    level_scale = (int *)malloc(sizeof(int) * pyramid_height);
    level_size = (cv::Size *)malloc(sizeof(cv::Size) * pyramid_height);
    level_image = (IplImage **)malloc(sizeof(IplImage *) * pyramid_height);

    // Allocate modern storage
    level_mats.resize(pyramid_height);
    level_ipl_wrappers.resize(pyramid_height);

    // Initialise variables
    image_size = cv::Size(p_image->width, p_image->height);
    level_size[0] = cv::Size(p_image->width, p_image->height);
    // Create first level using cv::Mat
    level_mats[0] = cv::Mat::zeros(level_size[0].height, level_size[0].width,
                                   CV_MAKETYPE(p_image->depth & 0xFF, p_image->nChannels));
    cv::Mat temp_mat_0 = level_mats[0];
    level_ipl_wrappers[0] = cvCreateImageHeader(cvSize(temp_mat_0.cols, temp_mat_0.rows), p_image->depth, p_image->nChannels);
    level_ipl_wrappers[0]->imageData = (char *)temp_mat_0.data;
    level_ipl_wrappers[0]->widthStep = temp_mat_0.step[0];
    level_ipl_wrappers[0]->imageDataOrigin = (char *)temp_mat_0.data;
    level_image[0] = level_ipl_wrappers[0];
    // Copy source image to the bottom of the pyramid
    cv::Mat src_mat = cv::cvarrToMat(p_image);
    src_mat.copyTo(level_mats[0]);

    // Derive any subsequent pyramid level by resolution reduction
    level_scale[0] = 1;
    for (i = 1; i < pyramid_height; i++)
    {
        if (i > 0)
        {
            level_scale[i] = level_scale[i - 1] * 2;
        }
        level_size[i].width = level_size[i - 1].width / 2;   // Set image width for the current pyramid level
        level_size[i].height = level_size[i - 1].height / 2; // Set image height for the current pyramid level

        // Create cv::Mat for current level
        level_mats[i] = cv::Mat::zeros(level_size[i].height, level_size[i].width,
                                       CV_MAKETYPE(p_image->depth & 0xFF, p_image->nChannels));
        cv::Mat temp_mat_i = level_mats[i];
        level_ipl_wrappers[i] = cvCreateImageHeader(cvSize(temp_mat_i.cols, temp_mat_i.rows), p_image->depth, p_image->nChannels);
        level_ipl_wrappers[i]->imageData = (char *)temp_mat_i.data;
        level_ipl_wrappers[i]->widthStep = temp_mat_i.step[0];
        level_ipl_wrappers[i]->imageDataOrigin = (char *)temp_mat_i.data;
        level_image[i] = level_ipl_wrappers[i];
        // Perform pyramidal resolution reduction using cv::pyrDown
        cv::pyrDown(level_mats[i - 1], level_mats[i]);
    }

    init_status = true;
}

//! Builds a pyramid bottom-up
/*!
  \param p_image source image
  \param p_pyr_levels pyramid height
  \param p_scale scaling factor that is applied to the source array elements
  \param p_shift value added to the scaled source array elements
  \return function status (0: failure, 1: success)
*/
int ImagePyramid::BuildPyramidUp(IplImage *p_image, int p_pyr_levels, double p_scale, double p_shift)
{
    // Local Variables
    int i;

    if (CheckImage(p_image, p_pyr_levels))
    {
        if (init_status == true)
        {
            // Copy source image to the bottom of the pyramid
            cv::Mat src_mat = cv::cvarrToMat(p_image);
            src_mat.convertTo(level_mats[0], level_mats[0].type(), p_scale, p_shift);

            // Derive any subsequent pyramid level by resolution reduction
            for (i = 1; i < pyramid_height; i++)
            {
                cv::pyrDown(level_mats[i - 1], level_mats[i]);
            }
            return (true);
        }
    }
    return (false);
}

//! Builds a pyramid top-down
/*!
  \param p_image source image
  \param p_scale scaling factor that is applied to the source array elements
  \param p_shift value added to the scaled source array elements
  \return function status (0: failure, 1: success)
*/
int ImagePyramid::BuildPyramidDown(IplImage *p_image, double p_scale, double p_shift)
{
    // Local Variables
    int i;

    if (init_status == true)
    {
        // Copy source image to the top of the pyramid
        cv::Mat src_mat = cv::cvarrToMat(p_image);
        src_mat.convertTo(level_mats[pyramid_height - 1], level_mats[pyramid_height - 1].type(), p_scale, p_shift);

        // Derive any subsequent pyramid level by resolution increase
        for (i = pyramid_height - 1; i > 0; i--)
        {
            cv::pyrUp(level_mats[i], level_mats[i - 1]);
        }
        return (true);
    }
    return (false);
}

//! Releases memory of a pyramid
void ImagePyramid::Release(void)
{
    int i;

    // Release Memory
    if (init_status == true)
    {
        init_status = false;

        // Clean up IplImage wrappers
        for (i = 0; i < pyramid_height; i++)
        {
            if (level_ipl_wrappers[i])
            {
                cvReleaseImageHeader(&level_ipl_wrappers[i]);
            }
        }
        level_ipl_wrappers.clear();

        // Clear cv::Mat vector (automatic memory management)
        level_mats.clear();

        // Free arrays
        free(level_scale);
        free(level_size);
        free(level_image);
    }
}

//! Checks an image is valid for processing
/*!
  \param p_image_src source image
  \param p_pyr_levels pyramid height
  \return function status (0: failure, 1: success)
*/
int ImagePyramid::CheckImage(IplImage *p_image_src, int p_pyr_levels)
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

//! Copies an image pyramid
/*!
  \param p_src_pyramid source pyramid
  \param p_dest_pyramid destination pyramid
*/
void CopyImagePyramid(ImagePyramid *p_src_pyramid, ImagePyramid *p_dest_pyramid)
{
    // Local Variables
    int i, pyramid_height;

    if (p_src_pyramid->init_status == true)
    {
        // Initialise variables
        if (p_dest_pyramid->init_status == false)
        {
            p_dest_pyramid->Create(p_src_pyramid->level_image[0], p_src_pyramid->pyramid_height);
        }

        // Set pyramid height as the shortest
        pyramid_height = std::min(p_src_pyramid->pyramid_height, p_dest_pyramid->pyramid_height);

        // Copy the image from source to destination for each pyramid level using cv::Mat
        for (i = 0; i < pyramid_height; i++)
        {
            p_src_pyramid->level_mats[i].copyTo(p_dest_pyramid->level_mats[i]);
        }
    }
}