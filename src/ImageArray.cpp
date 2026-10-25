

// Include Files
#include "stdafx.h"
#include "ImageArray.h"

//! Class constructor
ImageArray::ImageArray()
{
    init_flag = false;
}

//! Class destructor
ImageArray::~ImageArray()
{
}

//! Creates an image array
/*!
  \param p_image source image
  \param p_length number of image array elements
  \todo malloc should be replaced with "new".
*/
void ImageArray::InitArray(IplImage *p_image, uint p_length)
{
    // Local Variables
    int i;

    // Initialise variables
    array_length = p_length;

    // Allocate memory using modern approach
    image_mats.resize(array_length);
    image = (IplImage **)malloc(sizeof(IplImage *) * array_length);

    // Initialise variables
    image_size = cv::Size(p_image->width, p_image->height); // Get source image size

    // Derive any subsequent pyramid level by resolution reduction
    for (i = 0; i < array_length; i++)
    {
        // Create cv::Mat and initialize
        image_mats[i] = cv::Mat::zeros(image_size.height, image_size.width,
                                       CV_MAKETYPE(p_image->depth & 0xFF, p_image->nChannels));

        // Create IplImage wrapper for compatibility - FIX: Convert cv::Size to CvSize
        CvSize sz = cvSize(image_size.width, image_size.height);
        temp_ipl_images.push_back(cvCreateImageHeader(sz, p_image->depth, p_image->nChannels));
        // Manually set the IplImage data pointer to avoid cvSetData
        temp_ipl_images.back()->imageData = (char *)image_mats[i].data;
        temp_ipl_images.back()->widthStep = image_mats[i].step[0];
        temp_ipl_images.back()->imageDataOrigin = (char *)image_mats[i].data;
        image[i] = temp_ipl_images.back();
    }

    init_flag = true;
}

//! Releases memory of an image array
void ImageArray::ReleaseArray(void)
{
    int i;

    // Release Memory
    if (init_flag == true)
    {
        // Clean up IplImage wrappers
        for (auto &ipl_ptr : temp_ipl_images)
        {
            cvReleaseImageHeader(&ipl_ptr);
        }
        temp_ipl_images.clear();

        // Clear cv::Mat vector (automatic memory management)
        image_mats.clear();

        // Free the pointer array
        free(image);

        init_flag = false;
    }
}

//! Copies an image array
/*!
  \param p_src_array source array
  \param p_dest_array destination array
*/
void CopyImageArray(ImageArray *p_src_array, ImageArray *p_dest_array)
{
    // Local Variables
    int i, array_length;

    if (p_src_array->init_flag == true)
    {
        // Initialise variables
        if (p_dest_array->init_flag == false)
        {
            p_dest_array->InitArray(p_src_array->image[0], p_src_array->array_length);
        }

        // Set pyramid height as the shortest
        array_length = std::min(p_src_array->array_length, p_dest_array->array_length);

        // Copy the image from source to destination for each pyramid level.
        for (i = 0; i < array_length; i++)
        {
            // Use cv::Mat copy for better performance and safety
            p_src_array->image_mats[i].copyTo(p_dest_array->image_mats[i]);
        }
    }
}