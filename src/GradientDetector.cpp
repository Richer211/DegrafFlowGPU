

#include "stdafx.h"
#include "GradientDetector.h"
#include <chrono>

using namespace std; 

//! Class constructor
GradientDetector::GradientDetector()
{
    // Initialise variables
    image_size.width = 0;
    image_size.height = 0;
    window_size.width = 0;
    window_size.height = 0;
    step_x = 0;
    step_y = 0;
    matrix_size.width = 0;
    matrix_size.height = 0;
    init_flag = false;
}

//! Class destructor
GradientDetector::~GradientDetector()
{
    Release();
}

//! Initialises class
/*!
  \param p_image source image
  \param p_window_width width of gradient window
  \param p_window_height height of gradient window
  \param p_step_x step in x direction for moving window
  \param p_step_y step in y direction for moving window
*/
void GradientDetector::Create(IplImage *p_image, int p_window_width, int p_window_height, int p_step_x, int p_step_y)
{
    // Local Variables
    int i, j, x, y;

    if (init_flag == false)
    {
        // Initialise variables
        image_size.width = p_image->width;
        image_size.height = p_image->height;
        window_size.width = p_window_width;
        window_size.height = p_window_height;
        step_x = p_step_x;
        step_y = p_step_y;
        matrix_size.width = static_cast<int>(floor((double)(image_size.width - window_size.width) / (double)p_step_x));
        matrix_size.height = static_cast<int>(floor((double)(image_size.height - window_size.height) / (double)p_step_y));
        std::cout << "GradientDetector::Create - Image size: " << p_image->width << "x" << p_image->height << std::endl;
        std::cout << "Image depth: " << p_image->depth << ", channels: " << p_image->nChannels << std::endl;

        std::cout << "CPU matrix_size: " << matrix_size.width << "x" << matrix_size.height 
          << " = " << matrix_size.width * matrix_size.height << std::endl;
        // Allocate memory
        gradient_matrix = (Gradient **)malloc(matrix_size.height * sizeof(Gradient *));
        x = 0;
        y = 0;
        for (i = 0; i < matrix_size.height; i++)
        {
            gradient_matrix[i] = (Gradient *)malloc(matrix_size.width * sizeof(Gradient));
            for (j = 0; j < matrix_size.width; j++)
            {
                // Initialise variables
                gradient_matrix[i][j].active = false;
                gradient_matrix[i][j].is_keypoint = false;
                gradient_matrix[i][j].x = j * step_x;
                gradient_matrix[i][j].y = i * step_y;
                gradient_matrix[i][j].centre.x = (float)gradient_matrix[i][j].x + ((float)window_size.width / 2.0f);
                gradient_matrix[i][j].centre.y = (float)gradient_matrix[i][j].y + ((float)window_size.height / 2.0f);
                gradient_matrix[i][j].centroid.x = gradient_matrix[i][j].centre.x;
                gradient_matrix[i][j].centroid.y = gradient_matrix[i][j].centre.y;
                gradient_matrix[i][j].low_centroid.x = static_cast<int>(round(gradient_matrix[i][j].centroid.x));
                gradient_matrix[i][j].low_centroid.y = static_cast<int>(round(gradient_matrix[i][j].centroid.y));
                gradient_matrix[i][j].high_centroid.x = static_cast<int>(round(gradient_matrix[i][j].centroid.x));
                gradient_matrix[i][j].high_centroid.y = static_cast<int>(round(gradient_matrix[i][j].centroid.y));
                gradient_matrix[i][j].magnitude = 0.0;
                gradient_matrix[i][j].angle = 0.0;
                gradient_matrix[i][j].pos_neg_ratio = 0.0;
                gradient_matrix[i][j].xy_ratio = 0.0;
                gradient_matrix[i][j].average = 0.0;
                gradient_matrix[i][j].weight = 0.0;
            }
        }

        // Use cv::Mat instead of IplImage
        image_8u_mat = cv::Mat::zeros(image_size.height, image_size.width, CV_8UC1);
        image_f32_mat = cv::Mat::zeros(image_size.height, image_size.width, CV_32FC1);
        std::cout << "About to call cvSetData for image_8u" << std::endl;
        std::cout << "Mat size: " << image_8u_mat.cols << "x" << image_8u_mat.rows << std::endl;
        std::cout << "Mat step: " << image_8u_mat.step << std::endl;
        // Create IplImage wrappers for compatibility using proper OpenCV 4.x method
        image_8u = cvCreateImageHeader(cvSize(image_size.width, image_size.height), IPL_DEPTH_8U, 1);
        int step_8u = image_8u_mat.cols * image_8u_mat.elemSize();
        cvSetData(image_8u, image_8u_mat.data, step_8u);

        image_f32 = cvCreateImageHeader(cvSize(image_size.width, image_size.height), IPL_DEPTH_32F, 1);
        int step_f32 = image_f32_mat.cols * image_f32_mat.elemSize();
        cvSetData(image_f32, image_f32_mat.data, step_f32);

        // Clear keypoint buffer
        keypoints.clear();

        // Set initialisation flag
        init_flag = true;
    }
}

//! Releases memory
void GradientDetector::Release(void)
{
    // Local Variables
    int i;

    // Release Memory
    if (init_flag == true)
    {
        // Clear keypoint buffer
        keypoints.clear();

        for (i = 0; i < matrix_size.height; i++)
        {
            free(gradient_matrix[i]);
        }
        free(gradient_matrix);

        // Release IplImage wrappers (cv::Mat handles memory automatically)
        if (image_8u)
        {
            cvReleaseImageHeader(&image_8u);
        }
        if (image_f32)
        {
            cvReleaseImageHeader(&image_f32);
        }

        // Reset initialisation flag
        init_flag = false;
    }
}

//! Detects gradients
/*!
  \param p_image_src source image
  \param p_window_width width of gradient window
  \param p_window_height height of gradient window
  \param p_step_x step in x direction for moving window
  \param p_step_y step in y direction for moving window
  \param p_oriented_gradients enable/disable oriented gradients
  \return function status (0: failure, 1: success)
*/
int GradientDetector::DetectGradients(IplImage *p_image_src, int p_window_width, int p_window_height, int p_step_x, int p_step_y)
{
    // Local variables
    int i, j, x, y, counter;
    cv::Point2f divident, divident_high, divident_low;
    float divisor, divisor_high, divisor_low, pixel_value, dx, dx2, dy, dy2, max_value;

    // Check image
    if (p_image_src == NULL || p_image_src->width < 1 || p_image_src->height < 1)
    {
        return (0);
    }

    // Check initialisation status
    if (init_flag == false)
    {
        Create(p_image_src, p_window_width, p_window_height, p_step_x, p_step_y);
    }

    // Check image parameters
    if (p_image_src->width != image_size.width || p_image_src->height != image_size.height || p_window_width != window_size.width || p_window_height != window_size.height || p_step_x != step_x || p_step_y != step_y)
    {
        // Reset class parameters
        printf("GRAD RELEASE     ");
        Release();
        Create(p_image_src, p_window_width, p_window_height, p_step_x, p_step_y);
    }

    // Convert image using cv::Mat
    cv::Mat src_mat = cv::cvarrToMat(p_image_src);

    if (p_image_src->nChannels > 1)
    {
        cv::cvtColor(src_mat, image_8u_mat, cv::COLOR_RGB2GRAY);
        image_8u_mat.convertTo(image_f32_mat, CV_32FC1, 1.0f, 1.0f);
    }
    else
    {
        if (src_mat.type() != CV_8UC1)
        {
            src_mat.convertTo(image_8u_mat, CV_8UC1);
            image_8u_mat.convertTo(image_f32_mat, CV_32FC1, 1.0f, 1.0f);
        }
        else
        {
            src_mat.convertTo(image_f32_mat, CV_32FC1, 1.0f, 1.0f);
        }
    }

    // Clear keypoint buffer
    keypoints.clear();

    // Detect gradient in each window
    for (y = 0; y < matrix_size.height; y++)
    {
        for (x = 0; x < matrix_size.width; x++)
        {
            divident_high.x = 0.0f;
            divident_high.y = 0.0f;
            divident_low.x = 0.0f;
            divident_low.y = 0.0f;
            divisor_high = 0.0f;
            divisor_low = 0.0f;
            counter = 0;
            max_value = 0.0f;
            for (i = gradient_matrix[y][x].y; i <= gradient_matrix[y][x].y + window_size.height; i++)
            {
                for (j = gradient_matrix[y][x].x; j <= gradient_matrix[y][x].x + window_size.width; j++)
                {
                    pixel_value = image_f32_mat.at<float>(i, j);
                    if (pixel_value > max_value)
                    {
                        max_value = pixel_value;
                    }
                }
            }

            for (i = gradient_matrix[y][x].y; i <= gradient_matrix[y][x].y + window_size.height; i++)
            {
                for (j = gradient_matrix[y][x].x; j <= gradient_matrix[y][x].x + window_size.width; j++)
                {
                    pixel_value = image_f32_mat.at<float>(i, j);

                    if (i == 18 && j == 396)
                    {
                        // printf("Pixel value at %i , %i is %f \n", i, j, pixel_value);
                    }

                    divident_high.x += (float)j * pixel_value;
                    divident_high.y += (float)i * pixel_value;
                    divisor_high += pixel_value;
                    divident_low.x += (float)j * (max_value + 1 - pixel_value);
                    divident_low.y += (float)i * (max_value + 1 - pixel_value);
                    divisor_low += max_value + 1 - pixel_value;
                    counter++;
                }
            }

            // Set dominant divident and divisor
            if (1) // divisor_high > divisor_low)
            {
                divident.x = divident_high.x;
                divident.y = divident_high.y;
                divisor = divisor_high;
            }
            else
            {
                divident.x = divident_low.x;
                divident.y = divident_low.y;
                divisor = divisor_low;
            }

            gradient_matrix[y][x].average = divisor_high / (float)(counter);
            gradient_matrix[y][x].centroid.x = divident.x / divisor;
            gradient_matrix[y][x].centroid.y = divident.y / divisor;
            dx = 2.0f * (gradient_matrix[y][x].centroid.x - gradient_matrix[y][x].centre.x);
            dx2 = dx * dx;
            dy = 2.0f * (gradient_matrix[y][x].centroid.y - gradient_matrix[y][x].centre.y);
            dy2 = dy * dy;
            gradient_matrix[y][x].magnitude = pow(dx2 + dy2, 0.5f);
            gradient_matrix[y][x].angle = cv::fastAtan2(dy, dx);
            gradient_matrix[y][x].pos_neg_ratio = std::min(divisor_low / divisor_high, divisor_high / divisor_low);
            gradient_matrix[y][x].dx = dx;
            gradient_matrix[y][x].dy = dy;
            if (dy != 0 && dx != 0)
            {
                gradient_matrix[y][x].xy_ratio = std::min(fabs(dx / dy), fabs(dy / dx));
            }
            else
            {
                gradient_matrix[y][x].pos_neg_ratio = 0.0f;
            }
            gradient_matrix[y][x].weight = gradient_matrix[y][x].average; //(gradient_matrix[y][x].pos_neg_ratio + gradient_matrix[y][x].magnitude + gradient_matrix[y][x].average)/3.0;

            // Reset keypoint status
            gradient_matrix[y][x].is_keypoint = false;

            // Extract keypoints
            // For DeGraF-Flow do not down select points, leaves a uniform grid of points which is good for interpolation
            // if (gradient_matrix[y][x].pos_neg_ratio > 0.30)
            //{
            keypoints.push_back(cv::KeyPoint(cv::Point2f(gradient_matrix[y][x].centroid.x + dx, gradient_matrix[y][x].centroid.y + dy), (float)std::min(window_size.width, window_size.height)));
            //}

            /*gradient_matrix[y][x].active = false;
            if(fabs((float)gradient_matrix[y][x].centre.x - gradient_matrix[y][x].centroid.x) >= 1.0f ||
                    fabs((float)gradient_matrix[y][x].centre.y - gradient_matrix[y][x].centroid.y) >= 1.0f)
            {
                gradient_matrix[y][x].active = true;
                gradient_matrix[y][x].centre.x = gradient_matrix[y][x].centroid.x;
                gradient_matrix[y][x].centre.y = gradient_matrix[y][x].centroid.y;
                gradient_matrix[y][x].x = std::min(image_8u_mat.cols - window_size.width, std::max(0, static_cast<int>(round(gradient_matrix[y][x].centroid.x - (float)(window_size.width/2)))));
                gradient_matrix[y][x].y = std::min(image_8u_mat.rows - window_size.height, std::max(0, static_cast<int>(round(gradient_matrix[y][x].centroid.y - (float)(window_size.height/2)))));
            }*/
        }
    }

    return (1);
}