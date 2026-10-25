# 核心概念速记（模板）

> 用来记录关键变量/概念的定义与来源

## Window（窗口）

- 定义：图像上的小区域，每个线程负责一个窗口
- 来源变量：window_width、window_height、step_x、step_y
- 作用：在窗口内计算关键点与质量分数

## Keypoint（关键点）

- 定义：窗口内“最有代表性”的点（稳定、易匹配）
- 计算方式：亮度质心 + 偏移 + 亚像素微调
- 作用：用于后续匹配、跟踪、光流等任务

## Local Contrast（局部对比度）

- 定义：窗口内最亮与最暗像素的差值
- 公式：local_contrast = max_value - min_value
- 作用：判断局部纹理是否明显（对比度低则不稳定）

## Centroid（质心）

- 定义：亮度加权的“中心位置”
- 公式：centroid_x = Σ(x*I) / Σ(I)，centroid_y = Σ(y*I) / Σ(I)
- 作用：作为关键点位置的核心来源

## Quality Score（质量分数）

- 定义：关键点“可靠性”的评分
- 组成：magnitude × local_contrast
- 作用：用于筛选稳定关键点

## Gradient Strength（Ixx/Iyy/Ixy）

- 定义：窗口内梯度强度的统计量
- 计算方式：Ixx = Σ(gx*gx)，Iyy = Σ(gy*gy)，Ixy = Σ(gx*gy)
- 作用：区分角点与边缘，提高稳定性

## Sub-pixel（亚像素细化）

- 定义：把关键点从“像素级”微调到“亚像素级”
- 来源：基于窗口内亮度分布的加权偏移
- 作用：提高定位精度与匹配稳定性

## Kernel 计算关键点流程（完整步骤）

1. **线程定位窗口**  
   由 `blockIdx/threadIdx` 计算当前窗口坐标 `(x, y)`，再得到窗口左上角  
   `window_start_x = x * step_x`、`window_start_y = y * step_y`。

   notice:窗口坐标 (x, y) 指的是窗口在“窗口矩阵”里的索引位置，不是像素中心。
它会被转换成窗口左上角的像素坐标：
    - `window_start_x = x * step_x`、`window_start_y = y * step_y`
    也就是说：(x, y) 是窗口编号；窗口中心/像素坐标要用 step 换算出来。


2. **边界检查**  
   如果窗口超出图像范围，输出无效关键点并返回（`-1` + `0`）。

3. **第一次扫描窗口（max/min + 可选共享内存）**  
   遍历窗口像素，记录 `max_value` 与 `min_value`，得到  
   `local_contrast = max_value - min_value`。  
   小窗口（`window_size <= 49`）时把像素写入共享内存，减少后续全局访问。

4. **第二次扫描窗口（质心 + 梯度统计）**  
   ### 质心累加：  
     `divident_high_x += img_x * pixel_value`  
     `divident_high_y += img_y * pixel_value`  
     `divisor_high += pixel_value`  
   - 质心作用：质心代表窗口内“亮度重心”，是关键点位置的核心来源。
亮的区域权重大，所以质心会偏向结构更明显的位置，这样选出的点更稳定、更有辨识度。 
   ### 梯度统计（可用于角点/边缘过滤）：  
     `gx = I(x+1) - I(x-1)`  
     `gy = I(y+1) - I(y-1)`  
     `Ixx/Iyy/Ixy` 累加。
    梯度统计：为了判断关键点质量（角点 vs 边缘）
    - 仅靠对比度无法区分“边缘”和“角点”
    - 梯度统计（Ixx/Iyy/Ixy）能判断结构类型
    - 角点更稳定，边缘不稳定，所以需要梯度信息作为质量依据
5. **计算质心与偏移量**  
   `centroid = divident_high / divisor_high`  
   `dx, dy = 2 * (centroid - window_center)`  
   `magnitude = sqrt(dx*dx + dy*dy)`。
   - 计算偏移的原因：将关键点从窗口中心推向结构更明显的位置，提高定位稳定性  
   - 偏移的作用：把关键点从“窗口中心/质心附近”推到更明显的结构位置，提升定位稳定性和区分度  
   - 为什么要这么做：质心只是亮度重心，可能还不够“尖锐”；通过偏移让点沿亮度分布偏向更强区域，同时偏移大小可作为质量评分的一部分  

6. **亚像素细化（仅小窗口）**  
   基于亮度权重计算 `sub_pixel_offset_x/y`，微调关键点位置。

7. **输出关键点与质量分数**  
   - 位置：`keypoint_x/y = centroid + 偏移 + 亚像素修正`  
   - 质量：`keypoint_response = magnitude * local_contrast`  
   （梯度过滤逻辑在代码中保留但被注释）

8. **亮度 I 的来源（贯穿流程）**  
   - CPU 将输入图像转为灰度 `image_8u`  
   - GPU 使用 `convertToFloat32OptimizedKernel` 转为 float，得到 `image_data`  
   - Kernel 中的 `pixel_value` 就是 `image_data` 的像素值，也就是亮度 I  
   - 所以 I 就是当前窗口内像素的灰度值（float）

9. **质心的计算结果（明确公式）**  
   - `centroid_x = Σ(x * I) / Σ(I)`  
   - `centroid_y = Σ(y * I) / Σ(I)`  
   质心是关键点位置的核心来源，后续再加偏移与亚像素修正
