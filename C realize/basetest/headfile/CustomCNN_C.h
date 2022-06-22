#pragma once
#ifndef CNN_C_H
#define CNN_C_H

#define NUM_LAYERS 9
#define CNN_KERNEL_SIZE 7
#define CNN_KERNEL_NUM 4

// 输入测试样本大小
#define INPUT_X 96
#define INPUT_Y 96
#define INPUT_C 3
// 第一层卷积核
#define LAYER_1_CNN_W 6
#define LAYER_1_CNN_C 3
#define LAYER_1_CNN_X 7
#define LAYER_1_CNN_Y 7
// W表示输出通道数，C表示输入通道数，X，Y表示坐标

// 第一个池化
#define LAYER_1_POOL_INPUT_X 92
#define LAYER_1_POOL_INPUT_Y 92
#define LAYER_1_POOL_OUTPUT_X 46    // 46
#define LAYER_1_POOL_OUTPUT_Y 46
#define LAYER_1_POOL_C LAYER_1_CNN_W

// 第二层卷积核
#define LAYER_2_CNN_W 12
#define LAYER_2_CNN_C 6
#define LAYER_2_CNN_X 7
#define LAYER_2_CNN_Y 7

// 第二个池化
#define LAYER_2_POOL_INPUT_X 42
#define LAYER_2_POOL_INPUT_Y 42
#define LAYER_2_POOL_OUTPUT_X 21   // 21
#define LAYER_2_POOL_OUTPUT_Y 21
#define LAYER_2_POOL_C LAYER_2_CNN_W


// 展平后节点数量
#define FALTTEN_NODES 5292   // 12*21*21

// 输出类别数量
#define LAYER_3_CLASS_NODES 12



typedef struct CNN_weights
{
	// 卷积层1
    float w_1[LAYER_1_CNN_W][LAYER_1_CNN_C][LAYER_1_CNN_Y][LAYER_1_CNN_X];  
	float b_1[LAYER_1_CNN_W];  
	//	不同尺度信息加权权值
	float w_f_1[CNN_KERNEL_NUM];
    // BN层1
    float bn_w_1[LAYER_1_CNN_W];  
    float bn_b_1[LAYER_1_CNN_W];  
    // 卷积层2
	float w_2[LAYER_2_CNN_W][LAYER_2_CNN_C][LAYER_2_CNN_Y][LAYER_2_CNN_X];  
	float b_2[LAYER_2_CNN_W];
	//	不同尺度信息加权权值
	float w_f_2[CNN_KERNEL_NUM];  
    // BN层2
    float bn_w_2[LAYER_2_CNN_W];  
    float bn_b_2[LAYER_2_CNN_W];  
    // 全连接层1
	float w_4[LAYER_3_CLASS_NODES][FALTTEN_NODES];
	float b_4[LAYER_3_CLASS_NODES];  

	float sample[INPUT_C][INPUT_Y][INPUT_X];

    // 输出特征图
	float output_1[LAYER_1_CNN_W][LAYER_1_POOL_INPUT_Y][LAYER_1_POOL_INPUT_X];  // 6, 92, 92
	float output_1_pool[LAYER_1_CNN_W][LAYER_1_POOL_OUTPUT_Y][LAYER_1_POOL_OUTPUT_X];  // 6 46 46

	float output_2[LAYER_2_CNN_W][LAYER_2_POOL_INPUT_Y][LAYER_2_POOL_INPUT_X];  // 12 42 42 
	float output_2_pool[LAYER_2_CNN_W][LAYER_2_POOL_OUTPUT_Y][LAYER_2_POOL_OUTPUT_X];  // 12, 21, 21

	float faltten[FALTTEN_NODES];  // 5292
	float output_class[LAYER_3_CLASS_NODES];  // 8
    float result[LAYER_3_CLASS_NODES];
	int category[1];
}CNN_weights;

#endif