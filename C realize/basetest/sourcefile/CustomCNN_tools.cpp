#include "CustomCNN_tools.h"
#include <iostream>
using namespace std;

/*卷积计算*/
inline float Conv_1x1_compute(float *x1, float *x2, int x, int y, int c, int X, int Y)
{
    float prod = 0;
	int pos = (X*Y)*c + (y+0) * X + (x+0);

	prod += x1[pos - (0*X + 0)] * x2[0];
    
	return prod;
}


/*卷积计算*/
inline float Conv_3x3_compute(float *x1, float *x2, int x, int y, int c, int X, int Y)
{
    float prod = 0;
	int pos = (X*Y)*c + (y+1) * X + (x+1);

	prod += x1[pos - (1*X + 1)] * x2[0];
	prod += x1[pos - (1*X + 0)] * x2[1];
	prod += x1[pos - (1*X - 1)] * x2[2];

	prod += x1[pos - (0*X + 1)] * x2[3];
	prod += x1[pos - (0*X + 0)] * x2[4];
	prod += x1[pos - (0*X - 1)] * x2[5];

	prod += x1[pos + (1*X - 1)] * x2[6];
	prod += x1[pos + (1*X - 0)] * x2[7];
	prod += x1[pos + (1*X + 1)] * x2[8];
    
	return prod;
}

/*卷积计算*/
inline float Conv_5x5_compute(float *x1, float *x2, int x, int y, int c, int X, int Y)
{
    float prod = 0;
	int pos = (X*Y)*c + (y+2) * X + (x+2);	// located to center

	prod += x1[pos - (2*X + 2)] * x2[0];
	prod += x1[pos - (2*X + 1)] * x2[1];
	prod += x1[pos - (2*X + 0)] * x2[2];
	prod += x1[pos - (2*X - 1)] * x2[3];
	prod += x1[pos - (2*X - 2)] * x2[4];

	prod += x1[pos - (1*X + 2)] * x2[5];
	prod += x1[pos - (1*X + 1)] * x2[6];
	prod += x1[pos - (1*X + 0)] * x2[7];
	prod += x1[pos - (1*X - 1)] * x2[8];
	prod += x1[pos - (1*X - 2)] * x2[9];

	prod += x1[pos - (0*X + 2)] * x2[10];
	prod += x1[pos - (0*X + 1)] * x2[11];
	prod += x1[pos - (0*X + 0)] * x2[12];
	prod += x1[pos - (0*X - 1)] * x2[13];
	prod += x1[pos - (0*X - 2)] * x2[14];

	prod += x1[pos + (1*X - 2)] * x2[15];
	prod += x1[pos + (1*X - 1)] * x2[16];
	prod += x1[pos + (1*X - 0)] * x2[17];
	prod += x1[pos + (1*X + 1)] * x2[18];
	prod += x1[pos + (1*X + 2)] * x2[19];

	prod += x1[pos + (2*X - 2)] * x2[20];
	prod += x1[pos + (2*X - 1)] * x2[21];
	prod += x1[pos + (2*X - 0)] * x2[22];
	prod += x1[pos + (2*X + 1)] * x2[23];
	prod += x1[pos + (2*X + 2)] * x2[24];

	return prod;
}


/*卷积计算*/
inline float Conv_7x7_compute(float *x1, float *x2, int x, int y, int c, int X, int Y)
{
    float prod = 0;
	int pos = (X*Y)*c + (y+3) * X + (x+3);	// located to center

	prod += x1[pos - (3*X + 3)] * x2[0];
	prod += x1[pos - (3*X + 2)] * x2[1];
	prod += x1[pos - (3*X + 1)] * x2[2];
	prod += x1[pos - (3*X + 0)] * x2[3];
	prod += x1[pos - (3*X - 1)] * x2[4];
	prod += x1[pos - (3*X - 2)] * x2[5];
	prod += x1[pos - (3*X - 3)] * x2[6];

	prod += x1[pos - (2*X + 3)] * x2[7];
	prod += x1[pos - (2*X + 2)] * x2[8];
	prod += x1[pos - (2*X + 1)] * x2[9];
	prod += x1[pos - (2*X + 0)] * x2[10];
	prod += x1[pos - (2*X - 1)] * x2[11];
	prod += x1[pos - (2*X - 2)] * x2[12];
	prod += x1[pos - (2*X - 3)] * x2[13];

	prod += x1[pos - (1*X + 3)] * x2[14];
	prod += x1[pos - (1*X + 2)] * x2[15];
	prod += x1[pos - (1*X + 1)] * x2[16];
	prod += x1[pos - (1*X + 0)] * x2[17];
	prod += x1[pos - (1*X - 1)] * x2[18];
	prod += x1[pos - (1*X - 2)] * x2[19];
	prod += x1[pos - (1*X - 3)] * x2[20];

	prod += x1[pos - (0*X + 3)] * x2[21];
	prod += x1[pos - (0*X + 2)] * x2[22];
	prod += x1[pos - (0*X + 1)] * x2[23];
	prod += x1[pos - (0*X + 0)] * x2[24];
	prod += x1[pos - (0*X - 1)] * x2[25];
	prod += x1[pos - (0*X - 2)] * x2[26];
	prod += x1[pos - (0*X - 3)] * x2[27];

	prod += x1[pos + (1*X - 3)] * x2[28];
	prod += x1[pos + (1*X - 2)] * x2[29];
	prod += x1[pos + (1*X - 1)] * x2[30];
	prod += x1[pos + (1*X - 0)] * x2[31];
	prod += x1[pos + (1*X + 1)] * x2[32];
	prod += x1[pos + (1*X + 2)] * x2[33];
	prod += x1[pos + (1*X + 3)] * x2[34];

	prod += x1[pos + (2*X - 3)] * x2[35];
	prod += x1[pos + (2*X - 2)] * x2[36];
	prod += x1[pos + (2*X - 1)] * x2[37];
	prod += x1[pos + (2*X - 0)] * x2[38];
	prod += x1[pos + (2*X + 1)] * x2[39];
	prod += x1[pos + (2*X + 2)] * x2[40];
	prod += x1[pos + (2*X + 3)] * x2[41];

	prod += x1[pos + (3*X - 3)] * x2[42];
	prod += x1[pos + (3*X - 2)] * x2[43];
	prod += x1[pos + (3*X - 1)] * x2[44];
	prod += x1[pos + (3*X - 0)] * x2[45];
	prod += x1[pos + (3*X + 1)] * x2[46];
	prod += x1[pos + (3*X + 2)] * x2[47];
	prod += x1[pos + (3*X + 3)] * x2[48];

	return prod;
}


/*矩阵点积*/
float compute_innerprod(float x1[], float x2[], int len)
{
	float prod = 0;
	int i = 0;

	for (i = 0; i < len; i++)
	{
		prod += x1[i] * x2[i];
	}
	return prod;
}

/*ReLu激活函数*/
void relu_activation(float inp[], int len)
{
	int i = 0;
	for (i = 0; i < len; i++)
	{
		if (inp[i] < 0.0f)
		{
			inp[i] = 0.0f;
		}
	}
}

void flatternMatrix(float inp[], float oup[], int len)
{
	int i = 0;
	for (i = 0; i < len; i++)
	{
		oup[i] = inp[i];
	}
}

/*Softmax函数*/
void soft_max(float input[], float result[], int len)
{
	float maxval = input[0];
	float sum = 0.0f;
	int i = 0;
	// 找最大值
	for (i = 1; i < len; i++)
	{
		if (input[i] > maxval) maxval = input[i];
	}
	// 计算指数函数及指数函数和
	for (i = 0; i < len; i++)
	{
		input[i] = (float)exp((double)(input[i] - maxval)); // exp()以double作为输入输出
		sum += input[i];
	}
	for (i = 0; i < len; i++)
	{
		result[i] = input[i] / sum;
	}
}

/*BN*/
void bn(float inp[], float op[], int len, float bnw, float bnb)
{
	int i;
	float sum = 0, sumsq = 0, eps = 1e-5, mean = 0, sigma = 0;
	// 计算均值
	for (i = 0; i < len; i++)
	{
		sum += inp[i];
	}
	mean = sum / len;
	// 计算方差
	for (i = 0; i < len; i++)
	{
		sumsq += (inp[i] - mean) * (inp[i] - mean);
	}
	sigma = (float)sqrt((double)(sumsq / len + eps));
	for (i = 0; i < len; i++)
	{
		op[i] = bnw * (inp[i] - mean) / sigma + bnb;
	}
}

inline float max_of_two(float a, float b)
{
	if (a >= b) return a;
	else return b;
}

inline float max_of_three(float a, float b, float c)
{
	return max_of_two(max_of_two(a, b), c);
}

inline float max_of_four(float a, float b, float c, float d)
{
	return max_of_two(max_of_three(a, b, c), d);
}


void Inference(CNN_weights* pANN_weights)
{
	float prod = 0.0;
	int w = 0, c = 0, x = 0, y = 0;
	float inputDpadding[INPUT_C][INPUT_Y + 2][INPUT_X + 2] = {0}; //+2是实现填充0的效果
	for (c = 0; c < INPUT_C; c++)	//int k = 0; k < INPUT_C; k++
	{
		for (y = 1; y <= INPUT_Y; y++)
		{
			for (x = 1; x <= INPUT_X; x++)
			{
				inputDpadding[c][y][x] = pANN_weights->sample[c][y-1][x-1];
			}
			
		}
	}
	// float sample[INPUT_C][INPUT_Y + 2][INPUT_X + 2] = {include "sample1.h}";
	// 网络结构：卷积->BN->池化->卷积->BN->池化->全连接输出
	memset(pANN_weights->output_1, 0, sizeof(pANN_weights->output_1));
    // 上面语句的意思为将pANN_weights->output_1赋值为0
	for (w = 0; w < LAYER_1_CNN_W; w++)
	{
		for (y = 0; y < LAYER_1_POOL_INPUT_Y; y++)
		{
			for (x = 0; x < LAYER_1_POOL_INPUT_X; x++)
			{
				pANN_weights->output_1[w][y][x] = pANN_weights->b_1[w];
				for (c = 0; c < LAYER_1_CNN_C; c++)	//依次用卷积核对输入通道卷积，然后再求和
				{
					prod = Conv_7x7_compute(&inputDpadding[0][0][0], &pANN_weights->w_1[w][c][0][0], x, y, c, INPUT_X + 2, INPUT_Y + 2);
					pANN_weights->output_1[w][y][x] += prod;
				}
				pANN_weights->output_1[w][y][x] *= pANN_weights->w_f_1[0];
			}
		}
	}

	// 第二种尺寸的卷积
	float weight55[LAYER_1_CNN_W][LAYER_1_CNN_C][LAYER_1_CNN_Y - 2][LAYER_1_CNN_X - 2] = {0};
	// 第二种增广filter
	for (w = 0; w < LAYER_1_CNN_W; w++)
	{
		for (c = 0; c < LAYER_1_CNN_C; c++)
		{
			for (y = 1; y < LAYER_1_CNN_Y - 1; y++)
			{
				for (x = 1; x < LAYER_1_CNN_X - 1; x++)	
				{
					weight55[w][c][y-1][x-1] = pANN_weights->w_1[w][c][y][x];
				}
			}
		}
	}
	float inputD2[LAYER_1_CNN_C][INPUT_Y][INPUT_X] = {0};
	for (c = 0; c < LAYER_1_CNN_C; c++)	//int k = 0; k < INPUT_C; k++
	{
		for (y = 0; y < INPUT_Y; y++)
		{
			for (x = 0; x < INPUT_X; x++)
			{
				inputD2[c][y][x] = pANN_weights->sample[c][y][x];
			}
			
		}
	}
	float temOutFeature[LAYER_1_CNN_W][LAYER_1_POOL_INPUT_Y][LAYER_1_POOL_INPUT_X] = {0};
	for (w = 0; w < LAYER_1_CNN_W; w++)
	{
		for (y = 0; y < LAYER_1_POOL_INPUT_Y; y++)
		{
			for (x = 0; x < LAYER_1_POOL_INPUT_X; x++)
			{
				temOutFeature[w][y][x] = pANN_weights->b_1[w];
				for (c = 0; c < LAYER_1_CNN_C; c++)	//依次用卷积核对输入通道卷积，然后再求和
				{
					prod = Conv_5x5_compute(&inputD2[0][0][0], &weight55[w][c][0][0], x, y, c, INPUT_X, INPUT_Y);
					temOutFeature[w][y][x] += prod;
				}
				pANN_weights->output_1[w][y][x] += pANN_weights->w_f_1[1] * temOutFeature[w][y][x];
			}
		}
	}

	// 第三种尺寸的卷积
	float weight33[LAYER_1_CNN_W][LAYER_1_CNN_C][LAYER_1_CNN_Y - 2*2][LAYER_1_CNN_X - 2*2] = {0};
	// 第三种增广filter
	for (w = 0; w < LAYER_1_CNN_W; w++)
	{
		for (c = 0; c < LAYER_1_CNN_C; c++)
		{
			for (y = 2; y < LAYER_1_CNN_Y - 2; y++)
			{
				for (x = 2; x < LAYER_1_CNN_X - 2; x++)	
				{
					weight33[w][c][y-2][x-2] = pANN_weights->w_1[w][c][y][x];
				}
			}
		}
	}
	float inputD3[LAYER_1_CNN_C][LAYER_1_POOL_INPUT_Y + 2][LAYER_1_POOL_INPUT_X + 2] = {0};
	for (c = 0; c < LAYER_1_CNN_C; c++)	//int k = 0; k < INPUT_C; k++
	{
		for (y = 1; y < LAYER_1_POOL_INPUT_Y + 1; y++)
		{
			for (x = 1; x < LAYER_1_POOL_INPUT_X + 1; x++)
			{
				inputD3[c][y][x] = pANN_weights->sample[c][y+1][x+1];
			}
			
		}
	}	
	temOutFeature[LAYER_1_CNN_W][LAYER_1_POOL_INPUT_Y][LAYER_1_POOL_INPUT_X] = {0};
	for (w = 0; w < LAYER_1_CNN_W; w++)
	{
		for (y = 0; y < LAYER_1_POOL_INPUT_Y; y++)
		{
			for (x = 0; x < LAYER_1_POOL_INPUT_X; x++)
			{
				temOutFeature[w][y][x] = pANN_weights->b_1[w];
				for (c = 0; c < LAYER_1_CNN_C; c++)	//依次用卷积核对输入通道卷积，然后再求和
				{
					prod = Conv_3x3_compute(&inputD3[0][0][0], &weight33[w][c][0][0], x, y, c, LAYER_1_POOL_INPUT_X + 2, LAYER_1_POOL_INPUT_Y + 2);
					temOutFeature[w][y][x] += prod;
				}
				pANN_weights->output_1[w][y][x] += pANN_weights->w_f_1[2] * temOutFeature[w][y][x];
			}
		}
	}


	// 第四种尺寸的卷积
	float weight11[LAYER_1_CNN_W][LAYER_1_CNN_C][LAYER_1_CNN_Y - 2*3][LAYER_1_CNN_X - 2*3] = {0};
	// 第三种增广filter
	for (w = 0; w < LAYER_1_CNN_W; w++)
	{
		for (c = 0; c < LAYER_1_CNN_C; c++)
		{
			for (y = 3; y < LAYER_1_CNN_Y - 3; y++)
			{
				for (x = 3; x < LAYER_1_CNN_X - 3; x++)	
				{
					weight11[w][c][y-3][x-3] = pANN_weights->w_1[w][c][y][x];
				}
			}
		}
	}
	float inputD4[LAYER_1_CNN_C][LAYER_1_POOL_INPUT_Y][LAYER_1_POOL_INPUT_X]= {0};
	for (c = 0; c < LAYER_1_CNN_C; c++)	//int k = 0; k < INPUT_C; k++
	{
		for (y = 1; y < LAYER_1_POOL_INPUT_Y - 1; y++)
		{
			for (x = 1; x < LAYER_1_POOL_INPUT_X - 1; x++)
			{
				inputD4[c][y][x] = pANN_weights->sample[c][y+2][x+2];
			}
			
		}
	}	
	temOutFeature[LAYER_1_CNN_W][LAYER_1_POOL_INPUT_Y][LAYER_1_POOL_INPUT_X] = {0};
	for (w = 0; w < LAYER_1_CNN_W; w++)
	{
		for (y = 0; y < LAYER_1_POOL_INPUT_Y; y++)
		{
			for (x = 0; x < LAYER_1_POOL_INPUT_X; x++)
			{
				temOutFeature[w][y][x] = pANN_weights->b_1[w];
				for (c = 0; c < LAYER_1_CNN_C; c++)	//依次用卷积核对输入通道卷积，然后再求和
				{
					prod = Conv_1x1_compute(&inputD4[0][0][0], &weight11[w][c][0][0], x, y, c, LAYER_1_POOL_INPUT_X, LAYER_1_POOL_INPUT_Y);
					temOutFeature[w][y][x] += prod;
				}
				pANN_weights->output_1[w][y][x] += pANN_weights->w_f_1[3] * temOutFeature[w][y][x];
			}
		}
	}
	// BN层，先计算均值，方差，然后再计算BN后输出数据
	for (w = 0; w < LAYER_1_CNN_W; w++)
	{
		bn(&pANN_weights->output_1[w][0][0], &pANN_weights->output_1[w][0][0], LAYER_1_POOL_INPUT_Y*LAYER_1_POOL_INPUT_X, pANN_weights->bn_w_1[w], pANN_weights->bn_b_1[w]);
	}

	// 激活
	relu_activation(&pANN_weights->output_1[0][0][0], LAYER_1_POOL_INPUT_Y * LAYER_1_POOL_INPUT_X * LAYER_1_CNN_W);

    // 池化
	for (w = 0; w < LAYER_1_CNN_W; w++)
	{
		for (y = 0; y < LAYER_1_POOL_INPUT_Y; y+=2)
		{
			for (x = 0; x < LAYER_1_POOL_INPUT_X; x+=2)
			{
				pANN_weights->output_1_pool[w][y/2][x/2] = max_of_four(pANN_weights->output_1[w][y][x],
															  pANN_weights->output_1[w][y][x + 1],
															  pANN_weights->output_1[w][y + 1][x],
															  pANN_weights->output_1[w][y + 1][x + 1]);
			}
		}
	}

	// 第二层卷积
	memset(pANN_weights->output_2, 0, sizeof(pANN_weights->output_2));
	float feature1[LAYER_1_CNN_W][LAYER_1_POOL_OUTPUT_Y + 2][LAYER_1_POOL_OUTPUT_X + 2] = {0}; //+2是实现填充0的效果
	for (int w = 0; w < LAYER_1_CNN_W; w++)	
	{
		for (int y = 1; y <= LAYER_1_POOL_OUTPUT_Y; y++)
		{
			for (int x = 1; x <= LAYER_1_POOL_OUTPUT_X; x++)
			{
				feature1[w][y][x] = pANN_weights->output_1_pool[w][y-1][x-1];
			}			
		}
	}
	// 计算卷积
	for (w = 0; w < LAYER_2_CNN_W; w++)
	{
		for (y = 0; y < LAYER_2_POOL_INPUT_Y; y++)
		{
			for (x = 0; x < LAYER_2_POOL_INPUT_X; x++)
			{
				pANN_weights->output_2[w][y][x] = pANN_weights->b_2[w];
				for (c = 0; c < LAYER_1_CNN_W; c++)	//依次用卷积核对输入通道卷积，然后再求和
				{
					
					prod = Conv_7x7_compute(&feature1[0][0][0], &pANN_weights->w_2[w][c][0][0], x, y, c, LAYER_1_POOL_OUTPUT_X + 2, LAYER_1_POOL_OUTPUT_Y + 2);
					pANN_weights->output_2[w][y][x] += prod;
				}
				pANN_weights->output_2[w][y][x] *= pANN_weights->w_f_2[0];
			}
		}
	}

	// 第二种尺寸的卷积
	float weight552[LAYER_2_CNN_W][LAYER_2_CNN_C][LAYER_2_CNN_Y - 2][LAYER_2_CNN_X - 2] = {0};
	// 第二种增广filter
	for (w = 0; w < LAYER_2_CNN_W; w++)
	{
		for (c = 0; c < LAYER_2_CNN_C; c++)
		{
			for (y = 1; y < LAYER_2_CNN_Y - 1; y++)
			{
				for (x = 1; x < LAYER_2_CNN_X - 1; x++)	
				{
					weight552[w][c][y-1][x-1] = pANN_weights->w_2[w][c][y][x];
				}
			}
		}
	}
	float inputD22[LAYER_2_CNN_C][LAYER_1_POOL_OUTPUT_Y][LAYER_1_POOL_OUTPUT_X] = {0}; 
	for (c = 0; c < LAYER_2_CNN_C; c++)	//int k = 0; k < INPUT_C; k++
	{
		for (y = 0; y < LAYER_1_POOL_OUTPUT_Y; y++)
		{
			for (x = 0; x < LAYER_1_POOL_OUTPUT_X; x++)
			{
				inputD22[c][y][x] = pANN_weights->output_1_pool[w][y][x];
			}
			
		}
	}	
	float temOutFeature2[LAYER_2_CNN_W][LAYER_2_POOL_INPUT_Y][LAYER_2_POOL_INPUT_X] = {0};
	for (w = 0; w < LAYER_2_CNN_W; w++)
	{
		for (y = 0; y < LAYER_2_POOL_INPUT_Y; y++)
		{
			for (x = 0; x < LAYER_2_POOL_INPUT_X; x++)
			{
				temOutFeature2[w][y][x] = pANN_weights->b_2[w];
				for (c = 0; c < LAYER_2_CNN_C; c++)	//依次用卷积核对输入通道卷积，然后再求和
				{
					prod = Conv_5x5_compute(&inputD22[0][0][0], &weight552[w][c][0][0], x, y, c, LAYER_1_POOL_OUTPUT_X, LAYER_1_POOL_OUTPUT_Y);
					temOutFeature2[w][y][x] += prod;
				}
				pANN_weights->output_2[w][y][x] += pANN_weights->w_f_2[1] * temOutFeature2[w][y][x];
			}
		}
	}

	// 第三种尺寸的卷积
	float weight332[LAYER_2_CNN_W][LAYER_2_CNN_C][LAYER_2_CNN_Y - 2*2][LAYER_2_CNN_X - 2*2] = {0};
	// 第三种增广filter
	for (w = 0; w < LAYER_2_CNN_W; w++)
	{
		for (c = 0; c < LAYER_2_CNN_C; c++)
		{
			for (y = 2; y < LAYER_2_CNN_Y - 2; y++)
			{
				for (x = 2; x < LAYER_2_CNN_X - 2; x++)	
				{
					weight332[w][c][y-2][x-2] = pANN_weights->w_2[w][c][y][x];
				}
			}
		}
	}
	float inputD32[LAYER_2_CNN_C][LAYER_2_POOL_INPUT_Y + 2][LAYER_2_POOL_INPUT_X + 2] = {0};
	for (c = 0; c < LAYER_2_CNN_C; c++)	//int k = 0; k < INPUT_C; k++
	{
		for (y = 1; y < LAYER_2_POOL_INPUT_Y + 1; y++)
		{
			for (x = 1; x < LAYER_2_POOL_INPUT_X + 1; x++)
			{
				inputD32[c][y][x] = pANN_weights->output_1_pool[c][y+1][x+1];
			}
			
		}
	}	
	temOutFeature2[LAYER_2_CNN_W][LAYER_2_POOL_INPUT_Y][LAYER_2_POOL_INPUT_X] = {0};
	for (w = 0; w < LAYER_2_CNN_W; w++)
	{
		for (y = 0; y < LAYER_2_POOL_INPUT_Y; y++)
		{
			for (x = 0; x < LAYER_2_POOL_INPUT_X; x++)
			{
				temOutFeature2[w][y][x] = pANN_weights->b_2[w];
				for (c = 0; c < LAYER_2_CNN_C; c++)	//依次用卷积核对输入通道卷积，然后再求和
				{
					prod = Conv_3x3_compute(&inputD32[0][0][0], &weight332[w][c][0][0], x, y, c, LAYER_2_POOL_INPUT_X + 2, LAYER_2_POOL_INPUT_Y + 2);
					temOutFeature2[w][y][x] += prod;
				}
				pANN_weights->output_2[w][y][x] += pANN_weights->w_f_2[2] * temOutFeature2[w][y][x];
			}
		}
	}


	// 第四种尺寸的卷积
	float weight112[LAYER_2_CNN_W][LAYER_2_CNN_C][LAYER_2_CNN_Y - 2*3][LAYER_2_CNN_X - 2*3] = {0};
	// 第三种增广filter
	for (w = 0; w < LAYER_2_CNN_W; w++)
	{
		for (c = 0; c < LAYER_2_CNN_C; c++)
		{
			for (y = 3; y < LAYER_2_CNN_Y - 3; y++)
			{
				for (x = 3; x < LAYER_2_CNN_X - 3; x++)	
				{
					weight112[w][c][y-3][x-3] = pANN_weights->w_2[w][c][y][x];
				}
			}
		}
	}
	float inputD42[LAYER_2_CNN_C][LAYER_2_POOL_INPUT_Y][LAYER_2_POOL_INPUT_X]= {0};
	for (c = 0; c < LAYER_2_CNN_C; c++)	//int k = 0; k < INPUT_C; k++
	{
		for (y = 1; y < LAYER_2_POOL_INPUT_Y - 1; y++)
		{
			for (x = 1; x < LAYER_2_POOL_INPUT_X - 1; x++)
			{
				inputD42[c][y][x] = pANN_weights->output_1_pool[c][y+2][x+2];
			}
			
		}
	}	
	temOutFeature2[LAYER_2_CNN_W][LAYER_2_POOL_INPUT_Y][LAYER_2_POOL_INPUT_X] = {0};
	for (w = 0; w < LAYER_2_CNN_W; w++)
	{
		for (y = 0; y < LAYER_2_POOL_INPUT_Y; y++)
		{
			for (x = 0; x < LAYER_2_POOL_INPUT_X; x++)
			{
				temOutFeature2[w][y][x] = pANN_weights->b_2[w];
				for (c = 0; c < LAYER_2_CNN_C; c++)	//依次用卷积核对输入通道卷积，然后再求和
				{
					prod = Conv_1x1_compute(&inputD42[0][0][0], &weight112[w][c][0][0], x, y, c, LAYER_2_POOL_INPUT_X, LAYER_2_POOL_INPUT_Y);
					temOutFeature2[w][y][x] += prod;
				}
				pANN_weights->output_2[w][y][x] += pANN_weights->w_f_2[3] * temOutFeature2[w][y][x];
			}
		}
	}
	// BN层，先计算均值，方差，然后再计算BN后输出数据
	for (w = 0; w < LAYER_2_CNN_W; w++)
	{
		bn(&pANN_weights->output_2[w][0][0], &pANN_weights->output_2[w][0][0], LAYER_2_POOL_INPUT_Y*LAYER_2_POOL_INPUT_X, pANN_weights->bn_w_2[w], pANN_weights->bn_b_2[w]);
	}
	// 激活
	relu_activation(&pANN_weights->output_2[0][0][0], LAYER_2_POOL_INPUT_Y * LAYER_2_POOL_INPUT_X * LAYER_2_CNN_W);

    // 池化
    //池化层，后续改进可以写成支持非整数除的，也就是需要向上取整那种
	for (w = 0; w < LAYER_2_CNN_W; w++)
	{
		for (y = 0; y < LAYER_2_POOL_INPUT_Y; y+=2)
		{
			for (x = 0; x < LAYER_2_POOL_INPUT_X; x+=2)
			{
				pANN_weights->output_2_pool[w][y/2][x/2] = max_of_four(pANN_weights->output_2[w][y][x],
															  pANN_weights->output_2[w][y][x + 1],
															  pANN_weights->output_2[w][y + 1][x],
															  pANN_weights->output_2[w][y + 1][x + 1]);
			}
		}
	}

	// 全连接
	flatternMatrix(&pANN_weights->output_2_pool[0][0][0], &pANN_weights->faltten[0], FALTTEN_NODES);
	memset(pANN_weights->output_class, 0, sizeof(pANN_weights->output_class));
	for (w = 0; w < LAYER_3_CLASS_NODES; w++)
	{
		pANN_weights->output_class[w] = pANN_weights->b_4[w];
		prod = compute_innerprod(&pANN_weights->faltten[0], &pANN_weights->w_4[w][0], FALTTEN_NODES);
		pANN_weights->output_class[w] += prod;
	}
    
	soft_max(pANN_weights->output_class, pANN_weights->result, LAYER_3_CLASS_NODES);
	// 最大值所对的索引为类别标签
	pANN_weights->category[0] = 0;
	float temMax = pANN_weights->result[0];
	for (w = 1; w < LAYER_3_CLASS_NODES; w++)
	{
		if (pANN_weights->result[w] > temMax)
		{
			pANN_weights->category[0] = w;
			temMax = pANN_weights->result[w];
		}
	}
}