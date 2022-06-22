#include <iostream>
#include <fstream>
#include "time.h"
// #include "CNN_tools.h"
#include "CustomCNN_tools.h"
using namespace std;

int main()
{
	
	CNN_weights model = {
		#include "CustomCNN_weights.h"
		#include "sample3.h"
		,{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}
	};
	// CNN_weights model = {
	// 	0
	// };


	// 传入数据与模型
	// 样本--标签
	// sample1 -- 1
	// sample2 -- 3
	// sample3 -- 7
	// sample4 -- 10
	int label = 7;
	printf("\n该样本真实标签: %d \n", label);
	Inference(&model);
	// Inference(&input[0][0][0], &model);
	printf("\n预测概率向量为: \n");
	for (int i = 0; i < LAYER_3_CLASS_NODES; i++)
	{
		cout << model.result[i] << ' ';
		// cout << model.output_class[i] << ' ';
	}
	printf("\n预测标签: %d \n", model.category[0]);
	cin.get();
	return 0;
}