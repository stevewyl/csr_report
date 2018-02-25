# csr_report
CSR报告清理脚本使用说明

运行环境：Python 3.6
如何使用：打开命令行窗口，cd到脚本根目录，输入 python clean.py 输入目录名 输出目录名

脚本说明：
主要分为以下几个模块：
1. 文件的批量读取
2. 中文的转换（繁转简，全角转半角）
3. 清理
	a. 去除无意义符号或词
	b. 替换转化错误的字（可在脚本根目录新建keywords.txt， 格式为每行 错误的字\t正确的字 如：曰	日）
	c. 去除非标准空行
	d. 去除报告的开头内容
	e. 去除表格内容
	f. 切分过长行
4. 提取
	a. 多行段落
	b. 多行段落的最后一行
	c. 单行段落
	d. 目录信息（暂时不能使用）
	e. 标题信息（暂时不能使用）
5. 合并
6. 输出结果
	a. 清理后的文本（保存在输出目录）
	b. 转化失败的文件名（保存在当前目录下的result.txt）
