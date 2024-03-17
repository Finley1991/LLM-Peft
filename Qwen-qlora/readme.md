## python 版本 
3.10.13
## 依赖环境：
pip install -r requirements.txt

## 运行方式
1. 配置model_download.py中的模型下载路径
2. 运行model_download.py下载模型
3. 修改config/config.json中的参数，例如修改模型路径，数据路径等
4. 运行 `sh run.sh` 或者 `python train_qlora.py --train_args_file configs/qwen_config.json` 运行程序