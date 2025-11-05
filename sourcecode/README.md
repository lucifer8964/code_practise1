## 1 环境配置

**Python 版本**

基于Python 3.9.

**环境依赖下载**

```bash
pip install -r requirements.txt
```

## 2 数据集

c4待测评数据集存储在`/data`文件夹下

在测评过程中使用的验证数据集（reference task）存储在 `/lambada_openai`文件夹下.

## 3 实验

项目实现基于LitGPT，更多说明文档可参考https://github.com/Lightning-AI/litgpt?tab=readme-ov-file#quick-start

获取数据样本影响力:

```bash
python probe_oracle_data_influence.py --data_dir ./data/c4 --out_dir ./result/c4
```
