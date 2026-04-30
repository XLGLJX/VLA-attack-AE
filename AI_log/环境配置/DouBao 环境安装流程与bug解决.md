# 环境安装流程与bug解决

## 1. 环境配置流程

### 1.1 检查当前环境
- 查看当前conda环境：`conda env list`
- 检查项目目录结构：`ls -la`

### 1.2 创建并激活conda环境
```bash
conda create -n roboticAttack python=3.10 -y
source activate roboticAttack  # 或使用 conda activate roboticAttack
```

### 1.3 安装PyTorch和相关依赖
```bash
# 安装PyTorch 2.2.0（与flash-attn 2.5.5兼容）
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 安装项目依赖
pip install -e .
```

### 1.4 安装flash-attn
```bash
pip install packaging ninja
ninja --version  # 验证Ninja安装
pip install "flash-attn==2.5.5" --no-build-isolation
```

### 1.5 安装LIBERO评估环境
```bash
# 克隆LIBERO仓库
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git

# 安装LIBERO
cd LIBERO
pip install -e .

# 安装其他依赖
cd ..
pip install -r experiments/robot/libero/libero_requirements.txt
```

### 1.6 下载并配置LIBERO数据集
```bash
# 设置代理（如果需要）
export ALL_PROXY=http://127.0.0.1:7890

# 创建dataset目录
mkdir -p dataset

# 下载LIBERO数据集（从Hugging Face）
cd dataset
git clone https://huggingface.co/datasets/openvla/modified_libero_rlds
cd ..
```

## 2. 遇到的问题与解决方案

### 2.1 conda activate 命令失败
- **问题**：执行 `conda activate roboticAttack` 时出现 `CondaError: Run 'conda init' before 'conda activate'`
- **解决方案**：使用 `source activate roboticAttack` 命令激活环境

### 2.2 PyTorch 版本与flash-attn 不兼容
- **问题**：安装 flash-attn 时出现 `ImportError: /mnt/home/lvmingyuan/.conda/envs/roboticAttack/lib/python3.10/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent`
- **解决方案**：安装 PyTorch 2.2.0 版本，与 flash-attn 2.5.5 兼容

### 2.3 网络访问问题
- **问题**：无法从 GitHub 和 Hugging Face 下载代码和数据集
- **解决方案**：设置代理 `export ALL_PROXY=http://127.0.0.1:7890` 后重新尝试

### 2.4 LIBERO 版本属性问题
- **问题**：导入 LIBERO 后无法访问 `libero.__version__` 属性
- **解决方案**：LIBERO 模块可以正常导入，说明安装成功，版本属性不存在不影响使用

## 3. 环境验证

### 3.1 检查Python环境
```bash
python --version  # 应显示 Python 3.10.x
```

### 3.2 检查PyTorch安装
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 3.3 检查flash-attn安装
```bash
python -c "import flash_attn; print(flash_attn.__version__)"
```

### 3.4 检查LIBERO安装
```bash
python -c "import libero; print(libero.__version__)"
```

## 4. 数据集配置

### 4.1 LIBERO数据集结构
- 数据集应放置在 `dataset/` 目录中
- 预期结构：
  ```
  ├── dataset
  │   └── libero_spatial_no_noops
  │   └── libero_object_no_noops
  │   └── libero_goal_no_noops
  │   └── libero_10_no_noops
  ```

### 4.2 数据集下载方法
1. 从 Hugging Face 下载：https://huggingface.co/datasets/openvla/modified_libero_rlds
2. 解压到 `dataset/` 目录
3. 确保目录结构符合要求

## 5. 总结

- 成功创建并配置了 roboticAttack 环境
- 安装了 PyTorch 2.2.0 和 flash-attn 2.5.5
- 安装了 LIBERO 评估环境
- 遇到的主要问题是 PyTorch 版本兼容性和数据集下载问题
- 环境已准备就绪，可以开始进行对抗攻击的相关实验
