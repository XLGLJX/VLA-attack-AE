# SwanLab 使用指南

## 快速开始

### 1. 登录 SwanLab

**方式一：命令行登录**
```bash
swanlab login
# 输入你的 API Key
```

**方式二：环境变量**
```bash
export SWANLAB_API_KEY="your_api_key_here"
```

### 2. 获取 API Key

1. 访问 https://swanlab.cn
2. 注册/登录账号
3. 点击右上角头像 → 设置 → API Keys
4. 复制你的 API Key

### 3. 运行攻击脚本

```bash
# TMA 攻击
bash scripts/run_TMA.sh

# UADA 攻击
bash scripts/run_UADA.sh

# UPA 攻击
bash scripts/run_UPA.sh
```

### 4. 查看实验结果

访问 https://swanlab.cn 查看实验日志和可视化结果

---

## 参数说明

### 启用/禁用日志

```bash
# 启用（默认）
--swanlab_project "VLA-Attack"

# 禁用
--swanlab_project "false"
```

### 自定义项目名

```bash
--swanlab_project "my_custom_project"
```

---

## 需要提供的信息

请提供以下信息以完成配置：

1. **SwanLab API Key**（必需）
   - 从 https://swanlab.cn 获取

2. **项目名称**（可选）
   - 默认：`VLA-Attack`
   - 可通过 `--swanlab_project` 参数修改

---

## 迁移完成清单

✅ 已完成：
- SwanLab 安装
- 所有 wrapper 文件修改（TMA, UADA, UPA, UADA_ddp）
- 所有核心攻击文件修改（TMA.py, UADA.py, UPA.py, UADA_ddp.py）
- 运行脚本更新
- 迁移日志创建

⏳ 待完成：
- 提供 SwanLab API Key
- 运行 `swanlab login` 登录
- 测试运行验证功能
