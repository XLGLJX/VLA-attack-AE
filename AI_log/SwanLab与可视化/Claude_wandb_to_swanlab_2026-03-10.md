# SwanLab 迁移日志

**日期**: 2026-03-10
**任务**: 将训练数据记录从 WandB 迁移到 SwanLab

---

## 修改内容

### 1. 环境安装
- 已安装 swanlab: `pip install swanlab`
- 安装路径: `/mnt/home/lvmingyuan/.local/bin`

### 2. 代码修改

#### Wrapper 文件 (入口脚本)
- `VLAAttacker/TMA_wrapper.py`
- `VLAAttacker/UADA_wrapper.py`
- `VLAAttacker/UPA_wrapper.py`

**修改点**:
- `import wandb` → `import swanlab`
- `--wandb_project` → `--swanlab_project`
- 移除 `--wandb_entity` 参数
- `wandb.init()` → `swanlab.init()`
- 默认项目名: `VLA-Attack`

#### 核心攻击实现文件
- `VLAAttacker/white_patch/TMA.py`
- `VLAAttacker/white_patch/UADA.py`
- `VLAAttacker/white_patch/UADA_ddp.py`
- `VLAAttacker/white_patch/UPA.py`

**修改点**:
- `import wandb` → `import swanlab`
- `wandb.log()` → `swanlab.log()`

#### 运行脚本
- `scripts/run_TMA.sh`
- `scripts/run_UADA.sh`
- `scripts/run_UPA.sh`

**修改点**:
- `--wandb_project` → `--swanlab_project "VLA-Attack"`
- 移除 `--wandb_entity` 参数

---

## 使用说明

### 首次使用需要登录
```bash
swanlab login
# 或设置环境变量
export SWANLAB_API_KEY="your_api_key"
```

### 获取 API Key
1. 访问 https://swanlab.cn
2. 注册/登录账号
3. 进入设置页面获取 API Key

### 运行攻击脚本
```bash
# TMA 攻击
bash scripts/run_TMA.sh

# UADA 攻击
bash scripts/run_UADA.sh

# UPA 攻击
bash scripts/run_UPA.sh
```

### 禁用日志记录
如需禁用 SwanLab 记录，设置参数：
```bash
--swanlab_project "false"
```

---

## SwanLab vs WandB 主要差异

| 特性 | WandB | SwanLab |
|------|-------|---------|
| 初始化 | `wandb.init(entity=..., project=..., name=...)` | `swanlab.init(project=..., experiment_name=...)` |
| 配置 | `wandb.config = {...}` | `config={...}` 作为参数传入 |
| 日志记录 | `wandb.log({...})` | `swanlab.log({...})` |
| Entity | 需要指定 | 不需要（自动关联账号） |

---

## 注意事项

1. **API Key 管理**: 不要将 API Key 提交到代码仓库
2. **项目名称**: 默认为 `VLA-Attack`，可通过 `--swanlab_project` 修改
3. **兼容性**: SwanLab API 与 WandB 高度相似，迁移成本低
4. **数据可视化**: 访问 https://swanlab.cn 查看实验结果

---

## 验证清单

- [x] 安装 swanlab
- [x] 修改所有 wrapper 文件（TMA, UADA, UPA, UADA_ddp）
- [x] 修改所有核心攻击实现文件（TMA.py, UADA.py, UPA.py, UADA_ddp.py）
- [x] 更新运行脚本（run_TMA.sh, run_UADA.sh, run_UPA.sh）
- [x] 替换所有 wandb.log → swanlab.log
- [x] 替换所有 wandb.Image → swanlab.Image
- [x] 替换所有 args.wandb_project → args.swanlab_project
- [x] 替换所有 use_wandb → use_swanlab
- [x] 创建迁移日志和使用指南
- [ ] 用户提供 API Key
- [ ] 测试运行一次攻击验证功能

## 修改统计

- Wrapper 文件: 4 个（TMA_wrapper.py, UADA_wrapper.py, UPA_wrapper.py, UADA_wrapper_ddp.py）
- 核心实现文件: 4 个（TMA.py, UADA.py, UPA.py, UADA_ddp.py）
- 运行脚本: 3 个（run_TMA.sh, run_UADA.sh, run_UPA.sh）
- 总计修改: 11 个文件

---

## 后续步骤

1. 运行 `swanlab login` 并输入 API Key
2. 执行测试运行验证迁移成功
3. 如有问题，检查 swanlab 版本和 API Key 配置
