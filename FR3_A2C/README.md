FR3_A2C
======

简要说明：
- 基于 FR3_SAC 的目录结构，使用 PyTorch 实现一个简化的 A2C（Actor-Critic）训练/测试脚本。
- 文件：
  - `config.py` - 超参与路径
  - `fr3_env.py` - MuJoCo 环境（与 FR3_SAC 一致）
  - `model.py` - ActorCritic 网络
  - `train.py` - 训练入口
  - `test.py` - 测试/评估入口

使用方法（建议在 conda 环境中安装依赖）：

```bash
pip install -r ../requirements.txt  # 或按需安装 torch mujoco gymnasium
python train.py
# 训练完成后：
python test.py --model_path saved_models/fr3_reach_a2c_final.pt
```

注：这是一个教学/参考实现，默认为单环境单进程版本，性能与稳定性可按需改进（向量化、多进程、GAE、学习率调度等）。
