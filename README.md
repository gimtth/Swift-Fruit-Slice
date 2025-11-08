# 体感切水果游戏 🍉✋

一个使用 MediaPipe 和 OpenCV 开发的体感互动游戏，通过手部挥动轨迹切水果！

## 🎮 游戏特色

- **手势识别**：使用 MediaPipe 实时追踪手部动作
- **真实素材**：支持9种水果的PNG图片素材，带透明通道
- **刀光效果**：OpenCV 绘制红色渐变光带，模拟刀光轨迹
- **真实切割**：使用实际的左右切半图片，展现真实切水果效果
- **物理效果**：水果旋转、碎片飞散、渐隐动画
- **动态难度**：水果生成速度逐渐加快，挑战你的反应速度
- **实时反馈**：显示分数、漏接数和摄像头画面

## 🎯 游戏规则

1. 水果从屏幕底部随机抛出
2. 挥动食指，让轨迹碰到水果即可切开
3. 每切一个水果得 10 分
4. 漏接 5 个水果游戏结束

## 📋 系统要求

- Python 3.8+
- 摄像头
- Windows/Linux/MacOS

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行游戏

```bash
python main.py
```

### 3. 控制说明

- **挥动食指**：切水果
- **R 键**：重新开始游戏
- **Q 键**：退出游戏

## 🎨 游戏画面

- 主游戏区域：彩色水果和刀光轨迹
- 右上角小窗口：实时摄像头画面
- 左上角：分数和漏接计数
- 游戏结束时显示最终分数

## 🍎 水果素材包

游戏包含9种水果，每种水果有3张PNG图片（带透明通道）：

| 水果类型 | 完整图片 | 左半边 | 右半边 |
|---------|---------|--------|--------|
| 香蕉 (banana) | banana.png | bananal.png | bananar.png |
| 菠萝 (boluo) | boluo.png | boluol.png | boluor.png |
| 冰香蕉 (iceBanana) | iceBanana.png | iceBananal.png | iceBananar.png |
| 芒果 (Mango) | Mango.png | Mangol.png | Mangor.png |
| 木瓜 (mugua) | mugua.png | mugual.png | muguar.png |
| 桃子 (peach) | peach.png | peachl.png | peachr.png |
| 梨 (pear) | pear.png | pearl.png | pearr.png |
| 菠萝 (pineapple) | pineapple.png | pineapplel.png | pineappler.png |
| 草莓 (strawberry) | strawberry.png | strawberryl.png | strawberryr.png |

所有素材存放在 `素材包/` 目录下。

## 🔧 技术实现

### 核心技术栈
- **MediaPipe Hands**：手部21个关键点实时追踪
- **OpenCV**：图像处理、碰撞检测、PNG透明通道处理、图片旋转
- **NumPy**：数值计算和矩阵操作

### 关键功能
- 手指轨迹队列（deque）记录最近20个位置点
- 物理引擎模拟水果抛物线运动和重力效果
- PNG图片叠加：支持alpha通道透明度混合
- 切割系统：使用真实的左右半边图片，模拟真实切割效果
- 旋转动画：水果在空中自然旋转，切开后碎片旋转飞散
- 透明度渐变：刀光和碎片有渐变消失效果

## 🎯 玩法提示

1. 保持手部在摄像头视野内
2. 快速挥动食指效果最好
3. 注意光线充足，提高识别准确度
4. 可以画圆圈、直线、Z字等各种轨迹

## 🔨 自定义修改

可以在代码中调整以下参数：

```python
# 游戏窗口尺寸
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# 水果生成间隔（帧数）
spawn_interval = 60

# 最大漏接数
max_missed = 5

# 手部检测置信度
min_detection_confidence=0.7
```

## 📝 更新日志

- v1.0 (2025-11-06)
  - 初始版本发布
  - 实现基础手势识别和切水果功能
  - 添加刀光轨迹和碎片动画效果
  - 完整的游戏逻辑和UI

## 💡 未来优化方向

- [x] ~~使用真实水果图片素材~~ (已完成 ✅)
- [ ] 添加音效和背景音乐
- [ ] 增加更多水果种类和特殊道具（炸弹、冰冻等）
- [ ] 添加连击系统（Combo）
- [ ] 保存历史最高分和排行榜
- [ ] 多人对战模式
- [ ] 添加果汁飞溅特效

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 开源协议

MIT License

---

**祝你游戏愉快！🎉**

