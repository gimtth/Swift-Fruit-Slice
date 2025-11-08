import cv2
import mediapipe as mp
import numpy as np
import random
import math
import os
import warnings
from collections import deque

# 消除 protobuf 弃用警告
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')
try:
    import cvzone
    HAS_CVZONE = True
except ImportError:
    HAS_CVZONE = False
    print("提示: 安装cvzone可获得更好的文字显示效果 (pip install cvzone)")

# 尝试导入pygame用于音效播放
try:
    import pygame
    pygame.mixer.init()
    HAS_SOUND = True
except ImportError:
    HAS_SOUND = False
    print("提示: 安装pygame可启用音效 (pip install pygame)")

# ============== 坐标平滑处理类 ==============
class FingerSmoother:
    """手指坐标平滑处理类 - 支持多种滤波算法 + 自适应速度调整"""
    
    def __init__(self, method='ewma', alpha=0.5, buffer_size=5, adaptive=True):
        """
        初始化平滑器
        
        Args:
            method: 平滑方法 ('ewma', 'moving_avg', 'kalman')
            alpha: EWMA的平滑系数 (0-1)，越小越平滑但延迟越大
            buffer_size: 移动平均的缓冲区大小
            adaptive: 是否启用自适应平滑（根据速度调整平滑强度）
        """
        self.method = method
        self.alpha = alpha
        self.buffer_size = buffer_size
        self.adaptive = adaptive
        
        # EWMA相关
        self.smoothed_pos = None
        self.prev_raw_pos = None  # 用于计算速度
        
        # 移动平均相关
        self.position_buffer = deque(maxlen=buffer_size)
        
        # 卡尔曼滤波相关
        self.kalman_x = None
        self.kalman_y = None
        if method == 'kalman':
            self._init_kalman()
    
    def _init_kalman(self):
        """初始化卡尔曼滤波器"""
        # 状态：[位置, 速度]
        # 过程噪声（运动模型的不确定性）
        self.kalman_x = {
            'x': 0,      # 位置估计
            'v': 0,      # 速度估计
            'P': [[1, 0], [0, 1]],  # 误差协方差矩阵
            'Q': 0.001,  # 过程噪声 - 降低使追踪更平滑
            'R': 0.1     # 测量噪声 - 提高意味着更信任预测而非测量
        }
        self.kalman_y = {
            'x': 0,
            'v': 0,
            'P': [[1, 0], [0, 1]],
            'Q': 0.001,
            'R': 0.1
        }
    
    def _kalman_update(self, kalman, measurement):
        """简化的卡尔曼滤波更新"""
        # 预测步骤
        kalman['x'] = kalman['x'] + kalman['v']
        kalman['P'][0][0] += kalman['Q']
        kalman['P'][1][1] += kalman['Q']
        
        # 更新步骤
        K = kalman['P'][0][0] / (kalman['P'][0][0] + kalman['R'])  # 卡尔曼增益
        kalman['x'] = kalman['x'] + K * (measurement - kalman['x'])
        kalman['P'][0][0] = (1 - K) * kalman['P'][0][0]
        
        return kalman['x']
    
    def _calculate_speed(self, x, y):
        """计算移动速度（像素/帧）"""
        if self.prev_raw_pos is None:
            self.prev_raw_pos = (x, y)
            return 0
        
        prev_x, prev_y = self.prev_raw_pos
        speed = math.sqrt((x - prev_x)**2 + (y - prev_y)**2)
        self.prev_raw_pos = (x, y)
        return speed
    
    def _get_adaptive_alpha(self, speed):
        """根据速度自适应调整alpha值（更激进的设置）
        
        速度慢时：alpha小（平滑强，延迟大）- 减少抖动
        速度快时：alpha大（平滑弱，延迟小）- 保持响应
        """
        # 速度阈值 - 更激进
        slow_threshold = 3    # 像素/帧（降低阈值）
        fast_threshold = 20   # 像素/帧（降低阈值，更快触发）
        
        if speed < slow_threshold:
            # 慢速：强平滑
            return 0.25
        elif speed > fast_threshold:
            # 快速：几乎完全不平滑
            return 1.0  # 完全使用原始坐标
        else:
            # 中速：线性插值
            ratio = (speed - slow_threshold) / (fast_threshold - slow_threshold)
            return 0.25 + ratio * 0.75  # 从0.25到1.0
    
    def smooth(self, x, y):
        """
        平滑坐标（支持自适应）
        
        Args:
            x, y: 原始坐标
            
        Returns:
            (smoothed_x, smoothed_y): 平滑后的坐标
        """
        # 计算速度（用于自适应）
        speed = self._calculate_speed(x, y) if self.adaptive else 0
        
        if self.method == 'ewma':
            return self._smooth_ewma(x, y, speed)
        elif self.method == 'moving_avg':
            return self._smooth_moving_avg(x, y, speed)
        elif self.method == 'kalman':
            return self._smooth_kalman(x, y)
        else:
            return x, y  # 无平滑
    
    def _smooth_ewma(self, x, y, speed=0):
        """指数加权移动平均（支持自适应）"""
        if self.smoothed_pos is None:
            self.smoothed_pos = (x, y)
            return x, y
        
        # 自适应调整alpha
        alpha = self._get_adaptive_alpha(speed) if self.adaptive and speed > 0 else self.alpha
        
        smooth_x = alpha * x + (1 - alpha) * self.smoothed_pos[0]
        smooth_y = alpha * y + (1 - alpha) * self.smoothed_pos[1]
        self.smoothed_pos = (smooth_x, smooth_y)
        
        return int(smooth_x), int(smooth_y)
    
    def _smooth_moving_avg(self, x, y, speed=0):
        """移动平均（支持自适应）"""
        self.position_buffer.append((x, y))
        
        if len(self.position_buffer) < 2:
            return x, y
        
        # 自适应：快速移动时只使用最近的几个点
        if self.adaptive and speed > 20:
            recent_points = list(self.position_buffer)[-2:]  # 只用最近2个点
        else:
            recent_points = list(self.position_buffer)
        
        avg_x = sum(pos[0] for pos in recent_points) / len(recent_points)
        avg_y = sum(pos[1] for pos in recent_points) / len(recent_points)
        
        return int(avg_x), int(avg_y)
    
    def _smooth_kalman(self, x, y):
        """卡尔曼滤波"""
        if self.kalman_x is None:
            self._init_kalman()
            self.kalman_x['x'] = x
            self.kalman_y['x'] = y
            return x, y
        
        smooth_x = self._kalman_update(self.kalman_x, x)
        smooth_y = self._kalman_update(self.kalman_y, y)
        
        return int(smooth_x), int(smooth_y)
    
    def reset(self):
        """重置平滑器状态"""
        self.smoothed_pos = None
        self.prev_raw_pos = None
        self.position_buffer.clear()
        if self.method == 'kalman':
            self._init_kalman()

# 初始化Mediapipe手部检测
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,  # 使用最简单的模型（0=lite，1=full），提高帧率
    min_detection_confidence=0.6,  # 适中的检测阈值
    min_tracking_confidence=0.7    # 适中的追踪阈值
)

# 游戏窗口尺寸
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# 水果类型定义
FRUIT_TYPES = [
    'banana', 'boluo', 'iceBanana', 'Mango', 
    'mugua', 'peach', 'pear', 'pineapple', 'strawberry', 'b1'
]

# 多部分水果类型定义（切开后分成多片）
MULTI_FRUIT_TYPES = [
    'watermelon',  # 西瓜：8个切片
    'dragonfruit'  # 火龙果：8个切片
]

# 加载所有水果图片
def load_fruit_images():
    """加载素材包中的所有水果图片"""
    fruit_images = {}
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(script_dir, 'sucai')
    
    print(f"\n正在加载水果素材...")
    print(f"素材目录: {assets_dir}")
    
    for fruit_name in FRUIT_TYPES:
        # 加载完整水果
        whole_path = os.path.join(assets_dir, f'{fruit_name}.png')
        
        # 特殊处理：b1水果使用bl和br命名
        if fruit_name == 'b1':
            left_path = os.path.join(assets_dir, 'bl.png')
            right_path = os.path.join(assets_dir, 'br.png')
        else:
            left_path = os.path.join(assets_dir, f'{fruit_name}l.png')
            right_path = os.path.join(assets_dir, f'{fruit_name}r.png')
        
        if os.path.exists(whole_path):
            whole_img = cv2.imread(whole_path, cv2.IMREAD_UNCHANGED)
            left_img = cv2.imread(left_path, cv2.IMREAD_UNCHANGED)
            right_img = cv2.imread(right_path, cv2.IMREAD_UNCHANGED)
            
            # 缩放图片到合适大小（原图可能太大）
            if whole_img is not None:
                scale = 1.0  # 保持原图大小
                whole_img = cv2.resize(whole_img, None, fx=scale, fy=scale)
            if left_img is not None:
                left_img = cv2.resize(left_img, None, fx=scale, fy=scale)
            if right_img is not None:
                right_img = cv2.resize(right_img, None, fx=scale, fy=scale)
            
            fruit_images[fruit_name] = {
                'whole': whole_img,
                'left': left_img,
                'right': right_img
            }
            print(f"  ✓ 已加载: {fruit_name}")
        else:
            print(f"  ✗ 未找到: {fruit_name}")
    
    return fruit_images

# 加载水果图片（全局变量）
FRUIT_IMAGES = load_fruit_images()

# 加载多部分水果图片
def load_multi_fruit_images():
    """加载多部分水果图片（切开后分成多片）"""
    multi_fruit_images = {}
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(script_dir, 'sucai')
    
    print(f"\n正在加载多部分水果素材...")
    
    # 缩放比例（让西瓜和火龙果与普通水果大小一致）
    scale_watermelon = 0.5  # 西瓜缩小到50%
    scale_dragonfruit = 0.5  # 火龙果缩小到50%
    
    # 加载西瓜
    watermelon_whole = os.path.join(assets_dir, 'watermelon.png')
    if os.path.exists(watermelon_whole):
        whole_img = cv2.imread(watermelon_whole, cv2.IMREAD_UNCHANGED)
        if whole_img is not None:
            # 缩放完整西瓜图片
            whole_img = cv2.resize(whole_img, None, fx=scale_watermelon, fy=scale_watermelon)
            
            pieces = []
            for i in range(1, 9):
                piece_path = os.path.join(assets_dir, f'watermelon{i}.png')
                if os.path.exists(piece_path):
                    piece_img = cv2.imread(piece_path, cv2.IMREAD_UNCHANGED)
                    if piece_img is not None:
                        # 缩放西瓜切片
                        piece_img = cv2.resize(piece_img, None, fx=scale_watermelon, fy=scale_watermelon)
                        pieces.append(piece_img)
            
            if len(pieces) == 8:
                multi_fruit_images['watermelon'] = {
                    'whole': whole_img,
                    'pieces': pieces,
                    'piece_count': 8
                }
                print(f"  ✓ 已加载: 西瓜 (8个切片, 缩放{int(scale_watermelon*100)}%)")
    
    # 加载火龙果
    dragonfruit_whole = os.path.join(assets_dir, 'all.png')
    if os.path.exists(dragonfruit_whole):
        whole_img = cv2.imread(dragonfruit_whole, cv2.IMREAD_UNCHANGED)
        if whole_img is not None:
            # 缩放完整火龙果图片
            whole_img = cv2.resize(whole_img, None, fx=scale_dragonfruit, fy=scale_dragonfruit)
            
            pieces = []
            for i in range(1, 9):
                piece_path = os.path.join(assets_dir, f'00{i}.png')
                if os.path.exists(piece_path):
                    piece_img = cv2.imread(piece_path, cv2.IMREAD_UNCHANGED)
                    if piece_img is not None:
                        # 缩放火龙果切片
                        piece_img = cv2.resize(piece_img, None, fx=scale_dragonfruit, fy=scale_dragonfruit)
                        pieces.append(piece_img)
            
            if len(pieces) == 8:
                multi_fruit_images['dragonfruit'] = {
                    'whole': whole_img,
                    'pieces': pieces,
                    'piece_count': 8
                }
                print(f"  ✓ 已加载: 火龙果 (8个切片, 缩放{int(scale_dragonfruit*100)}%)")
    
    return multi_fruit_images

# 加载多部分水果图片（全局变量）
MULTI_FRUIT_IMAGES = load_multi_fruit_images()

# 加载炸弹图片
def load_bomb_images():
    """加载炸弹和爆炸图片"""
    bomb_images = {}
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bomb_dir = os.path.join(script_dir, 'zhadan')
    
    print(f"\n正在加载炸弹素材...")
    print(f"炸弹目录: {bomb_dir}")
    
    # 加载炸弹图片（boom是炸弹，zha是爆炸效果）
    bomb_path1 = os.path.join(bomb_dir, 'boom1.png')
    bomb_path2 = os.path.join(bomb_dir, 'boom2.png')
    explosion_path1 = os.path.join(bomb_dir, 'zha01.png')
    explosion_path2 = os.path.join(bomb_dir, 'zha02.png')
    
    if os.path.exists(bomb_path1):
        bomb_img1 = cv2.imread(bomb_path1, cv2.IMREAD_UNCHANGED)
        if bomb_img1 is not None:
            scale = 1.0
            bomb_img1 = cv2.resize(bomb_img1, None, fx=scale, fy=scale)
            bomb_images['bomb1'] = bomb_img1
            print(f"  ✓ 已加载: 普通炸弹图片 (boom1)")
    
    if os.path.exists(bomb_path2):
        bomb_img2 = cv2.imread(bomb_path2, cv2.IMREAD_UNCHANGED)
        if bomb_img2 is not None:
            scale = 1.0
            bomb_img2 = cv2.resize(bomb_img2, None, fx=scale, fy=scale)
            bomb_images['bomb2'] = bomb_img2
            print(f"  ✓ 已加载: 致命炸弹图片 (boom2) - 切到即死")
    
    if os.path.exists(explosion_path1):
        explosion_img1 = cv2.imread(explosion_path1, cv2.IMREAD_UNCHANGED)
        if explosion_img1 is not None:
            scale = 2.0  # 爆炸效果放大一些
            explosion_img1 = cv2.resize(explosion_img1, None, fx=scale, fy=scale)
            bomb_images['explosion1'] = explosion_img1
            print(f"  ✓ 已加载: 爆炸效果1")
    
    if os.path.exists(explosion_path2):
        explosion_img2 = cv2.imread(explosion_path2, cv2.IMREAD_UNCHANGED)
        if explosion_img2 is not None:
            scale = 2.0
            explosion_img2 = cv2.resize(explosion_img2, None, fx=scale, fy=scale)
            bomb_images['explosion2'] = explosion_img2
            print(f"  ✓ 已加载: 爆炸效果2")
    
    return bomb_images

# 加载炸弹图片（全局变量）
BOMB_IMAGES = load_bomb_images()

# 加载刀光图片
def load_blade_images():
    """加载刀光图片"""
    blade_images = {}
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    blade_dir = os.path.join(script_dir, 'daoguang')
    
    print(f"\n正在加载刀光素材...")
    print(f"刀光目录: {blade_dir}")
    
    # 加载刀光图片1
    blade_path1 = os.path.join(blade_dir, 'dao1.png')
    if os.path.exists(blade_path1):
        blade_img1 = cv2.imread(blade_path1, cv2.IMREAD_UNCHANGED)
        if blade_img1 is not None:
            blade_images['dao1'] = blade_img1
            print(f"  ✓ 已加载: 刀光1 (dao1.png)")
    
    # 加载刀光图片2
    blade_path2 = os.path.join(blade_dir, 'dao2.png')
    if os.path.exists(blade_path2):
        blade_img2 = cv2.imread(blade_path2, cv2.IMREAD_UNCHANGED)
        if blade_img2 is not None:
            blade_images['dao2'] = blade_img2
            print(f"  ✓ 已加载: 刀光2 (dao2.png)")
    
    if not blade_images:
        print(f"  ✗ 未找到刀光图片，将使用默认线条效果")
    
    return blade_images

# 加载刀光图片（全局变量）
BLADE_IMAGES = load_blade_images()

# 加载连击特效图片
def load_combo_images():
    """加载连击特效图片"""
    combo_images = {}
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    combo_dir = os.path.join(script_dir, 'texiao')
    
    print(f"\n正在加载连击特效素材...")
    print(f"特效目录: {combo_dir}")
    
    # 加载3个combo图片
    for i in range(1, 4):
        combo_path = os.path.join(combo_dir, f'combo{i}.png')
        if os.path.exists(combo_path):
            combo_img = cv2.imread(combo_path, cv2.IMREAD_UNCHANGED)
            if combo_img is not None:
                combo_images[f'combo{i}'] = combo_img
                print(f"  ✓ 已加载: combo{i}.png")
    
    if not combo_images:
        print(f"  ✗ 未找到连击特效图片")
    
    return combo_images

# 加载连击特效图片（全局变量）
COMBO_IMAGES = load_combo_images()

# 加载音效
def load_sound_effects():
    """加载游戏音效"""
    sound_effects = {}
    
    if not HAS_SOUND:
        print("\n⚠ pygame未安装，音效功能已禁用")
        return sound_effects
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sound_dir = os.path.join(script_dir, 'yinxiao')
    
    print(f"\n正在加载音效...")
    print(f"音效目录: {sound_dir}")
    
    try:
        # 加载切水果音效
        slice_sound_path = os.path.join(sound_dir, 'qieshuiguoyinxiao.mp3')
        if os.path.exists(slice_sound_path):
            sound_effects['slice'] = pygame.mixer.Sound(slice_sound_path)
            print(f"  ✓ 已加载: 切水果音效")
        
        # 加载爆炸音效
        explosion_sound_path = os.path.join(sound_dir, 'baozhayinxiao.mp3')
        if os.path.exists(explosion_sound_path):
            sound_effects['explosion'] = pygame.mixer.Sound(explosion_sound_path)
            print(f"  ✓ 已加载: 爆炸音效")
        
        if not sound_effects:
            print(f"  ✗ 未找到音效文件")
    except Exception as e:
        print(f"  ✗ 音效加载失败: {e}")
    
    return sound_effects

# 加载音效（全局变量）
SOUND_EFFECTS = load_sound_effects()

def overlay_image(background, overlay, x, y, rotation=0, alpha=1.0):
    """将带透明通道的图片叠加到背景上
    
    Args:
        background: 背景图片
        overlay: 要叠加的图片（带alpha通道）
        x, y: 中心位置坐标
        rotation: 旋转角度（度）
        alpha: 整体透明度 (0-1)
    """
    if overlay is None or overlay.size == 0:
        return
    
    try:
        overlay_copy = overlay.copy()
        
        # 旋转图片（如果需要）
        if rotation != 0:
            h, w = overlay_copy.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
            
            # 计算旋转后的新边界框大小，避免裁剪
            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            
            # 调整旋转矩阵的平移部分，使图片居中
            matrix[0, 2] += (new_w / 2) - center[0]
            matrix[1, 2] += (new_h / 2) - center[1]
            
            # 使用新的尺寸进行旋转，保持透明背景
            overlay_copy = cv2.warpAffine(overlay_copy, matrix, (new_w, new_h), 
                                         borderMode=cv2.BORDER_CONSTANT, 
                                         borderValue=(0, 0, 0, 0))
        
        h, w = overlay_copy.shape[:2]
        
        # 计算叠加位置（中心对齐）
        x1 = int(x - w // 2)
        y1 = int(y - h // 2)
        x2 = x1 + w
        y2 = y1 + h
        
        # 边界检查
        if x1 >= background.shape[1] or y1 >= background.shape[0] or x2 <= 0 or y2 <= 0:
            return
        
        # 裁剪超出边界的部分
        overlay_x1 = max(0, -x1)
        overlay_y1 = max(0, -y1)
        overlay_x2 = w - max(0, x2 - background.shape[1])
        overlay_y2 = h - max(0, y2 - background.shape[0])
        
        bg_x1 = max(0, x1)
        bg_y1 = max(0, y1)
        bg_x2 = min(background.shape[1], x2)
        bg_y2 = min(background.shape[0], y2)
        
        if overlay_x2 <= overlay_x1 or overlay_y2 <= overlay_y1:
            return
        
        # 提取图片和alpha通道
        overlay_img = overlay_copy[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
        
        if overlay_img.size == 0:
            return
        
        if len(overlay_img.shape) == 3 and overlay_img.shape[2] == 4:  # 有alpha通道
            overlay_colors = overlay_img[:, :, :3]
            overlay_alpha = overlay_img[:, :, 3:] / 255.0 * alpha
        else:
            overlay_colors = overlay_img[:, :, :3] if len(overlay_img.shape) == 3 else overlay_img
            overlay_alpha = np.ones((overlay_img.shape[0], overlay_img.shape[1], 1)) * alpha
        
        # 获取背景区域
        bg_region = background[bg_y1:bg_y2, bg_x1:bg_x2]
        
        # 确保尺寸匹配
        if bg_region.shape[:2] != overlay_colors.shape[:2]:
            return
        
        # 混合
        background[bg_y1:bg_y2, bg_x1:bg_x2] = (
            overlay_alpha * overlay_colors + (1 - overlay_alpha) * bg_region
        ).astype(np.uint8)
    except Exception as e:
        # 静默处理错误，避免游戏崩溃
        pass

# 水果类
class Fruit:
    def __init__(self):
        self.x = random.randint(100, WINDOW_WIDTH - 100)
        self.y = WINDOW_HEIGHT + 50
        
        # 随机选择水果类型
        self.fruit_type = random.choice(FRUIT_TYPES)
        self.images = FRUIT_IMAGES.get(self.fruit_type)
        
        # 获取水果图片尺寸（用于碰撞检测）
        if self.images and self.images['whole'] is not None:
            h, w = self.images['whole'].shape[:2]
            self.width = w
            self.height = h
            self.radius = max(w, h) // 2  # 用于碰撞检测的半径
        else:
            self.radius = 50
            self.width = 100
            self.height = 100
            
        self.velocity_x = random.uniform(-1, 1)  # 降低横向速度
        self.velocity_y = random.uniform(-18, -14)  # 适中的向上速度，避免飞出屏幕
        self.gravity = 0.3  # 降低重力，让水果运动更慢
        self.rotation = random.uniform(0, 360)  # 初始旋转角度
        self.rotation_speed = random.uniform(-5, 5)  # 旋转速度
        self.is_cut = False
        self.cut_pieces = []
        self.has_entered_screen = False  # 标记水果是否曾进入过屏幕
        
    def update(self):
        if not self.is_cut:
            self.velocity_y += self.gravity
            self.x += self.velocity_x
            self.y += self.velocity_y
            self.rotation += self.rotation_speed  # 旋转
            
            # 标记水果是否进入过屏幕（在屏幕可视区域内）
            if not self.has_entered_screen and 0 <= self.y <= WINDOW_HEIGHT - 50:
                self.has_entered_screen = True
        else:
            # 更新碎片位置
            for piece in self.cut_pieces:
                piece['vy'] += self.gravity * 2.0  # 增大重力影响，让碎片快速下降
                piece['x'] += piece['vx']
                piece['y'] += piece['vy']
                piece['rotation'] += piece['rotation_speed']
                piece['alpha'] -= 3  # 稍微加快渐隐速度
                
    def draw(self, frame):
        if not self.is_cut:
            # 绘制完整水果
            if self.images and self.images['whole'] is not None:
                overlay_image(frame, self.images['whole'], 
                            int(self.x), int(self.y), self.rotation, 1.0)
            else:
                # 如果图片加载失败，绘制彩色圆圈作为备用
                cv2.circle(frame, (int(self.x), int(self.y)), 40, (0, 255, 255), -1)
        else:
            # 绘制切开的碎片
            for piece in self.cut_pieces:
                if piece['alpha'] > 0:
                    alpha = piece['alpha'] / 255.0
                    if piece['image'] is not None:
                        overlay_image(frame, piece['image'], 
                                    int(piece['x']), int(piece['y']), 
                                    piece['rotation'], alpha)
                    else:
                        # 备用显示
                        cv2.circle(frame, (int(piece['x']), int(piece['y'])), 20, (0, 200, 200), -1)
                    
    def cut(self, cut_angle):
        """切水果，生成左右两半碎片"""
        self.is_cut = True
        
        if not self.images:
            return
            
        # 创建左半边碎片
        if self.images['left'] is not None:
            left_piece = {
                'x': self.x,
                'y': self.y,
                'image': self.images['left'],
                'vx': random.uniform(-8, -5),  # 增大向左飞的速度，分开更快
                'vy': random.uniform(2, 4),  # 直接向下的初速度，快速下降
                'rotation': self.rotation,
                'rotation_speed': random.uniform(-8, -4),  # 增加旋转速度
                'alpha': 255
            }
            self.cut_pieces.append(left_piece)
        
        # 创建右半边碎片
        if self.images['right'] is not None:
            right_piece = {
                'x': self.x,
                'y': self.y,
                'image': self.images['right'],
                'vx': random.uniform(5, 8),  # 增大向右飞的速度，分开更快
                'vy': random.uniform(2, 4),  # 直接向下的初速度，快速下降
                'rotation': self.rotation,
                'rotation_speed': random.uniform(4, 8),  # 增加旋转速度
                'alpha': 255
            }
            self.cut_pieces.append(right_piece)
            
    def is_out_of_screen(self):
        if not self.is_cut:
            return self.y > WINDOW_HEIGHT + 150
        else:
            # 检查所有碎片是否都消失
            return all(piece['alpha'] <= 0 or piece['y'] > WINDOW_HEIGHT + 150 
                      for piece in self.cut_pieces)
            
    def check_collision(self, trail_points):
        """检查与轨迹的碰撞"""
        if self.is_cut or len(trail_points) < 2:
            return False
            
        # 只检查最近的几个轨迹点（提高响应速度）
        recent_points = list(trail_points)[-15:]  # 只检查最近15个点
        
        for point in recent_points:
            if point is not None:
                px, py = point
                distance = math.sqrt((px - self.x)**2 + (py - self.y)**2)
                # 使用更大的碰撞半径，更容易切中
                if distance < self.radius * 0.8:
                    return True
        return False

# 多部分水果类（切开后分成多片）
class MultiFruit:
    def __init__(self):
        self.x = random.randint(100, WINDOW_WIDTH - 100)
        self.y = WINDOW_HEIGHT + 50
        
        # 随机选择多部分水果类型
        self.fruit_type = random.choice(MULTI_FRUIT_TYPES)
        self.images = MULTI_FRUIT_IMAGES.get(self.fruit_type)
        
        # 获取水果图片尺寸
        if self.images and self.images['whole'] is not None:
            h, w = self.images['whole'].shape[:2]
            self.width = w
            self.height = h
            self.radius = max(w, h) // 2
        else:
            self.radius = 50
            self.width = 100
            self.height = 100
        
        self.velocity_x = random.uniform(-1, 1)
        self.velocity_y = random.uniform(-18, -14)
        self.gravity = 0.3
        self.rotation = random.uniform(0, 360)
        self.rotation_speed = random.uniform(-5, 5)
        self.is_cut = False
        self.cut_pieces = []
        self.has_entered_screen = False
        
    def update(self):
        if not self.is_cut:
            self.velocity_y += self.gravity
            self.x += self.velocity_x
            self.y += self.velocity_y
            self.rotation += self.rotation_speed
            
            if not self.has_entered_screen and 0 <= self.y <= WINDOW_HEIGHT - 50:
                self.has_entered_screen = True
        else:
            # 更新碎片位置
            for piece in self.cut_pieces:
                piece['vy'] += self.gravity * 2.0
                piece['x'] += piece['vx']
                piece['y'] += piece['vy']
                piece['rotation'] += piece['rotation_speed']
                piece['alpha'] -= 3
                
    def draw(self, frame):
        if not self.is_cut:
            # 绘制完整水果
            if self.images and self.images['whole'] is not None:
                overlay_image(frame, self.images['whole'], 
                            int(self.x), int(self.y), self.rotation, 1.0)
            else:
                cv2.circle(frame, (int(self.x), int(self.y)), 40, (0, 255, 255), -1)
        else:
            # 绘制切开的碎片
            for piece in self.cut_pieces:
                if piece['alpha'] > 0:
                    alpha = piece['alpha'] / 255.0
                    if piece['image'] is not None:
                        overlay_image(frame, piece['image'], 
                                    int(piece['x']), int(piece['y']), 
                                    piece['rotation'], alpha)
                    
    def cut(self, cut_angle):
        """切水果，生成多个碎片（8片）"""
        self.is_cut = True
        
        if not self.images or not self.images.get('pieces'):
            return
        
        # 创建8个碎片，呈放射状飞出
        piece_count = self.images['piece_count']
        angle_step = 360 / piece_count  # 每个碎片之间的角度
        
        for i, piece_img in enumerate(self.images['pieces']):
            # 计算飞出角度（放射状）
            angle = i * angle_step
            angle_rad = math.radians(angle)
            
            # 根据角度计算速度向量
            speed = random.uniform(6, 10)
            vx = math.cos(angle_rad) * speed
            vy = math.sin(angle_rad) * speed - 2  # 稍微向上，然后受重力影响
            
            piece = {
                'x': self.x,
                'y': self.y,
                'image': piece_img,
                'vx': vx,
                'vy': vy,
                'rotation': self.rotation + random.uniform(-30, 30),
                'rotation_speed': random.uniform(-10, 10),
                'alpha': 255
            }
            self.cut_pieces.append(piece)
            
    def is_out_of_screen(self):
        if not self.is_cut:
            return self.y > WINDOW_HEIGHT + 150
        else:
            return all(piece['alpha'] <= 0 or piece['y'] > WINDOW_HEIGHT + 150 
                      for piece in self.cut_pieces)
            
    def check_collision(self, trail_points):
        """检查与轨迹的碰撞"""
        if self.is_cut or len(trail_points) < 2:
            return False
            
        recent_points = list(trail_points)[-15:]
        
        for point in recent_points:
            if point is not None:
                px, py = point
                distance = math.sqrt((px - self.x)**2 + (py - self.y)**2)
                if distance < self.radius * 0.8:
                    return True
        return False

# 炸弹类
class Bomb:
    def __init__(self, bomb_type='normal'):
        """
        初始化炸弹
        Args:
            bomb_type: 炸弹类型 ('normal' 普通炸弹, 'deadly' 致命炸弹)
        """
        self.x = random.randint(100, WINDOW_WIDTH - 100)
        self.y = WINDOW_HEIGHT + 50
        self.bomb_type = bomb_type  # 炸弹类型
        
        # 根据类型选择不同的图片
        if bomb_type == 'deadly':
            self.image = BOMB_IMAGES.get('bomb2')  # boom2.png - 致命炸弹
        else:
            self.image = BOMB_IMAGES.get('bomb1')  # boom1.png - 普通炸弹
        
        # 获取炸弹图片尺寸（用于碰撞检测）
        if self.image is not None:
            h, w = self.image.shape[:2]
            self.width = w
            self.height = h
            self.radius = max(w, h) // 2
        else:
            self.radius = 50
            self.width = 100
            self.height = 100
            
        self.velocity_x = random.uniform(-1, 1)
        self.velocity_y = random.uniform(-18, -14)
        self.gravity = 0.3
        self.rotation = random.uniform(0, 360)
        self.rotation_speed = random.uniform(-5, 5)
        self.is_exploded = False
        self.has_entered_screen = False
        
        # 爆炸动画相关
        self.explosion_frame = 0
        self.explosion_max_frames = 20  # 爆炸动画持续帧数
        
    def update(self):
        if not self.is_exploded:
            self.velocity_y += self.gravity
            self.x += self.velocity_x
            self.y += self.velocity_y
            self.rotation += self.rotation_speed
            
            # 标记炸弹是否进入过屏幕
            if not self.has_entered_screen and 0 <= self.y <= WINDOW_HEIGHT - 50:
                self.has_entered_screen = True
        else:
            # 更新爆炸动画帧
            self.explosion_frame += 1
                
    def draw(self, frame):
        if not self.is_exploded:
            # 绘制炸弹
            if self.image is not None:
                overlay_image(frame, self.image, 
                            int(self.x), int(self.y), self.rotation, 1.0)
            else:
                # 备用显示：黑色圆圈（普通炸弹）或红色圆圈（致命炸弹）
                color = (0, 0, 255) if self.bomb_type == 'deadly' else (0, 0, 0)
                cv2.circle(frame, (int(self.x), int(self.y)), 40, color, -1)
        else:
            # 绘制爆炸效果
            if self.explosion_frame < self.explosion_max_frames:
                # 交替显示两个爆炸图片
                boom_img = BOMB_IMAGES.get('explosion1' if self.explosion_frame % 4 < 2 else 'explosion2')
                if boom_img is not None:
                    # 爆炸效果逐渐变大并渐隐
                    scale = 1.0 + (self.explosion_frame / self.explosion_max_frames) * 0.5
                    alpha = 1.0 - (self.explosion_frame / self.explosion_max_frames)
                    overlay_image(frame, boom_img, int(self.x), int(self.y), 0, alpha)
                    
    def explode(self):
        """触发爆炸"""
        self.is_exploded = True
        self.explosion_frame = 0
        
    def is_out_of_screen(self):
        if not self.is_exploded:
            return self.y > WINDOW_HEIGHT + 150
        else:
            # 爆炸动画播放完毕
            return self.explosion_frame >= self.explosion_max_frames
            
    def check_collision(self, trail_points):
        """检查与轨迹的碰撞"""
        if self.is_exploded or len(trail_points) < 2:
            return False
            
        recent_points = list(trail_points)[-15:]
        
        for point in recent_points:
            if point is not None:
                px, py = point
                distance = math.sqrt((px - self.x)**2 + (py - self.y)**2)
                if distance < self.radius * 0.8:
                    return True
        return False

# 连击特效类
class ComboEffect:
    def __init__(self, combo_count):
        """
        连击特效
        Args:
            combo_count: 连击数（用于选择显示哪个combo图片）
        """
        self.combo_count = combo_count
        self.x = WINDOW_WIDTH // 2  # 屏幕中心
        self.y = WINDOW_HEIGHT // 2 - 100  # 屏幕中上方
        self.alpha = 0  # 初始透明度
        self.scale = 0.5  # 初始缩放
        self.frame = 0
        self.duration = 60  # 持续帧数（2秒）
        
        # 根据连击数选择combo图片
        if combo_count >= 20:
            self.combo_key = 'combo3'
        elif combo_count >= 15:
            self.combo_key = 'combo2'
        elif combo_count >= 10:
            self.combo_key = 'combo1'
        else:
            self.combo_key = None
    
    def update(self):
        self.frame += 1
        
        # 淡入阶段（前15帧）
        if self.frame < 15:
            self.alpha = int(255 * (self.frame / 15))
            self.scale = 0.5 + (self.frame / 15) * 0.5  # 从0.5放大到1.0
        # 保持阶段（15-45帧）
        elif self.frame < 45:
            self.alpha = 255
            self.scale = 1.0
        # 淡出阶段（45-60帧）
        else:
            fade_progress = (self.frame - 45) / 15
            self.alpha = int(255 * (1 - fade_progress))
            self.scale = 1.0 + fade_progress * 0.2  # 稍微放大
    
    def is_finished(self):
        return self.frame >= self.duration
    
    def draw(self, frame):
        if self.combo_key is None or self.combo_key not in COMBO_IMAGES:
            return
        
        combo_img = COMBO_IMAGES[self.combo_key]
        if combo_img is None:
            return
        
        # 缩放图片
        h, w = combo_img.shape[:2]
        new_w = int(w * self.scale)
        new_h = int(h * self.scale)
        scaled_combo = cv2.resize(combo_img, (new_w, new_h))
        
        # 绘制combo特效
        alpha = self.alpha / 255.0
        overlay_image(frame, scaled_combo, int(self.x), int(self.y), 0, alpha)

# 刀光特效类
class SlashEffect:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle
        self.alpha = 255  # 初始透明度
        self.scale = 1.2  # 初始缩放
        self.duration = 15  # 持续帧数
        self.frame = 0
        
    def update(self):
        self.frame += 1
        # 逐渐淡出并放大
        self.alpha = int(255 * (1 - self.frame / self.duration))
        self.scale = 1.2 + (self.frame / self.duration) * 0.3
        
    def is_finished(self):
        return self.frame >= self.duration
    
    def draw(self, frame, blade_img):
        if blade_img is None:
            return
        
        # 缩放刀光图片
        h, w = blade_img.shape[:2]
        new_w = int(w * self.scale)
        new_h = int(h * self.scale)
        scaled_blade = cv2.resize(blade_img, (new_w, new_h))
        
        # 绘制刀光
        alpha = self.alpha / 255.0
        overlay_image(frame, scaled_blade, int(self.x), int(self.y), self.angle, alpha)

# 游戏管理类
class Game:
    def __init__(self, selected_blade='dao1'):
        self.fruits = []
        self.multi_fruits = []  # 多部分水果列表
        self.bombs = []  # 炸弹列表
        self.slash_effects = []  # 刀光特效列表
        self.combo_effects = []  # 连击特效列表
        self.selected_blade = selected_blade  # 选中的刀光
        self.score = 0
        self.missed = 0
        self.bombs_hit = 0  # 切到普通炸弹的次数
        self.combo_count = 0  # 当前连击数
        self.max_combo = 0  # 最大连击数（记录）
        self.last_combo_milestone = 0  # 上一个达到的连击里程碑
        self.spawn_timer = 0
        self.spawn_interval = 5  # 帧数
        self.game_over = False
        self.game_over_reason = ""  # 游戏结束原因
        self.max_missed = 5
        self.max_bombs_hit = 3  # 最多切到3个普通炸弹
        self.max_fruits_on_screen = 10  # 屏幕上最多同时存在水果
        self.bomb_spawn_chance = 0.2  # 20%的概率生成炸弹
        self.deadly_bomb_chance = 0.3  # 30%概率生成致命炸弹
        self.multi_fruit_chance = 0.15  # 15%概率生成多部分水果（西瓜/火龙果）
        
        # 连续生成控制（防止炸弹连续出现）
        self.last_spawn_type = None  # 上次生成的类型 ('bomb', 'fruit')
        self.consecutive_bombs = 0  # 连续生成的炸弹数量
        self.max_consecutive_bombs = 2  # 最多连续2个炸弹
        
        # 初始生成1个水果
        self.spawn_single_fruit()
        
    def spawn_single_fruit(self):
        """生成单个水果或炸弹（优化：防止炸弹连续出现）"""
        rand = random.random()
        
        # 如果已经连续生成了太多炸弹，强制生成水果
        if self.consecutive_bombs >= self.max_consecutive_bombs:
            # 强制生成水果
            self._spawn_fruit()
            self.last_spawn_type = 'fruit'
            self.consecutive_bombs = 0
            return
        
        # 基础概率判断
        should_spawn_bomb = rand < self.bomb_spawn_chance
        
        # 如果上次生成的是炸弹，降低这次生成炸弹的概率（减半）
        if self.last_spawn_type == 'bomb' and should_spawn_bomb:
            # 50%概率强制改为生成水果
            if random.random() < 0.5:
                should_spawn_bomb = False
        
        # 生成炸弹
        if should_spawn_bomb:
            if random.random() < self.deadly_bomb_chance:
                self.bombs.append(Bomb(bomb_type='deadly'))  # 致命炸弹
            else:
                self.bombs.append(Bomb(bomb_type='normal'))  # 普通炸弹
            self.last_spawn_type = 'bomb'
            self.consecutive_bombs += 1
        # 生成水果
        else:
            self._spawn_fruit()
            self.last_spawn_type = 'fruit'
            self.consecutive_bombs = 0  # 重置连续炸弹计数
    
    def _spawn_fruit(self):
        """生成水果（普通或多部分）"""
        rand = random.random()
        
        # 15%概率生成多部分水果（西瓜/火龙果）
        if rand < self.multi_fruit_chance and MULTI_FRUIT_IMAGES:
            self.multi_fruits.append(MultiFruit())
        # 其余概率生成普通水果
        else:
            self.fruits.append(Fruit())
    
    def update(self):
        if self.game_over:
            return
            
        # 计算屏幕上未被切割的水果数量和未爆炸的炸弹数量
        active_fruits = sum(1 for fruit in self.fruits if not fruit.is_cut)
        active_multi_fruits = sum(1 for fruit in self.multi_fruits if not fruit.is_cut)
        active_bombs = sum(1 for bomb in self.bombs if not bomb.is_exploded)
        active_objects = active_fruits + active_multi_fruits + active_bombs
        
        # 只有当屏幕上的水果+炸弹少于最大数量时才生成新对象
        if active_objects < self.max_fruits_on_screen:
            self.spawn_timer += 1
            if self.spawn_timer >= self.spawn_interval:
                self.spawn_single_fruit()
                self.spawn_timer = 0
                # 逐渐加快生成速度（每次减少1帧，最快30帧约1秒生成一个）
                self.spawn_interval = max(30, self.spawn_interval - 1)
        
        # 更新所有普通水果
        for fruit in self.fruits[:]:
            fruit.update()
            
            # 移除屏幕外的水果
            if fruit.is_out_of_screen():
                # 只有未被切割、进入过屏幕且掉落的水果才算漏掉
                if not fruit.is_cut and fruit.has_entered_screen:
                    self.missed += 1
                    # 漏掉水果，连击中断
                    self.combo_count = 0
                    self.last_combo_milestone = 0
                    if self.missed >= self.max_missed:
                        self.game_over = True
                        self.game_over_reason = "Too Many Missed!"
                self.fruits.remove(fruit)
        
        # 更新所有多部分水果
        for fruit in self.multi_fruits[:]:
            fruit.update()
            
            # 移除屏幕外的多部分水果
            if fruit.is_out_of_screen():
                # 只有未被切割、进入过屏幕且掉落的水果才算漏掉
                if not fruit.is_cut and fruit.has_entered_screen:
                    self.missed += 1
                    # 漏掉水果，连击中断
                    self.combo_count = 0
                    self.last_combo_milestone = 0
                    if self.missed >= self.max_missed:
                        self.game_over = True
                        self.game_over_reason = "Too Many Missed!"
                self.multi_fruits.remove(fruit)
        
        # 更新所有炸弹
        for bomb in self.bombs[:]:
            bomb.update()
            
            # 移除屏幕外的炸弹
            if bomb.is_out_of_screen():
                self.bombs.remove(bomb)
        
        # 更新刀光特效
        for effect in self.slash_effects[:]:
            effect.update()
            if effect.is_finished():
                self.slash_effects.remove(effect)
        
        # 更新连击特效
        for effect in self.combo_effects[:]:
            effect.update()
            if effect.is_finished():
                self.combo_effects.remove(effect)
                
    def check_collisions(self, trail_points):
        """检查碰撞"""
        # 计算切割角度（从轨迹最后几个点）
        cut_angle = 0
        if len(trail_points) >= 2:
            recent_points = list(trail_points)[-5:]  # 取最近5个点
            if len(recent_points) >= 2:
                x1, y1 = recent_points[0]
                x2, y2 = recent_points[-1]
                cut_angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        
        # 标记是否切中了水果
        fruit_hit = False
        
        # 检查普通水果碰撞
        for fruit in self.fruits:
            if fruit.check_collision(trail_points):
                fruit.cut(0)  # 切水果
                self.score += 10
                fruit_hit = True
                # 添加刀光特效
                self.slash_effects.append(SlashEffect(fruit.x, fruit.y, cut_angle))
                # 播放切水果音效
                if HAS_SOUND and 'slice' in SOUND_EFFECTS:
                    SOUND_EFFECTS['slice'].play()
        
        # 检查多部分水果碰撞（西瓜、火龙果）
        for fruit in self.multi_fruits:
            if fruit.check_collision(trail_points):
                fruit.cut(cut_angle)  # 切多部分水果
                self.score += 20  # 多部分水果得分更高
                fruit_hit = True
                # 添加刀光特效
                self.slash_effects.append(SlashEffect(fruit.x, fruit.y, cut_angle))
                # 播放切水果音效
                if HAS_SOUND and 'slice' in SOUND_EFFECTS:
                    SOUND_EFFECTS['slice'].play()
        
        # 如果切中了水果，增加连击数
        if fruit_hit:
            self.combo_count += 1
            # 更新最大连击记录
            if self.combo_count > self.max_combo:
                self.max_combo = self.combo_count
            
            # 检查是否达到连击里程碑（10、15、20）
            self._check_combo_milestone()
        
        # 检查炸弹碰撞
        for bomb in self.bombs:
            if bomb.check_collision(trail_points):
                bomb.explode()  # 触发爆炸
                # 添加刀光特效
                self.slash_effects.append(SlashEffect(bomb.x, bomb.y, cut_angle))
                # 播放爆炸音效
                if HAS_SOUND and 'explosion' in SOUND_EFFECTS:
                    SOUND_EFFECTS['explosion'].play()
                
                # 切到炸弹，连击中断
                self.combo_count = 0
                self.last_combo_milestone = 0
                
                # 判断炸弹类型
                if bomb.bomb_type == 'deadly':
                    # 致命炸弹 - 直接结束游戏
                    self.game_over = True
                    self.game_over_reason = "Deadly Bomb Hit! Game Over!"
                else:
                    # 普通炸弹 - 扣分并计数
                    self.score = max(0, self.score - 20)  # 扣20分，但不低于0
                    self.bombs_hit += 1  # 增加炸弹计数
                    if self.bombs_hit >= self.max_bombs_hit:
                        self.game_over = True  # 切到3个普通炸弹，游戏结束
                        self.game_over_reason = "Too Many Bombs!"
    
    def _check_combo_milestone(self):
        """检查是否达到连击里程碑并触发特效"""
        # 连击里程碑：10、15、20
        milestones = [10, 15, 20]
        
        for milestone in milestones:
            # 如果达到这个里程碑且之前没触发过
            if self.combo_count >= milestone and self.last_combo_milestone < milestone:
                self.last_combo_milestone = milestone
                # 添加连击特效
                self.combo_effects.append(ComboEffect(milestone))
                print(f"🔥 COMBO x{milestone}!")
                break
                
    def draw(self, frame):
        # 绘制普通水果
        for fruit in self.fruits:
            fruit.draw(frame)
        
        # 绘制多部分水果
        for fruit in self.multi_fruits:
            fruit.draw(frame)
        
        # 绘制炸弹
        for bomb in self.bombs:
            bomb.draw(frame)
        
        # 绘制刀光特效（使用选中的刀光）
        blade_img = BLADE_IMAGES.get(self.selected_blade)
        for effect in self.slash_effects:
            effect.draw(frame, blade_img)
        
        # 绘制连击特效
        for effect in self.combo_effects:
            effect.draw(frame)
            
        # 绘制分数和状态
        if HAS_CVZONE:
            # 使用cvzone绘制更清晰的文字（带背景框）
            cvzone.putTextRect(frame, f'Score: {self.score}', [20, 80], 
                             scale=3, thickness=3, offset=10, 
                             colorR=(0, 255, 0), colorT=(255, 255, 255))
            cvzone.putTextRect(frame, f'Missed: {self.missed}/{self.max_missed}', [20, 150], 
                             scale=2, thickness=2, offset=8, 
                             colorR=(0, 0, 255), colorT=(255, 255, 255))
            cvzone.putTextRect(frame, f'Bombs Hit: {self.bombs_hit}/{self.max_bombs_hit}', [20, 210], 
                             scale=2, thickness=2, offset=8, 
                             colorR=(255, 100, 0), colorT=(255, 255, 255))
            # 显示连击数
            if self.combo_count > 0:
                combo_color = (255, 215, 0) if self.combo_count >= 10 else (100, 100, 100)
                cvzone.putTextRect(frame, f'Combo: x{self.combo_count}', [20, 270], 
                                 scale=2.5, thickness=3, offset=10, 
                                 colorR=combo_color, colorT=(255, 255, 255))
            # 显示当前水果和炸弹数量（调试用）
            total_fruits = len(self.fruits) + len(self.multi_fruits)
            cvzone.putTextRect(frame, f'Objects: {total_fruits}F ({len(self.multi_fruits)}M) + {len(self.bombs)}B', [20, 330], 
                             scale=1.5, thickness=2, offset=6, 
                             colorR=(100, 100, 100), colorT=(255, 255, 255))
        else:
            # 使用OpenCV原生方法绘制文字
            cv2.putText(frame, f'Score: {self.score}', (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
            cv2.putText(frame, f'Missed: {self.missed}/{self.max_missed}', (20, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(frame, f'Bombs Hit: {self.bombs_hit}/{self.max_bombs_hit}', (20, 210), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
            # 显示连击数
            if self.combo_count > 0:
                combo_color = (0, 215, 255) if self.combo_count >= 10 else (200, 200, 200)
                cv2.putText(frame, f'Combo: x{self.combo_count}', (20, 270), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2.0, combo_color, 4)
            total_fruits = len(self.fruits) + len(self.multi_fruits)
            cv2.putText(frame, f'Objects: {total_fruits}F ({len(self.multi_fruits)}M) + {len(self.bombs)}B', (20, 330), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                   
        if self.game_over:
            # 游戏结束画面
            if HAS_CVZONE:
                cvzone.putTextRect(frame, 'GAME OVER!', [300, 300], 
                                 scale=7, thickness=5, offset=20,
                                 colorR=(0, 0, 255), colorT=(255, 255, 255))
                if self.game_over_reason:
                    cvzone.putTextRect(frame, self.game_over_reason, [350, 400], 
                                     scale=3, thickness=3, offset=12,
                                     colorR=(255, 100, 0), colorT=(255, 255, 255))
                cvzone.putTextRect(frame, f'Final Score: {self.score}', [300, 500], 
                                 scale=5, thickness=4, offset=15,
                                 colorR=(50, 50, 50), colorT=(255, 255, 0))
                if self.max_combo > 0:
                    cvzone.putTextRect(frame, f'Max Combo: x{self.max_combo}', [350, 580], 
                                     scale=3, thickness=3, offset=12,
                                     colorR=(255, 215, 0), colorT=(255, 255, 255))
                cvzone.putTextRect(frame, 'Press R to Restart or Q to Quit', [200, 660], 
                                 scale=2, thickness=2, offset=10,
                                 colorR=(100, 100, 100), colorT=(255, 255, 255))
            else:
                # 半透明黑色遮罩
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                
                cv2.putText(frame, 'GAME OVER!', (WINDOW_WIDTH//2 - 200, WINDOW_HEIGHT//2 - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 5)
                if self.game_over_reason:
                    cv2.putText(frame, self.game_over_reason, (WINDOW_WIDTH//2 - 150, WINDOW_HEIGHT//2 + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
                cv2.putText(frame, f'Final Score: {self.score}', (WINDOW_WIDTH//2 - 180, WINDOW_HEIGHT//2 + 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 0), 3)
                if self.max_combo > 0:
                    cv2.putText(frame, f'Max Combo: x{self.max_combo}', (WINDOW_WIDTH//2 - 150, WINDOW_HEIGHT//2 + 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 215, 255), 3)
                cv2.putText(frame, 'Press R to Restart or Q to Quit', 
                           (WINDOW_WIDTH//2 - 300, WINDOW_HEIGHT//2 + 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

def blade_selection_screen(cap):
    """刀光选择界面
    
    Args:
        cap: 摄像头对象
        
    Returns:
        选中的刀光名称 ('dao1' 或 'dao2')
    """
    # 选择区域定义
    blade1_area = {'x': 200, 'y': 300, 'w': 300, 'h': 300}
    blade2_area = {'x': 780, 'y': 300, 'w': 300, 'h': 300}
    
    # 悬停计时器
    hover_timer = {'dao1': 0, 'dao2': 0}
    hover_threshold = 90  # 3秒 * 30fps = 90帧
    current_hover = None
    
    # 坐标平滑器
    smoother = FingerSmoother(method='ewma', alpha=0.4, buffer_size=5, adaptive=True)
    
    print("\n🎮 请选择刀光样式（手指悬停3秒）...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            return 'dao1'  # 默认返回dao1
        
        # 翻转图像
        frame = cv2.flip(frame, 1)
        
        # 创建半透明遮罩
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # 绘制标题
        if HAS_CVZONE:
            cvzone.putTextRect(frame, 'Choose Your Blade Style', [280, 100], 
                             scale=3, thickness=3, offset=15,
                             colorR=(50, 50, 50), colorT=(255, 255, 0))
            cvzone.putTextRect(frame, 'Hover finger over blade for 3 seconds', [250, 180], 
                             scale=2, thickness=2, offset=10,
                             colorR=(100, 100, 100), colorT=(255, 255, 255))
        else:
            cv2.putText(frame, 'Choose Your Blade Style', (280, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 0), 4)
            cv2.putText(frame, 'Hover finger over blade for 3 seconds', (250, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # 绘制刀光选择框
        for i, (blade_name, area) in enumerate([('dao1', blade1_area), ('dao2', blade2_area)]):
            x, y, w, h = area['x'], area['y'], area['w'], area['h']
            
            # 判断是否悬停
            is_hovering = (current_hover == blade_name)
            progress = hover_timer[blade_name] / hover_threshold if is_hovering else 0
            
            # 边框颜色：悬停时为绿色，否则为白色
            border_color = (0, 255, 0) if is_hovering else (255, 255, 255)
            thickness = 5 if is_hovering else 3
            
            # 绘制选择框
            cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, thickness)
            
            # 绘制刀光预览图
            blade_img = BLADE_IMAGES.get(blade_name)
            if blade_img is not None:
                # 缩小刀光图片以适应选择框
                scale = 0.8
                h_img, w_img = blade_img.shape[:2]
                new_w = int(w_img * scale)
                new_h = int(h_img * scale)
                scaled_blade = cv2.resize(blade_img, (new_w, new_h))
                
                # 在选择框中心绘制刀光
                center_x = x + w // 2
                center_y = y + h // 2
                overlay_image(frame, scaled_blade, center_x, center_y, 0, 1.0)
            
            # 绘制标签
            label = f"Blade {i+1}"
            if HAS_CVZONE:
                cvzone.putTextRect(frame, label, [x + 80, y - 30], 
                                 scale=2, thickness=2, offset=8,
                                 colorR=(50, 50, 50), colorT=(255, 255, 255))
            else:
                cv2.putText(frame, label, (x + 80, y - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            # 绘制进度条
            if is_hovering and progress > 0:
                bar_x = x + 10
                bar_y = y + h + 20
                bar_w = w - 20
                bar_h = 30
                
                # 背景
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
                # 进度
                progress_w = int(bar_w * progress)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_w, bar_y + bar_h), (0, 255, 0), -1)
                # 边框
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
        
        # 手部检测
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        current_hover = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 获取食指指尖位置
                index_finger_tip = hand_landmarks.landmark[8]
                raw_x = int(index_finger_tip.x * WINDOW_WIDTH)
                raw_y = int(index_finger_tip.y * WINDOW_HEIGHT)
                
                # 平滑坐标
                smooth_x, smooth_y = smoother.smooth(raw_x, raw_y)
                
                # 绘制手指位置
                cv2.circle(frame, (smooth_x, smooth_y), 25, (0, 255, 255), 3)
                cv2.circle(frame, (smooth_x, smooth_y), 15, (0, 255, 0), cv2.FILLED)
                
                # 检查悬停在哪个区域
                for blade_name, area in [('dao1', blade1_area), ('dao2', blade2_area)]:
                    x, y, w, h = area['x'], area['y'], area['w'], area['h']
                    if x <= smooth_x <= x + w and y <= smooth_y <= y + h:
                        current_hover = blade_name
                        hover_timer[blade_name] += 1
                        
                        # 达到阈值，选择该刀光
                        if hover_timer[blade_name] >= hover_threshold:
                            print(f"✓ 已选择: {blade_name}")
                            return blade_name
                    else:
                        hover_timer[blade_name] = max(0, hover_timer[blade_name] - 2)  # 快速衰减
        else:
            smoother.reset()
            # 没检测到手指，衰减所有计时器
            for blade_name in hover_timer:
                hover_timer[blade_name] = max(0, hover_timer[blade_name] - 2)
        
        # 显示窗口
        cv2.imshow('Fruit Ninja - 体感切水果', frame)
        
        # 按键控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return 'dao1'  # 默认返回dao1
        elif key == ord('1'):
            return 'dao1'
        elif key == ord('2'):
            return 'dao2'

def main():
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
    
    # 尝试提高摄像头质量
    cap.set(cv2.CAP_PROP_FPS, 30)  # 设置帧率
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # 自动对焦
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 使用自动曝光
    # 如果画面太暗或太亮，可以手动调整曝光值（范围通常是-13到-1）
    # cap.set(cv2.CAP_PROP_EXPOSURE, -3)  # 可选：手动曝光调整
    
    # 刀光选择界面
    selected_blade = blade_selection_screen(cap)
    print(f"\n✓ 使用刀光: {selected_blade}")
    
    # 手指轨迹队列（用于绘制刀光）- 30个点，适中的轨迹长度
    trail_points = deque(maxlen=30)
    
    # 创建坐标平滑器 - 你可以切换不同的方法
    # 方法选项: 'ewma' (推荐), 'moving_avg', 'kalman'
    # adaptive=True 启用自适应平滑（快速移动时减少延迟）
    smoother = FingerSmoother(method='ewma', alpha=0.4, buffer_size=5, adaptive=True)
    
    # 创建游戏实例（使用选中的刀光）
    game = Game(selected_blade)
    
    # 调试模式开关（按 D 键切换）
    debug_mode = False
    
    print("=" * 50)
    print("🍉 体感切水果游戏启动成功！")
    print("=" * 50)
    print(f"✅ 已加载 {len(FRUIT_IMAGES)} 种普通水果素材")
    print(f"   水果类型: {', '.join(FRUIT_IMAGES.keys())}")
    print(f"✅ 已加载 {len(MULTI_FRUIT_IMAGES)} 种多部分水果素材（分值更高）")
    print(f"   特殊水果: {', '.join(MULTI_FRUIT_IMAGES.keys())}")
    print(f"✅ 已加载 {len(COMBO_IMAGES)} 种连击特效")
    print(f"   连击系统: 10连击→Combo1, 15连击→Combo2, 20连击→Combo3")
    if HAS_SOUND and SOUND_EFFECTS:
        print(f"🔊 已加载 {len(SOUND_EFFECTS)} 种音效")
        print(f"   音效类型: {', '.join(SOUND_EFFECTS.keys())}")
    else:
        print("🔇 音效已禁用 (安装pygame启用音效)")
    print(f"\n🎯 平滑算法: {smoother.method.upper()} (自适应模式 - 激进版)")
    print("   慢速移动(<3px/帧)：强平滑，减少抖动")
    print("   快速移动(>20px/帧)：完全不平滑，原始坐标")
    print("   刀光轨迹长度: 30点")
    print("   MediaPipe: Lite模式（高性能）")
    print("\n🎮 控制说明：")
    print("   - 挥动食指来切水果")
    print("   - 按 R 键重新开始")
    print("   - 按 Q 键退出游戏")
    print("   - 按 D 键开启/关闭调试模式")
    print("   - 按 1/2/3 键切换平滑算法")
    print("\n🏆 游戏规则：")
    print("   - 普通水果: 10分")
    print("   - 特殊水果(西瓜/火龙果): 20分")
    print("   - 连续切中水果触发连击，漏掉或切到炸弹会中断连击")
    print("   - 切到普通炸弹: -20分，最多3次")
    print("   - 切到致命炸弹(红色): 游戏直接结束")
    print("=" * 50)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 翻转图像（镜像效果）
        frame = cv2.flip(frame, 1)
        
        # 转换颜色空间用于手部检测
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # 处理手部检测结果
        current_finger_pos = None
        current_speed = 0
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 获取食指指尖位置 (landmark 8)
                index_finger_tip = hand_landmarks.landmark[8]
                raw_x = int(index_finger_tip.x * WINDOW_WIDTH)
                raw_y = int(index_finger_tip.y * WINDOW_HEIGHT)
                
                # ⭐ 应用平滑算法 - 关键优化点
                smooth_x, smooth_y = smoother.smooth(raw_x, raw_y)
                current_finger_pos = (smooth_x, smooth_y)
                
                # 计算当前速度（用于调试显示）
                if smoother.prev_raw_pos:
                    prev_x, prev_y = smoother.prev_raw_pos
                    current_speed = math.sqrt((raw_x - prev_x)**2 + (raw_y - prev_y)**2)
                
                # 只有检测到手指时才添加到轨迹（使用平滑后的坐标）
                trail_points.append((smooth_x, smooth_y))
        else:
            # 没检测到手指时重置平滑器
            smoother.reset()
        
        # 如果没有检测到手指，清空部分轨迹（保持最近的10个点）
        if current_finger_pos is None and len(trail_points) > 10:
            # 逐渐清空轨迹
            for _ in range(min(5, len(trail_points))):
                if len(trail_points) > 0:
                    trail_points.popleft()
        
        # 绘制手指轨迹（轻微的线条，用于指示手指移动路径）
        for i in range(1, len(trail_points)):
            if trail_points[i] is not None and trail_points[i-1] is not None:
                # 半透明的细线条
                alpha = (i / len(trail_points)) * 0.3  # 很低的透明度
                thickness = int(2 + alpha * 6)  # 很细的线条
                color = (100, 100, 255)  # 淡蓝色
                cv2.line(frame, trail_points[i-1], trail_points[i], color, thickness)
        
        # 在最上层绘制手指位置标记
        if current_finger_pos is not None:
            x, y = current_finger_pos
            # 根据速度改变手指标记颜色
            if current_speed > 20:
                # 快速移动：红色
                color = (0, 0, 255)
            elif current_speed > 10:
                # 中速：橙色
                color = (0, 165, 255)
            else:
                # 慢速：绿色
                color = (0, 255, 0)
            
            cv2.circle(frame, (x, y), 25, color, 3)  # 彩色外圈
            cv2.circle(frame, (x, y), 15, (0, 255, 255), cv2.FILLED)  # 黄色实心内圈
            
            # 调试模式：显示速度和平滑参数
            if debug_mode:
                # 显示速度
                cv2.putText(frame, f'Speed: {current_speed:.1f} px/f', (x + 35, y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                # 显示实际使用的alpha值
                if smoother.adaptive and current_speed > 0:
                    alpha = smoother._get_adaptive_alpha(current_speed)
                    cv2.putText(frame, f'Alpha: {alpha:.2f}', (x + 35, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
        # 更新游戏逻辑
        if not game.game_over:
            game.update()
            game.check_collisions(trail_points)
            
        # 绘制游戏元素（直接在摄像头画面上）
        game.draw(frame)
        
        # 调试模式：显示FPS和状态
        if debug_mode:
            cv2.putText(frame, '[DEBUG MODE]', (WINDOW_WIDTH - 250, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f'Trail Points: {len(trail_points)}', (WINDOW_WIDTH - 250, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 显示游戏窗口
        cv2.imshow('Fruit Ninja - 体感切水果', frame)
        
        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # 重新开始游戏（保持相同的刀光选择）
            game = Game(selected_blade)
            trail_points.clear()
            smoother.reset()
        elif key == ord('d'):
            # 切换调试模式
            debug_mode = not debug_mode
            print(f"✓ 调试模式: {'开启' if debug_mode else '关闭'}")
        elif key == ord('1'):
            # 切换到EWMA（推荐）
            smoother = FingerSmoother(method='ewma', alpha=0.4, adaptive=True)
            print("✓ 已切换到 EWMA 自适应平滑算法 (推荐)")
        elif key == ord('2'):
            # 切换到移动平均
            smoother = FingerSmoother(method='moving_avg', buffer_size=5, adaptive=True)
            print("✓ 已切换到 移动平均 自适应平滑算法")
        elif key == ord('3'):
            # 切换到卡尔曼滤波
            smoother = FingerSmoother(method='kalman', adaptive=False)
            print("✓ 已切换到 卡尔曼滤波 算法")
        elif key == ord('0'):
            # 完全关闭平滑（测试用）
            smoother = FingerSmoother(method='ewma', alpha=1.0, adaptive=False)
            print("✓ 已关闭平滑（原始坐标）")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()

