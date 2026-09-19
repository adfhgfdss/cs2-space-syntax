# data\ 目录：原始游戏资产不入库，请自行生成

仓库里**不含** CS2 的地图资产（版权归 Valve）。跑分析前按下面两步在本机生成。

## 1. 导航网格（可通行空间）

需要本机安装了 CS2（Steam），脚本会从地图包里抽 `maps/<map>.nav`：

```bash
python nav_extract.py de_dust2
python nav_extract.py de_mirage
```

生成 `data/game_nav/de_dust2.nav` 与 `data/game_nav/de_mirage.nav`。
本项目分析时用的文件是：

| 地图 | 大小 | SHA256（前 16 位） |
|---|---|---|
| de_dust2 | 481,872 B | `64bbc202edae8a92` |
| de_mirage | 469,448 B | `f9b23bb5d2303549` |

## 2. 雷达图与 overview 坐标文件

```
data/de_dust2.png      # 雷达底图（1024×1024 RGBA）
data/de_dust2.txt      # overview：pos_x / pos_y / scale / rotate
data/de_mirage.png
data/de_mirage.txt
```

来源：社区镜像仓库 `2mlml/cs2-radar-images`。

已验证：镜像的 overview 文件与本机 CS2 的 `resource/overviews/*.txt`
**逐字节相同**（SHA256 一致），坐标变换参数是权威的；雷达图通过
"导航网格与图像内容区重合率"间接验证（Dust2 100.0%、Mirage 99.5%）。

## 3. 校验

```bash
python check_map_files.py      # 检查 png 与 txt 是否配对
python verify_alignment.py     # 用"回合开局玩家必在出生点"验证坐标映射
```
