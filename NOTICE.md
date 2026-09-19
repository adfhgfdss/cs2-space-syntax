# 第三方资源与数据说明

## 1. 本仓库包含什么

* **代码**：全部 `.py` 脚本（MIT 许可，见 `LICENSE`）。
* **文档**：`docs/` 下的报告与方法说明。
* **结果**：`out/` 下的图（PNG）、数据表（CSV）、指标文件（JSON），
  以及 depthmapX 的中间图文件（`.graph`）。
* 部分图使用 CS2 地图雷达图作底图，其版权归 Valve Corporation，
  仅用于学术研究与结果展示，如权利人要求可立即移除。

## 2. 本仓库不包含什么（需自行获取）

| 内容 | 原因 | 获取方式 |
|---|---|---|
| 比赛 demo（约 6.7 GB） | HLTV 录像不随仓库分发 | 见 `docs/数据获取.md` |
| 雷达图 / overview / 导航网格 | 版权归 Valve，且可从本机游戏重新生成 | `nav_extract.py`；雷达图见 `data/README.md` |
| depthmapX 命令行版 | 第三方二进制，按其自身许可分发 | 见 `docs/仓库说明.md` |
| 浏览器登录凭据（抓取 HLTV 用） | 个人敏感信息 | 由 `hltv_session.py` 在本机重新生成 |

## 3. 数据来源

| 数据 | 来源 |
|---|---|
| 行为数据 | HLTV（`hltv.org`）公开的职业比赛录像，2026-09-19 抓取 |
| 雷达底图 | 社区镜像仓库 `2mlml/cs2-radar-images` |
| 坐标文件（overview） | 本机 CS2 安装的 `pak01_dir.vpk`，与镜像版 SHA256 一致 |
| 导航网格 | 本机 CS2 安装的 `maps/<map>.vpk` 内的 `maps/<map>.nav` |
| depthmapX | SpaceGroupUCL/depthmapX v0.9.1（命令行版） |

## 4. 抓取工具的使用边界

`hltv_session.py` 与 `fetch_pro_demos.py` 会访问 HLTV 站点。使用前请遵守其
服务条款与 robots 约定；脚本已内置请求间隔（默认 1.5 秒）。
