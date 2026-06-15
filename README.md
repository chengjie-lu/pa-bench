# PA-Bench — VLA 精密装配评测 Benchmark（M1 纵切实现）

对应需求文档 `../rq.md` 的 M1 最小闭环（§11）：**瓶盖拧紧 (screw_cap, T1)** 单任务的
端到端评测链路 —— 场景生成（标称 + 变异 + 变质 MR-1）→ 仿真执行（2 模型 × 2 硬件档位）
→ L0/L2/L3 三层指标 → e_plan/e_track 故障归因（含 oracle 回放对照实验）→ HTML/JSON 报告。
固定 seed 全链路可复现（NFR-1，demo 内置哈希自检）。

## 从零跑起来

```bash
# 环境: Python 3.11 (开发验证于 /Users/chengjielu/opt/anaconda3/envs/sandbox)
pip install -r requirements.txt

# 端到端 demo (≈1s): 控制台摘要 + out/report.html + out/report.json + out/episodes.jsonl
python demo.py --episodes 24 --seed 7 --out out

# 测试套件 (57 个测试: 41 核心 + 16 platform API)
python -m pytest tests/ -q

# Web 控制台 A — M-FE1 静态模式 (无后端, 见 ../fe-rq.md):
python build_web_data.py                    # out/ 产物 → web/data/ (索引+预聚合+单回合拆分)
python -m http.server 8765 --directory web  # 访问 http://localhost:8765/

# Web 控制台 B — M-FE2 服务模式 (FastAPI + SSE, 浏览器发起评测):
python serve.py                             # 访问 http://127.0.0.1:8000/
#   ① 评测运行 → 新建运行: 3 步向导 (选模型×硬件 / 策略+预算 / seed+启动)
#   ② 启动后看 SSE 实时进度 → 完成自动转结果页 → 下钻单回合调试
#   已有 out/ 产物会被自动导入为历史运行 run-000-legacy-out
```

> 前端同一份 `web/app.js` 双模式运行: 启动时探测 `/api/ping` —— 命中即服务模式
> (platform API + 发起评测 + SSE 进度), 否则回退 M-FE1 静态模式 (读 `web/data/`)。

Web 控制台视图：①总览(红绿灯卡片+Δ箭头+Top-3 短板下钻) ②运行结果(组合对比表 + 雷达/
鲁棒性曲线/归因桑基/首败直方图, 图表点击下钻) ③回合浏览器(多维筛选, 状态写入 URL)
④单回合调试页(指令 vs 实测轨迹回放 + 公差放大镜 + 四面板同步时序 + 播放头联动,
空格播放/暂停, ←/→ 切换回合)。服务模式额外: ②评测运行(列表 + 3 步新建向导 + SSE 进度页)。
ECharts 已本地化 (web/vendor/), 离线可用。

## 文件树

```
pa-bench/
├── demo.py                      # 端到端 CLI (纵切入口, 薄包装 → pabench.pipeline)
├── serve.py                     # M-FE2 服务模式入口 (FastAPI + 静态前端同进程)
├── build_web_data.py            # M-FE1 静态适配 (薄包装 → pabench.webdata)
├── requirements.txt
├── pabench/
│   ├── schema.py                # ★ Episode 核心契约 + 谱系机检 EpisodeStore (rq.md §5, FR-1.6)
│   ├── pipeline.py              # ★ 共享评测编排 RunConfig/run_benchmark (demo 与 API 共用, 进度回调+取消)
│   ├── webdata.py               # 前端数据适配: 回合索引 + C1/C2/C3/C5 预聚合 (NFR-FE N2)
│   ├── platform/                # M-FE2 服务层 (注: 子包避开 stdlib platform 同名遮蔽)
│   │   ├── run_manager.py       # 运行发起/进度事件/取消/落盘/历史导入 (后台线程 + SSE 数据源)
│   │   └── api.py               # FastAPI: §8 全部接口 + SSE + stride 降采样 + 静态挂载
│   ├── scenegen/
│   │   ├── nominal.py           # FR-1.1 标称任务 (screw_cap T1) + 阶段计划
│   │   ├── mutation.py          # FR-1.2 变异生成 (位姿/光照/摩擦, 参数全记录)
│   │   └── metamorphic.py       # FR-1.3 MR-1 旋转等变 + 协议级中位数判定
│   ├── models/
│   │   ├── base.py              # FR-2.4 VLAModel 标准接口 (uncertainty 可选)
│   │   └── fake.py              # 【FAKE】2 个脚本化假模型 (precise / sloppy)
│   ├── runners/
│   │   ├── base.py              # Backend 抽象 + HardwareProfile (硬件标定档案)
│   │   ├── fake_sim.py          # 【FAKE 后端】解析运动学假仿真 + oracle 回放 (FR-2.5)
│   │   └── mujoco_sim.py        # 【桩】MuJoCo 插件位, import 失败优雅降级
│   ├── metrics/
│   │   ├── l0_outcome.py        # SR + Wilson CI, 效率分, 首败分布 (FR-3.1/3.2/3.4)
│   │   ├── l2_process.py        # 对准残差/jerk/力超限/不确定性AUROC/时延 (FR-3.5–3.10)
│   │   ├── l3_hardware.py       # e_track RMS(稳态), 5–50Hz 抖动 PSD (FR-3.11/3.12)
│   │   └── registry.py          # FR-5.1 指标注册表 + R-8 机检 (无改进动作不许上线)
│   ├── attribution/engine.py    # FR-4 归因决策树 + oracle 对照实验编排
│   └── report/html_report.py    # FR-6 纵切版: 红绿灯摘要 + 工程对比表 (静态 HTML)
├── web/                         # 前端控制台 (原生 ES 模块 + ECharts, M-FE1 静态 / M-FE2 服务双模式)
└── tests/                       # 57 个行为断言测试 (含 test_api.py platform API)
```

## 真实现 vs 假实现（桩）

| 部分 | 状态 | 说明 |
|---|---|---|
| Episode schema / 谱系机检 / 哈希复现 | ✅ 真实现 | 全系统契约 |
| 场景生成（变异 / MR-1 / 协议判定） | ✅ 真实现 | MR-2/3/4 未做（见下） |
| 指标计算（L0/L2/L3 共 10 项）+ 注册表机检 | ✅ 真实现 | Wilson/AUROC/PSD/jerk 均 numpy 手写并有公式级单测 |
| 归因决策树 + oracle 对照编排 | ✅ 真实现 | 阈值版本化 `attr-rules-0.1` |
| HTML/JSON 报告 | ✅ 真实现 | 三层信息架构的前两层 |
| Web 控制台 (M-FE1 静态 + M-FE2 服务) | ✅ 真实现 | FastAPI + SSE 进度 + 浏览器发起评测; 同一前端双模式 |
| VLA 模型 | 🔶 **FAKE** `pabench/models/fake.py` | 脚本化、行为可控的假模型 |
| 仿真后端 | 🔶 **FAKE** `pabench/runners/fake_sim.py` | 解析运动学近似物理 |
| MuJoCo 后端 | ⬜ 桩 `pabench/runners/mujoco_sim.py` | import 优雅降级, 明确报错指引 |

**接真实后端要改的文件：**
1. 真实 VLA → 新增 `pabench/models/<your_model>.py` 实现 `VLAModel.infer()`（内部调 gRPC/HTTP），其余零改动；
2. 真实仿真 → 实现 `pabench/runners/mujoco_sim.MujocoBackend` 的 `run_episode/run_oracle`（场景 XML 生成 + mj_step 循环 + 100Hz 遥测采样）；
3. 真机 → 新增 `pabench/runners/real_robot.py` 实现同一 `Backend` 接口（外加 FR-2.2 初始化检查与 NFR-4 安全回路，本纵切未覆盖）。

## 假设与待确认

- 变异场景在所有(模型×硬件)组合间共享以保证公平对比（NFR-2）；其 `parent_episode_id`
  指向标称场景的锚点记录而非某一具体回合（标称回合 id 因组合而异）。
- 假物理的误差链刻意构造为「感知误差(→e_plan) + 跟踪误差(→e_track)」可分解，
  用于验证归因机制本身；真实仿真/真机下阈值需按 rq.md O-6 用首批数据重标定。
- 归因用的 e_track 取 fasten 稳态窗口 RMS（运动段跟踪迟滞不污染硬件判别）。
- MR 判定用时间同步逐点距离而非 DTW（DTW 时间规整会吸收沿路径方向的偏置，
  弱化非等变检出；DTW 保留给将来不等长轨迹）。
- M-FE2 服务层落在 `pabench/platform/` 而非 fe-rq.md O-F4 建议的仓库顶层 `platform/`：
  顶层同名目录会遮蔽 Python 标准库 `platform` 模块（numpy 等启动即 import）。
- 评测执行用后台线程而非进程/任务队列（FakeSim 是 CPU 顺序仿真, 单 worker 足够 M1 纵切）；
  接真实/重型后端时换 §8 所述任务队列。服务重启时 running 中的孤儿运行标记为 failed。

## 已知限制与下一步（按 rq.md 里程碑推进）

1. **M1 余项**：MR-2/3/4 变质关系；闭环分块推理（当前模型一次输出整条轨迹）。
2. **M2**：真机 Runner + 安全回路；FR-3.13 重复定位精度 / FR-3.14 摩擦辨识
   （需真机标定流程）；归因专家盲标校准（FR-4.4）。
3. **M3**：FR-1.4 对抗/优化搜索（CMA-ES 失败边界探索）；FR-1.5 真机分层抽样；
   sim-real 排序一致性测量（G4）。
4. **M4**：任务矩阵扩展（5 任务型 × 3 公差级）；FR-5.2 定向采集清单。

### 前端里程碑 (fe-rq.md §13)

- **M-FE1 静态模式** ✅：总览/运行结果/回合浏览器/单回合调试 + C1/C2/C3/C5 图表，读 `web/data/`。
- **M-FE2 服务化** ✅：FastAPI 包装 + `/runs/new` 3 步向导（不含场景编辑器）+ SSE 实时进度 +
  运行列表/详情/取消/重建 + legacy `out/` 导入。同 seed 经 API 发起两次产物一致（FR-FE-2.1）。
- **M-FE3 交互场景** ⬜：场景编辑器键鼠交互（拖拽位姿/调光照）、oracle 对照按钮、双运行并排对比。
- **M-FE4 客户化** ⬜：报告导出 PDF、`/hardware` 硬件趋势页、R-9 非专业用户测试。
- 仍按降级条款执行：3D 视口为 2D 俯视 canvas（N5）；FakeSim 无渲染故调试页视频区恒为占位（O-F2）。