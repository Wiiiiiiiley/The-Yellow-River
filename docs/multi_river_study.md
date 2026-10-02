# 多河流预测实验说明

本项目在原黄河研究代码旁增加六河流实验：黄河，以及新增的密西西比河、长江、亚马逊河、尼罗河和多瑙河。它提供可重复训练的模型、留出时段预测、基线比较和个人网站展示材料。当前多河流实验使用合成水文序列；图中的“参考值”来自生成器，不能解释为测站观测或真实洪水预报。

## 展示内容与研究问题

每条河有五个示意河段 `r1` 至 `r5`，依次表示上游至下游的位置。它们是抽象节点，不对应同名真实测站，图结构也不是六个流域的真实地图。模型读取过去24小时的多节点记录，预测未来6小时的水深、流量和悬沙浓度。问题有三个：短时序列是否比“维持最新值”基线更准确；增加简化水力关系是否改善同一架构的结果；黄河与密西西比河的预训练知识是否能帮助另一条河。

下面的机制背景用于安排示例场景；具体河宽、坡度、糙率、雨量、流量规模、泥沙系数和滞后均为未经观测校准的设计值。

| 河流 | 可学习的水文背景 | 当前示例着重演示 |
| --- | --- | --- |
| 黄河 `yellow_river` | 黄土区侵蚀与水土保持影响入河泥沙；流域还面临水资源短缺。[世界银行项目报告](https://documents1.worldbank.org/curated/en/820471468769214283/pdf/257011CN1Loess1d0Rehab0Project01ICR.pdf)、[流域水资源背景](https://www.worldbank.org/en/news/press-release/2022/03/31/new-program-will-reduce-water-scarcity-protect-ecosystems-in-china-s-yellow-river-basin) | 较强的示意含沙变化与水沙关联 |
| 密西西比河 `mississippi` | 支流、水库、堤防和河道工程影响流态、输沙与洪泛平原联系。[USGS工程综述](https://pubs.usgs.gov/publication/cir1375) | 较宽示意断面与较长传播滞后；没有实际工程运行模型 |
| 长江 `yangtze` | 干流、洞庭湖、鄱阳湖与水库群调度共同影响洪水过程，2024年的防汛部署提供实际案例。[水利部报道](https://www.gov.cn/lianbo/bumen/202407/content_6961645.htm) | 夏季型流量峰值；没有真实江湖交换与水库群调度 |
| 亚马逊河 `amazon` | 季节性降雨与洪泛森林蓄水伴随河水涨落。[NASA流域介绍](https://science.nasa.gov/earth/earth-observatory/introduction-to-the-large-scale-biosphere-atmosphere-experiment-in-amazonia/) | 较大示意水量与较长传播滞后；没有实际洪泛森林蓄水模型 |
| 尼罗河 `nile` | 不同支流的季节性差别明显，湖泊和湿地影响水量调节；赤道湖区与东尼罗河区雨季也不同。[NBI流域报告](https://nilebasin.org/sites/default/files/2023-09/State%2520of%2520Basin%2520Report%25202021_0.pdf)、[季节水文说明](https://nilebasin.org/outlook) | 较小季节振幅；没有真实水库放流规则 |
| 多瑙河 `danube` | 上、中、下游地形不同，拦蓄、取水与水电调峰改变水文过程。[ICPDR流域说明](https://www.icpdr.org/danube-basin/danube-river-basin)、[水文地貌说明](https://www.icpdr.org/tasks-topics/topics/hydromorphology) | 春季型峰值与中等传播滞后；没有真实融雪和支流水量模型 |

这些来源支持上述机制描述，不给合成参数提供测站级依据。季节曲线也不是对某一真实河段2024年水情的重建。

## 模型与物理关系

多河流模块采用轻量的有向图时序网络：沿上游到下游的连接聚合节点信息，以GRU学习时间变化，再直接输出6个未来时刻。每条河分别训练 `physics_guided` 与 `data_only` 两个版本；架构、输入输出和参数数量一致，区别是是否加入Manning关系的软惩罚。`persistence` 将最后一次记录保持到每个预测步，用作无需训练的基线。六个输入特征是水深、流量、泥沙浓度、温度、每小时降水，以及由时间戳派生的 `season_sin`；输出只预测前三个水文变量。

在SI单位下，矩形示意断面满足

\[
Q_M(h)=\frac{1}{n} A(h)R(h)^{2/3}\sqrt{S},\qquad
A(h)=b h,\quad R(h)=\frac{b h}{b+2h}.
\]

`h` 是相对于河床的水深（m），`Q` 是流量（m³/s），`b` 是示意河宽（m），`n` 是糙率，`S` 是能坡。损失在恢复物理单位后比较预测流量与 `Q_M(h)`，再以情景配置中的 `base_discharge` 缩放；还包含按情景水深、流量和泥沙尺度缩放的负值惩罚。默认物理损失权重为0.05。公式与均匀流适用背景见 [HEC-RAS官方说明](https://www.hec.usace.army.mil/confluence/rasdocs/ras1dtechref/latest/stable-channel-design-functions/uniform-flow-computations)。两个网络在预测导出和评分前都将负输出截到零，以同一规则计算结果。

这是物理引导的预测模型。它没有在新增模块中求解完整圣维南偏微分方程、Exner方程或洪泛区二维水动力学；也未把全部支流、潮汐、回水、漫滩和水库操作纳入守恒方程。Manning软惩罚不等于质量或动量守恒证明。泥沙是预测目标，当前不施加泥沙守恒约束。

合成生成器的水深与流量也受简化关系影响，因此物理版本若表现良好，部分原因可能是训练先验与生成规则一致。真实河道存在非恒定流、断面变化与回水，单一水深—流量关系可能失效。不能从这项示例推导真实洪水中的物理模型优势。[USGS测流说明](https://www.usgs.gov/water-science-school/science/how-streamflow-measured)也说明率定关系依赖具体站点和时期，须持续复核。

## 可重复运行

在项目根目录执行：

```bash
python3 scripts/run_multi_river.py --epochs 25 --hours 8760 --seed 42 --transfer-seeds 42,43,44
```

默认生成自 `2024-01-01 00:00 UTC` 起连续8760小时的序列；这是8760条小时样本，不是对2024闰年完整366天的观测。固定种子控制数据和训练随机性。运行会训练六条河各自的两个模型，评估持久性基线，再执行黄河与密西西比河的双向迁移实验。具体运行参数、训练历史、模型文件、预测CSV和指标写入 `experiments/multi_river/`；运行结束再阅读 [生成的结果报告](multi_river_results.md)，以实际输出为准。

主要文件包括 `manifest.json`（生成文件的大小和SHA-256摘要）、`metrics/{河流ID}.json`（留出指标与协议）、`models/{河流ID}_{方法}.pt`（检查点），以及 `predictions/{河流ID}_{方法}_{forecast|backtest}.csv`（外推/回测预测）。默认合成输入写入 `data/{河流ID}.csv`；显式观测目录模式读取用户CSV，不自动复制原始文件。`transfer/results.json` 保存双向迁移实验。`website/assets/river_lab_data.js` 是网页读取的结果快照，不应手工修改数值。

载入已训练模型并重新预测的例子：

```bash
python3 scripts/predict_river.py --river mississippi --input experiments/multi_river/data/mississippi.csv --source-status synthetic --output experiments/multi_river/predictions/mississippi_reloaded.csv
```

外部输入的来源状态要与实际资料相符；不能因为CSV通过校验就把合成或混合资料标成真实观测。

结果应同时发布代码、配置和协议。网站呈现结果，重新训练仍需Python和PyTorch；静态网页不在访客浏览器中重新训练模型。

## 切分与比较规则

按原始时间轴划分70%训练、15%验证、15%测试，以24小时步长抽取预测窗口。每个窗口读取24小时历史，预测随后6小时，并不是为每个小时都发起一次回测。每个预测窗口的全部目标时刻必须属于同一分区；跨越分区边界的目标窗口不参加训练或评估。验证或测试窗口可以读取其预测起点之前已经发生的历史，这是短时预报所需的可用信息，不应读取未来目标。

归一化统计量只从训练段计算，验证与测试复用它。模型选择只看验证集。测试集只用于最终评估，不用于挑选轮数、物理权重、随机种子或最佳示例。所有版本使用同一窗口、节点与预测时长；输出的节点是真正独立的目标，不用站点均值复制出五个结果。

分别报告水深、流量、泥沙的MAE、RMSE、偏差、NSE和NRMSE，当前指标合并所有测试窗口、五节点与六预测步。NSE为 `1 - Σ(预测-参考)² / Σ(参考-参考均值)²`，负值表示劣于测试参考均值；NRMSE为RMSE除以测试参考值的最大值与最小值之差，常数目标的NSE/NRMSE返回空值。NRMSE使用测试范围做评估标度，不用于模型训练或选择。不同河流的物理尺度差异很大，直接以总流量MAE排序不能判断模型优劣；应比较同一条河中的版本，并结合NRMSE和持久性基线。测试段是同一个合成序列的后部，不代表独立的极端洪水、另一年份或另一个真实流域。若需逐预测步指标，可以从回测CSV按 `lead_hours` 分组重新计算；当前不声称已有逐步指标报告。

网站中的6小时外推以输入文件的末时刻为预报起点；默认合成数据的这个时间并不是今天。没有后续参考序列时，不能给这6个点附上验证指标。模型指标来自有参考值的留出测试窗口。可视化的历史或参考曲线须保留“合成”标识。

外推区间使用物理版本在验证集上、逐预测步和逐节点的绝对残差90%分位数，围绕点预测形成对称范围，下限截到零；还报告独立测试窗口中的实际覆盖率。它是经验误差范围，不是模型参数置信区间，也不保证未来90%覆盖。相同验证集同时用于模型选择与区间宽度估计，加上时间自相关与季节变化，都会限制该区间的解释。

## 接入真实数据前的准备

真实接入须先整理成固定每小时、五个节点同时齐全的长表，并使用节点编号 `r1`、`r2`、`r3`、`r4`、`r5`。CSV必须包含以下精确列名，每个 `(timestamp, station_id)` 只能有一条记录；节点排列、变量单位、时间基准和采样间隔必须一致。

```csv
timestamp,station_id,water_depth,discharge,sediment,temperature,rainfall
2024-01-01T00:00:00,r1,3.2,900.0,0.15,8.0,0.0
```

这一行仅说明格式，不构成可训练的数据集。每个时间戳还须有 `r2` 至 `r5` 的记录，随后每隔一小时连续记录。`season_sin` 在读取时派生，无需CSV提供；额外的来源状态或质量列可归档，当前模型不将它们作为特征。数值须是有限数，水深、流量、泥沙和雨量不能为负，温度可为负；导入器会拒绝缺列、重复、缺站和不规则时间轴。ISO时间戳统一转成UTC，未带时区的时间被视为UTC，因此实际本地记录应显式标注 `+08:00` 等偏移。完整格式仍以当前命令的 `--help` 和模块校验为准。

按河流ID保存文件，例如 `data/observed/yellow_river.csv` 和 `data/observed/mississippi.csv`，再执行：

```bash
python3 scripts/run_multi_river.py --data-dir data/observed --rivers yellow_river mississippi --epochs 25 --seed 42 --transfer-seeds 42,43,44
```

指定真实资料目录时，所选河流的CSV须全部存在且通过校验，缺失或无效文件不会自动替换成合成数据。目标迁移需要同时选择黄河和密西西比河。完整目录可以继续加入 `yangtze.csv`、`amazon.csv`、`nile.csv`、`danube.csv`。传入 `--data-dir` 会将运行资料标为 `observed`，使用者须确保CSV确实来自观测；通过格式校验不等于数据真实性认证。此选项也不会把未经校准的示意断面变成真实几何；真实水力校准仍需另行完成。

水深字段 `water_depth` 必须表示河床以上的有效水深。测站 `stage` / `gage height` 是相对测站基准面的水位，不能直接改名为水深：应记录基准面与河床高程，先换算为同一高程基准的水面高程，再计算 `water_depth = water_surface_elevation - bed_elevation`。河床冲淤时，还需说明床面高程的适用日期。没有断面和基准面证据时，应修改模型处理测站水位，不能把它套入当前Manning水深公式。

流量单位统一为m³/s，悬沙浓度统一为kg/m³；mg/L除以1000才是kg/m³。水位/水深以m计，温度以°C计，降水须明确为每小时累计量而非日累计量。日均流量不能复制成小时观测，离散采样泥沙不能视为每小时真实标签。原始缺测、传感器质量码、异常、暂定值与批准值应留在原始归档中；公开指标说明采用了哪些过滤规则。

当前完整样本接口要求整点时间及五节点全部变量完整。缺测应在预处理阶段明确处理：可以删除受影响的预测窗口，或另行实现带缺测掩码的模型；若填补输入，填补算法须只使用预报起点之前可获得的数据，并在训练段拟合。禁止双向插值、用测试均值填补、用未来雨量、用未来调度记录，或跨训练/验证/测试边界补齐。原始数据应先按时间切分，再应用各自允许的预处理规则。

密西西比河数据可从 [USGS现代API迁移指南](https://api.waterdata.usgs.gov/docs/ogcapi/migration/)了解站点目录和连续观测；新项目优先采用 `https://api.waterdata.usgs.gov/ogcapi/v1/`，例如 `collections/continuous` 与 `collections/monitoring-locations`。官方于2026年发布v1，并计划于2027年一季度退役旧WaterServices。[v1公告](https://waterdata.usgs.gov/blog/api-v1-release/)、[退役公告](https://waterdata.usgs.gov/blog/api-waterservices-decom/)。本实验不自动联网下载数据；获得真实资料后仍需核验五个站点的时段交集、单位、基准面、质量和使用许可。

进一步比较协议见 [黄河与密西西比河比较学习](yellow_mississippi_comparison.md)，展示与打包见 [个人网站展示指引](publishing_guide.md)。出处清单在 `configs/river_sources.json`，来源访问日期为2026-10-02。
