# Cloudflare Pages 部署指南

这是一个纯静态网站：黄河与密西西比河比较文章、季节流量图和模型下载均可直接发布。Cloudflare 不需要运行训练脚本、安装 PyTorch 或启动后端；下载的模型仍在本地运行。已有研究内容和数据来源不变。

## 本地构建

在仓库根目录运行（Python 3.10+，不需要安装任何依赖）：

```bash
python3 -B scripts/build_cloudflare_pages.py
```

输出目录为 `dist/`，其 `index.html` 是完整首页，不是跳转页。原来的 `/website/` 路径和七个文章锚点继续可用。模型、数据、代码和文档来自同一份审核过的发布清单；不会带入旧合成实验、数据库、环境或备份。

构建检查原始数据校验和、发布权限、文件数量及单文件大小，并生成 `publication_manifest.json`。可以重复运行：只有未被修改的本工具输出才会被替换，上一版保留在 `.pages-build-backups/`；非构建目录或自行添加、修改过的内容不会被覆盖。

本地预览：

```bash
python3 -m http.server 8080 --directory dist
```

访问 [http://localhost:8080/](http://localhost:8080/)。检查文章导航、图表、手机布局以及两个模型下载。普通本地 HTTP 服务不会解释 Cloudflare 的 `_headers`，响应头和自定义 404 的最终行为需在 Pages 预览部署中检查。

## 方式一：连接 GitHub（推荐）

将这些改动提交到你准备部署的分支，然后在 Cloudflare 的 **Workers & Pages → Create application → Pages → Connect to Git** 中选择 `Wiiiiiiiley/The-Yellow-River`。如果 GitHub 仓库是私有仓库，先在 Cloudflare 的 GitHub 授权界面允许访问该仓库。

构建设置：

| 设置 | 值 |
| --- | --- |
| Framework preset | `None` |
| Root directory | 留空（仓库根目录） |
| Build command | `python3 -B scripts/build_cloudflare_pages.py` |
| Build output directory | `dist` |
| Production branch | 选择包含本次改动的实际发布分支 |
| 构建环境变量（Production 和 Preview 都设置） | `SKIP_DEPENDENCY_INSTALL=1` |
| 可选固定 Python 版本 | `PYTHON_VERSION=3.13.3` |

**不要省略 `SKIP_DEPENDENCY_INSTALL`**：仓库原有 `requirements.txt` 用于本地模型研究，静态发布不应自动安装这些重量级依赖。这个变量要在 Pages 的构建环境变量中设置，不要放到 Wrangler 的 `[vars]`。

`wrangler.toml` 已指定 `pages_build_output_dir = "./dist"`。默认项目名是 `the-yellow-river`；如果使用其他 Pages 项目名，请同步修改文件中的 `name`。配置不包含 Worker 入口、Functions、数据库或密钥，也不能代替上面的 Git 构建设置。

## 方式二：直接上传

如果暂时不连接 GitHub，先构建一个新 ZIP：

```bash
python3 -B scripts/build_cloudflare_pages.py --zip cloudflare_pages.zip
```

在 Pages 创建 **Direct Upload** 项目，上传 `cloudflare_pages.zip`，或者拖入 `dist/` 文件夹。ZIP 解压后直接得到 `index.html`、`404.html`、`_headers`、`website/` 和下载目录，不套额外父文件夹。不要上传整个研究仓库，也不要只上传 `website/`，否则模型下载会失效。

ZIP 不会覆盖已有同名文件；再次导出请指定新的名字，例如 `--zip cloudflare_pages_v2.zip`。现有 `public_river_lab.zip` 是通用便携包，不是这个根目录直接展开的 Pages 上传包。

也可在安装并登录 Wrangler 后手动部署目录：

```bash
npx wrangler pages deploy dist --project-name the-yellow-river
```

Wrangler 上传目录，不接受 ZIP。不要把这条部署命令设为 Git 构建命令。Direct Upload 项目不能直接改成 Git 集成；需要自动随提交更新的话，建议一开始选择 Git 集成。

## 部署后检查

- 域名根路径 `/` 直接显示首页；`/#comparison`、`/#study-culture` 和 `/website/#study-culture` 均可访问。
- 手机宽度下导航和季节流量图正常，两个模型权重以及 CSV、来源文档、源码能打开或下载。
- 一个不存在的路径应返回 404，而不是把首页 HTML 当作模型文件返回。
- `_headers` 指定源码/文档的文本类型和模型的附件下载类型；无版本哈希的 CSS/JS 不设置永久缓存。
- 网站展示的是已有历史预测结果，不是在线实时预测服务。

当前发布包远低于 Pages 的单文件 25 MiB 上限；构建也限制最多 20,000 个文件。Dashboard 拖拽上传另有 1,000 文件限制；若未来超过该数量，使用 Git 集成或 Wrangler。

本指南和本地构建不代表已部署，也不会自动提交或推送 GitHub。

## Cloudflare 官方说明

- [构建设置](https://developers.cloudflare.com/pages/configuration/build-configuration/)
- [构建环境与跳过依赖安装](https://developers.cloudflare.com/pages/configuration/build-image/)
- [Pages 的 Wrangler 配置](https://developers.cloudflare.com/pages/functions/wrangler-configuration/)
- [静态页面、路由与 404](https://developers.cloudflare.com/pages/configuration/serving-pages/)
- [自定义响应头](https://developers.cloudflare.com/pages/configuration/headers/)
- [直接上传](https://developers.cloudflare.com/pages/get-started/direct-upload/)
- [平台限制](https://developers.cloudflare.com/pages/platform/limits/)
