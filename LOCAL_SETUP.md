# 在网页与 VS Code 之间切换

## 网页实验

网站提供 01、02、06 三份入门 Notebook 的可编辑代码格。按顺序运行；同一次实验的代码格共享 Python 变量。首次运行通过 CDN 加载固定版本 Pyodide 0.27.7，只使用浏览器 CPU。

编辑保存到当前浏览器的 localStorage，不会自动提交 GitHub。点击“下载我的 Notebook”保存修改后的 `.ipynb`。刷新、切换实验或停止运行会清空 Python 内存；输出不自动保存到 Notebook。浏览器虚拟文件不是你的真实磁盘，清除网站数据会丢失浏览器内保存的编辑。

目前提供标准库练习；未提供网页内的 PyTorch/CUDA、服务器 Jupyter kernel 或集群连接。训练课的完整实现仍是仓库中的 Python 程序。

## VS Code 的第一份 Notebook

1. 克隆仓库，VS Code 的“打开文件夹”选择 `LLM_Agent_Full_link`。
2. 安装仓库推荐的 Python 和 Jupyter 扩展。
3. 在 VS Code 终端创建专用环境：

```bash
python3 -m venv .venv-notebook
source .venv-notebook/bin/activate
python -m pip install -r requirements-notebook.txt
```

Windows PowerShell 的激活命令为 `.venv-notebook\Scripts\Activate.ps1`。如果公司限制脚本执行，使用已有的合规 Python 环境，无需调整系统策略。

4. 打开 `notebooks/01_python_success.ipynb`，点击右上角“选择内核”，选择刚创建的环境。
5. 依次运行代码格。网站下载的 Notebook 同样可以打开；它包含你改过的代码。

第一次成功后再使用 `notebooks/02_jsonl.ipynb` 和 `notebooks/06_attention.ipynb`。这三份 Notebook 使用标准库，数学计算与网页一致。

## 在集群使用

使用公司的现有 VS Code Remote SSH / 开发机入口。把仓库放在节点可访问的目录，按 `configs/cluster.env.example` 配置，再按讲义通过调度器申请资源。选择对应训练环境的 Python；Notebook 并不会自动申请 GPU。不要在登录节点运行长时间训练。

## 本地预览网站

在已安装 requirements-notebook.txt 的环境中：

```bash
python tools/build_book.py
python tools/build_site.py
python -m http.server 8000 --bind 127.0.0.1 --directory _site
```

打开 `http://127.0.0.1:8000`。网站通过 HTTP 加载课程与 Python worker，不能直接双击 `site/index.html`。离线阅读可直接双击根目录 `课程讲义.html`；独立的 `visuals/attention.html` 也可以直接打开。

## GitHub Pages 的首次设置

仓库 Settings → Pages → Build and deployment → Source 选择 **GitHub Actions**。之后每次 main 更新，工作流自动重新构建并部署。工作流的 build 步骤通过并不等于网站部署成功，必须看到 deploy 成功。

课程仓库已获授权公开，网站使用 `https://shichengf.github.io/LLM_Agent_Full_link/`。实际服务器路径、地址与登录信息只保存在本地配置中；示例文件中的路径和地址均为教学占位值。

网站只发布构建出的 `_site`，不会发布 runs、集群配置或模型目录。发布内容包含课程、教学源码阅读版和 Notebook。

参考：[Pyodide worker](https://pyodide.org/en/0.27.7/usage/webworker.html)、[GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)。
