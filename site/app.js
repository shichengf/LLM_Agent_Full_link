const content = document.getElementById("content"),
  crumb = document.getElementById("breadcrumb");
const repo = "https://github.com/shichengf/LLM_Agent_Full_link";
let course,
  notebooks,
  worker = null,
  busy = false,
  timer = null,
  activeOutput = null,
  currentNotebook = null,
  frameObserver = null;
const groups = [
  ["01–05", "恢复编程手感", 1, 5],
  ["06–10", "读懂与训练 Transformer", 6, 10],
  ["11–15", "开源模型与后训练", 11, 15],
  ["16–19", "多卡走向多节点", 16, 19],
  ["20–22", "推理服务与压测", 20, 22],
  ["23–26", "Agent 系统与评测", 23, 26],
  ["27–30", "为 Agent 训练策略", 27, 30],
  ["31–32", "规模实验与完整交付", 31, 32],
];
function saved(key, fallback) {
  try {
    return JSON.parse(localStorage.getItem(key)) ?? fallback;
  } catch {
    return fallback;
  }
}
function save(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}
function el(tag, text, className) {
  const n = document.createElement(tag);
  if (text !== undefined) n.textContent = text;
  if (className) n.className = className;
  return n;
}
function nav() {
  const box = document.getElementById("lessons");
  groups.forEach(([range, name, start, end]) => {
    const d = el("details");
    d.open = start === 1;
    d.appendChild(el("summary", range + " / " + name));
    course.lessons
      .filter((x) => x.number >= start && x.number <= end)
      .forEach((x) => {
        const a = el("a", x.title.replace(/^第 /, "").replace(" 课 ", " · "));
        a.href = "#lesson-" + String(x.number).padStart(2, "0");
        d.appendChild(a);
      });
    box.appendChild(d);
  });
}
document.getElementById("search").addEventListener("input", (e) => {
  const q = e.target.value.trim().toLowerCase();
  document.querySelectorAll("#lessons details").forEach((d) => {
    let visible = 0;
    d.querySelectorAll("a").forEach((a) => {
      a.hidden = !a.textContent.toLowerCase().includes(q);
      if (!a.hidden) visible++;
    });
    d.hidden = !visible;
    if (q) d.open = !!visible;
  });
});
document.getElementById("menu").addEventListener("click", () => {
  const open = document.getElementById("sidebar").classList.toggle("open");
  document.getElementById("menu").setAttribute("aria-expanded", String(open));
});
function copyButtons() {
  content.querySelectorAll("pre").forEach((p) => {
    if (!p.querySelector("code")) return;
    const b = el("button", "复制", "copy");
    b.type = "button";
    b.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(
          p.querySelector("code").textContent,
        );
        b.textContent = "已复制";
      } catch {
        b.textContent = "请选中复制";
      }
    });
    p.appendChild(b);
  });
}
function shutdown() {
  if (worker) worker.terminate();
  worker = null;
  busy = false;
  clearTimeout(timer);
  timer = null;
  activeOutput = null;
}
function setStatus(text) {
  const s = document.getElementById("lab-status");
  if (s) s.textContent = text;
}
function controls() {
  document.querySelectorAll("[data-run]").forEach((b) => (b.disabled = busy));
  const stop = document.getElementById("stop");
  if (stop) stop.disabled = !busy;
}
function appendOutput(text) {
  if (activeOutput) {
    const max = 30000;
    const next = activeOutput.textContent + text;
    activeOutput.textContent =
      next.length > max ? "[保留最后 30000 字符]\n" + next.slice(-max) : next;
  }
}
function run(code, output) {
  if (busy) return;
  busy = true;
  activeOutput = output;
  output.textContent = "";
  controls();
  setStatus(worker ? "运行中…" : "首次加载浏览器 Python，可能需要数十秒…");
  if (!worker) {
    worker = new Worker("python-worker.js", { type: "module" });
    worker.onmessage = (e) => {
      const { type, text } = e.data;
      if (type === "status") setStatus(text);
      else if (type === "stdout" || type === "stderr")
        appendOutput(text + "\n");
      else if (type === "done" || type === "error") {
        clearTimeout(timer);
        if (type === "error") appendOutput(text);
        else if (!activeOutput.textContent)
          appendOutput("运行完成（没有输出；可以使用 print 查看变量）。");
        busy = false;
        controls();
        setStatus(
          type === "error"
            ? "运行失败；阅读下方错误，修改后再试。"
            : "运行完成 · 变量保留在当前实验内",
        );
      }
    };
    worker.onerror = (e) => {
      appendOutput(
        "Python 加载或运行失败：" +
          e.message +
          "\n可下载 Notebook 在本地运行。",
      );
      shutdown();
      controls();
      setStatus("浏览器 Python 不可用；检查网络或使用 VS Code。");
    };
  }
  timer = setTimeout(() => {
    appendOutput("\n已超过 90 秒，停止并清空 Python 内存；编辑内容保留。");
    shutdown();
    controls();
    setStatus("运行超时；可修改代码后重试。");
  }, 90000);
  worker.postMessage({ code });
}
function home() {
  crumb.textContent = "课程首页";
  content.innerHTML = `<section class="hero"><div class="eyebrow">A HANDS-ON LEARNING LAB</div><h1>先看懂一个 token，<br>再调通一整个集群。</h1><p class="lede">从久违的第一行 Python，到多节点 LLM 训练、推理与 Agent。用可操作的小例子建立直觉，再让真实实验检验它。</p><div class="actions"><a class="button primary" href="#lesson-01">从第 01 课开始 →</a><a class="button" href="#attention">先看一个交互例子</a></div><p class="subtle">32 课讲义 · 3 份可运行 Notebook · 1 个交互式 Attention 实验</p></section><div class="features"><section class="feature"><span class="number">01 / 看见</span><h3>把抽象概念变成动作</h3><p>点击一个 token，观察可见范围与权重。改变分数，检验你的预测。</p><a href="#attention">操作 Attention →</a></section><section class="feature"><span class="number">02 / 动手</span><h3>就在网页里写 Python</h3><p>编辑、运行、读错误。下载修改后的 Notebook，接着在 VS Code 完成。</p><a href="#lab-01">打开 Python 实验室 →</a></section><section class="feature"><span class="number">03 / 放大</span><h3>把规模当成实验变量</h3><p>从单卡到 2–4 节点，每节点 8 卡。记录通信、显存、吞吐和失败。</p><a href="#lesson-16">进入分布式课程 →</a></section></div><h2>一条完整的实践路径</h2><div id="routes"></div><div class="notice">浏览器实验在你电脑的 CPU 上运行，不连接公司集群。GPU 与多节点章节在 VS Code / Remote SSH 和已分配的计算节点执行；相关脚本尚需真实集群验收。</div><h2>在本地继续</h2><pre><code>git clone https://github.com/shichengf/LLM_Agent_Full_link.git\ncd LLM_Agent_Full_link\npython3 -m src.basics</code></pre><p><a href="${repo}/blob/main/LOCAL_SETUP.md">VS Code 与 Notebook 配置</a> · <a href="book.html">完整离线讲义</a> · <a href="${repo}/blob/main/VALIDATION.md">已验证与待验证范围</a></p>`;
  const routes = document.getElementById("routes");
  groups.forEach(([range, name, start]) => {
    const row = el("div", undefined, "route"),
      a = el("a", name);
    a.href = "#lesson-" + String(start).padStart(2, "0");
    row.append(el("span", range), a, el("small", "讲解 / 练习 / 验收"));
    routes.appendChild(row);
  });
  copyButtons();
}
function lesson(number) {
  const item = course.lessons.find((x) => x.number === number);
  if (!item) {
    home();
    return;
  }
  crumb.textContent = "课程 / " + String(number).padStart(2, "0");
  content.innerHTML = "<article>" + item.html + "</article>";
  const article = content.querySelector("article");
  if ([1, 2, 6].includes(number)) {
    const a = el("a", "打开本课可运行 Notebook →", "button");
    a.href = "#lab-" + String(number).padStart(2, "0");
    article.prepend(a);
  }
  const label = el("label", undefined, "done"),
    check = document.createElement("input");
  check.type = "checkbox";
  check.checked = saved("llm-agent-done-" + number, false);
  check.addEventListener("change", () =>
    save("llm-agent-done-" + number, check.checked),
  );
  label.append(
    check,
    document.createTextNode(" 我已完成本课验收（仅保存到当前浏览器）"),
  );
  article.appendChild(label);
  const pager = el("div", undefined, "lesson-pager");
  for (const n of [number - 1, number + 1]) {
    if (n >= 1 && n <= 32) {
      const a = el(
        "a",
        (n < number ? "← 上一课" : "下一课 →") +
          " · " +
          String(n).padStart(2, "0"),
      );
      a.href = "#lesson-" + String(n).padStart(2, "0");
      pager.appendChild(a);
    }
  }
  article.appendChild(pager);
  copyButtons();
}
function attention() {
  crumb.textContent = "交互实验 / 因果 Attention";
  content.innerHTML =
    '<div class="actions"><a class="button" href="#lab-06">用 Python 重现 →</a><a class="button" href="#lesson-06">对应课程讲解</a></div><iframe class="visual-frame" title="因果 Attention 交互实验" src="visuals/attention.html"></iframe>';
  const frame = content.querySelector("iframe");
  frame.addEventListener("load", () => {
    const inner = frame.contentDocument?.querySelector("main");
    if (inner) {
      const resize = () => {
        const height = Math.ceil(inner.getBoundingClientRect().height) + "px";
        if (frame.style.height !== height) frame.style.height = height;
      };
      resize();
      frameObserver = new ResizeObserver(resize);
      frameObserver.observe(inner);
    }
  });
}
function notebook(number) {
  const nb = notebooks.find((n) => n.number === number) || notebooks[0];
  currentNotebook = nb;
  crumb.textContent = "Python 实验室 / " + nb.title;
  content.innerHTML =
    '<div class="eyebrow">BROWSER PYTHON / 同一份代码，本地继续</div><h1></h1><p class="lede">先预测输出，再按顺序运行代码格。改变输入，看看你的解释是否成立。</p><div class="lab-toolbar"><select id="notebook-picker" aria-label="选择实验"></select><button id="download">下载我的 Notebook</button><button id="stop" disabled>停止运行</button></div><div id="lab-status" class="lab-status" role="status">Python 尚未加载；点击“运行”开始。</div><p class="subtle">编辑自动保存到当前浏览器。代码格共享变量；切换实验、刷新或停止会清空 Python 内存。首次运行需联网下载 Python。网页文件位于临时内存，重要结果请下载。</p><div id="cells"></div><p><a href="' +
    repo +
    '/blob/main/LOCAL_SETUP.md">在 VS Code 打开 Notebook →</a></p>';
  content.querySelector("h1").textContent = nb.title;
  const select = document.getElementById("notebook-picker");
  notebooks.forEach((n) => {
    const o = el("option", n.title);
    o.value = n.number;
    o.selected = n.number === nb.number;
    select.appendChild(o);
  });
  select.addEventListener("change", () => {
    location.hash = "lab-" + String(select.value).padStart(2, "0");
  });
  const cells = document.getElementById("cells");
  nb.cells.forEach((cell, index) => {
    if (cell.cell_type === "markdown") {
      const prose = el("div", undefined, "cell-prose");
      prose.innerHTML = cell.html;
      cells.appendChild(prose);
    } else {
      const box = el("section", undefined, "cell");
      const head = el("div", undefined, "cell-header");
      head.appendChild(el("span", "Python · 代码格 " + (index + 1)));
      const button = el("button", "运行 ▶");
      button.dataset.run = "true";
      const editor = document.createElement("textarea");
      editor.spellcheck = false;
      editor.setAttribute("aria-label", nb.title + " 代码格 " + (index + 1));
      editor.dataset.index = index;
      editor.value = saved(
        "llm-agent-nb-" + nb.number + "-" + index,
        cell.source.join(""),
      );
      editor.rows = Math.min(
        22,
        Math.max(6, editor.value.split("\n").length + 1),
      );
      editor.addEventListener("input", () => {
        if (!save("llm-agent-nb-" + nb.number + "-" + index, editor.value))
          setStatus("浏览器无法保存编辑；请下载 Notebook 备份。");
      });
      editor.addEventListener("keydown", (e) => {
        if (e.key === "Tab") {
          e.preventDefault();
          const start = editor.selectionStart;
          editor.setRangeText("    ", start, editor.selectionEnd, "end");
          editor.dispatchEvent(new Event("input"));
        }
        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
          e.preventDefault();
          run(editor.value, output);
        }
      });
      const output = el("pre", "尚未运行", "cell-output");
      output.setAttribute("aria-label", "Python 输出");
      button.addEventListener("click", () => run(editor.value, output));
      head.appendChild(button);
      box.append(head, editor, output);
      cells.appendChild(box);
    }
  });
  document.getElementById("stop").addEventListener("click", () => {
    appendOutput("\n已停止运行，Python 内存已清空。");
    shutdown();
    controls();
    setStatus("已停止；代码编辑保留，请从第一格重新运行。");
  });
  document.getElementById("download").addEventListener("click", () => {
    const data = JSON.parse(JSON.stringify(nb.notebook));
    content.querySelectorAll("textarea[data-index]").forEach((t) => {
      const c = data.cells[Number(t.dataset.index)];
      c.source = t.value.split(/(?<=\n)/);
      c.outputs = [];
      c.execution_count = null;
    });
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/x-ipynb+json",
    });
    const url = URL.createObjectURL(blob),
      a = el("a");
    a.href = url;
    a.download = nb.filename;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
}
function render() {
  shutdown();
  if (frameObserver) {
    frameObserver.disconnect();
    frameObserver = null;
  }
  const hash = location.hash.slice(1) || "home";
  if (hash.startsWith("lesson-")) lesson(Number(hash.slice(7)));
  else if (hash.startsWith("lab-")) notebook(Number(hash.slice(4)));
  else if (hash === "attention") attention();
  else home();
  document.querySelectorAll("nav a").forEach((a) => {
    const active = a.getAttribute("href") === "#" + hash;
    a.classList.toggle("active", active);
    if (active && a.closest("details")) a.closest("details").open = true;
  });
  document.getElementById("sidebar").classList.remove("open");
  document.getElementById("menu").setAttribute("aria-expanded", "false");
  window.scrollTo(0, 0);
}
try {
  const results = await Promise.all(
    ["course.json", "notebooks.json"].map(async (url) => {
      const r = await fetch(url);
      if (!r.ok) throw new Error("无法加载 " + url);
      return r.json();
    }),
  );
  [course, notebooks] = results;
  nav();
  window.addEventListener("hashchange", render);
  render();
} catch (e) {
  content.innerHTML =
    '<h1>课程暂时无法加载</h1><p>请刷新页面；本地预览需要启动 HTTP 服务。</p><p><a href="' +
    repo +
    '/blob/main/COURSE.md">先阅读 GitHub 上的完整讲义 →</a></p>';
  content.appendChild(el("pre", e.message));
}
