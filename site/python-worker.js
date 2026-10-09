import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.mjs";
let runtime = null;
self.onmessage = async (event) => {
  try {
    if (!runtime) {
      self.postMessage({
        type: "status",
        text: "正在加载 Pyodide 0.27.7 浏览器 Python 环境…",
      });
      runtime = await loadPyodide({
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/",
        stdout: (text) => self.postMessage({ type: "stdout", text }),
        stderr: (text) => self.postMessage({ type: "stderr", text }),
      });
    }
    self.postMessage({ type: "status", text: "运行中…" });
    const result = await runtime.runPythonAsync(event.data.code);
    if (result !== undefined) {
      self.postMessage({ type: "stdout", text: String(result) });
      if (result?.destroy) result.destroy();
    }
    self.postMessage({ type: "done" });
  } catch (error) {
    self.postMessage({ type: "error", text: String(error) });
  }
};
