"""Build a self-contained, offline-readable course book; authoring dependency: markdown."""
import html
import re
from pathlib import Path
import markdown

root=Path(__file__).resolve().parents[1]
md=markdown.Markdown(extensions=['fenced_code','tables','toc'],extension_configs={'toc':{'toc_depth':'2'}})
body=md.convert((root/'COURSE.md').read_text())
appendix=[]
for folder in ('src','cluster','tools','answers','configs','tests'):
    for path in sorted((root/folder).iterdir()):
        if not path.is_file() or path.suffix not in ('.py','.sh','.sbatch','.md','.example') or path.name=='build_book.py':continue
        text=path.read_text();name=str(path.relative_to(root))
        content=markdown.markdown(text,extensions=['fenced_code','tables']) if path.suffix=='.md' else '<pre><code>'+html.escape(text)+'</code></pre>'
        appendix.append('<details class="source"><summary>'+html.escape(name)+'</summary>'+content+'</details>')
validation=markdown.markdown((root/'VALIDATION.md').read_text(),extensions=['tables'])
sources=markdown.markdown((root/'SOURCES.md').read_text())
css='''
:root{--ink:#172b35;--muted:#597078;--accent:#007a72;--paper:#fff;--line:#dce6e8;--code:#11242d}
*{box-sizing:border-box}body{margin:0;background:#f3f6f6;color:var(--ink);font:16px/1.85 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
aside{position:fixed;inset:0 auto 0 0;width:285px;padding:28px 20px;background:#eaf1f1;overflow:auto;border-right:1px solid var(--line)}
aside strong{display:block;font-size:20px;margin-bottom:10px}aside p{font-size:13px;color:var(--muted)}
nav ul{padding:0;list-style:none}nav li{margin:7px 0}nav a{font-size:13px;color:#31525a;text-decoration:none;display:block;padding:4px 8px;border-radius:5px}nav a:hover{background:#d9e8e7}
main{margin-left:285px;padding:48px 5vw 100px;max-width:1280px}article{background:white;padding:44px 56px;border:1px solid var(--line);border-radius:12px}
.eyebrow{color:var(--accent);font-size:12px;font-weight:700;letter-spacing:2px}.chips{display:flex;gap:9px;flex-wrap:wrap;margin:22px 0 32px}.chips span{background:#e7f3f0;padding:4px 12px;border-radius:18px;font-size:13px}
h1{font-size:34px;line-height:1.35;margin-top:12px}h2{font-size:24px;line-height:1.5;border-top:1px solid var(--line);padding-top:32px;margin-top:50px;scroll-margin-top:24px}h3{font-size:19px;margin-top:30px}p{margin:15px 0}a{color:var(--accent);overflow-wrap:anywhere}
pre{background:var(--code);color:#e0f0f3;padding:22px 20px;border-radius:8px;overflow:auto;line-height:1.6;font-size:13px;position:relative}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}p code,li code,td code{background:#edf2f3;padding:2px 5px;border-radius:3px;font-size:.9em}
table{border-collapse:collapse;width:100%;font-size:14px;margin:22px 0;display:block;overflow:auto}th,td{padding:10px 13px;border:1px solid var(--line);vertical-align:top}th{background:#edf4f3;text-align:left}
.source{border:1px solid var(--line);border-radius:6px;margin:12px 0;padding:12px 16px}.source summary{cursor:pointer;font-family:monospace;font-size:14px;font-weight:600}.source pre{max-height:650px}
.copy{position:absolute;right:8px;top:8px;border:1px solid #4d6973;color:#c9e0e4;background:#203944;border-radius:4px;cursor:pointer;padding:3px 7px;font-size:11px}
input{width:100%;padding:9px;border:1px solid #b6cccc;border-radius:5px;background:#fff;margin-bottom:8px}footer{color:var(--muted);font-size:13px;margin-top:35px}
@media(max-width:900px){aside{position:relative;width:auto;max-height:320px;border-bottom:1px solid var(--line)}main{margin:0;padding:18px}article{padding:26px 21px}h1{font-size:27px}h2{font-size:22px}.chips{margin-bottom:20px}}
@media print{aside,.copy,.chips{display:none}main{margin:0;padding:0;max-width:none}article{border:0;padding:0}body{background:white;font-size:11pt}pre{white-space:pre-wrap;color:black;background:#f3f3f3}h2{break-before:page}.source{break-inside:avoid}}
'''
script='''
document.querySelectorAll('pre').forEach(p=>{const b=document.createElement('button');b.className='copy';b.textContent='复制';b.onclick=async()=>{try{await navigator.clipboard.writeText(p.querySelector('code')?.innerText||p.innerText);b.textContent='已复制';setTimeout(()=>b.textContent='复制',1200)}catch(e){b.textContent='请选中复制'}};p.appendChild(b)});
document.getElementById('search').addEventListener('input',e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('nav li').forEach(li=>li.style.display=li.textContent.toLowerCase().includes(q)?'':'none')});
'''
document='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>从 Python 到多节点 Agent 训练与部署</title><style>'+css+'</style></head><body><aside><strong>LLM × Agent 实践课</strong><p>32 课 · 完整讲义与实验源码<br>从编程基础到多节点系统</p><input id="search" placeholder="搜索课程目录"><nav>'+md.toc+'<a href="#code">源码与参考答案</a><a href="#validation">验证记录</a></nav></aside><main><article><div class="eyebrow">HANDS-ON COURSE · SHICHENG</div><div class="chips"><span>中文讲解</span><span>可运行代码</span><span>8 / 16 / 24 / 32 GPU</span><span>含练习与验收</span></div>'+body+'<h2 id="code">源码与参考答案</h2><p>点击文件名展开。完整目录与可执行文件位于 GitHub 仓库；本阅读版也包含所有主要实现。</p>'+''.join(appendix)+'<section id="validation">'+validation+'</section><section id="sources">'+sources+'</section><footer>按课号学习，按证据验收。先读 README，再执行第一课。</footer></article></main><script>'+script+'</script></body></html>'
(root/'课程讲义.html').write_text(document,encoding='utf-8')
print('book bytes',len(document.encode()),'source sections',len(appendix))
