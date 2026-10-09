"""Build the static learning site from COURSE.md and canonical .ipynb files."""
import html
import json
import re
import shutil
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '_site'

def render(text):
    return markdown.markdown(text, extensions=['fenced_code', 'tables', 'toc'])

def main():
    OUT.mkdir(exist_ok=True)
    for path in (ROOT / 'site').iterdir():
        if path.is_file():
            shutil.copy2(path, OUT / path.name)
    source = (ROOT / 'COURSE.md').read_text(encoding='utf-8')
    matches = list(re.finditer(r'^## 第 (\d+) 课 (.+)$', source, re.M))
    lessons = []
    for i, match in enumerate(matches):
        body = source[match.start():matches[i+1].start() if i+1<len(matches) else len(source)]
        body = body.replace('(visuals/attention.html)', '(#attention)')
        lessons.append({'number': int(match[1]), 'title': f'第 {match[1]} 课 {match[2]}', 'html': render(body)})
    assert len(lessons) == 32, f'Expected 32 lessons, got {len(lessons)}'
    (OUT / 'course.json').write_text(json.dumps({'lessons': lessons}, ensure_ascii=False), encoding='utf-8')
    notebooks = []
    for path in sorted((ROOT / 'notebooks').glob('*.ipynb')):
        nb = json.loads(path.read_text(encoding='utf-8'))
        course_meta = nb['metadata']['course']
        cells = []
        for cell in nb['cells']:
            item = {'cell_type':cell['cell_type'], 'source':cell['source']}
            if cell['cell_type']=='markdown':
                item['html']=render(''.join(cell['source']))
            cells.append(item)
        notebooks.append({'number':course_meta['number'], 'title':course_meta['title'], 'filename':path.name, 'notebook':nb, 'cells':cells})
    (OUT / 'notebooks.json').write_text(json.dumps(notebooks, ensure_ascii=False), encoding='utf-8')
    shutil.copytree(ROOT / 'notebooks', OUT / 'notebooks', dirs_exist_ok=True)
    shutil.copytree(ROOT / 'visuals', OUT / 'visuals', dirs_exist_ok=True)
    shutil.copy2(ROOT / '课程讲义.html', OUT / 'book.html')
    (OUT / '.nojekyll').write_text('')
    print(f'Built {len(lessons)} lessons, {len(notebooks)} notebooks → _site')

if __name__ == '__main__':
    main()
