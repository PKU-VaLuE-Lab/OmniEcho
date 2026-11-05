from pathlib import Path
import csv
import html
import json

SITE = Path(__file__).resolve().parents[1] / 'docs'
columns = [('cameraRotation','Camera<br>Rotation'),('sourceDirection','Source<br>Direction'),('localization','3D<br>Localization'),('sourceMotion','Source<br>Motion'),('cognitiveMap','Cognitive<br>Map'),('overall','Overall')]
raw = [
    ('OmniEcho', 'Qwen3-Omni-30B-A3B', 'A-foa + V', 'paper', [41.7,22.6,14.2,24.7,47.5,28.5]),
    ('Qwen3-Omni-30B-A3B', 'Qwen3-Omni-30B-A3B', 'A-single + V', 'paper', [32.7,16.3,7.7,20.9,22.3,18.5]),
    ('Qwen2.5-Omni-7B', 'Qwen2.5-Omni-7B', 'A-single + V', 'paper', [24.9,8.0,5.4,14.0,20.0,12.6]),
    ('SO-7B', 'Qwen2.5-Omni-7B', 'A-foa + V', 'paper', [21.4,5.2,0.0,17.1,22.8,11.4]),
    ('Qwen3.5-Omni-Plus', None, 'Audio + V', 'additional', [52.4,16.1,19.6,22.2,25.2,23.3]),
    ('Gemini3.7-flash', None, 'Audio + V', 'additional', [45.3,32.8,24.4,33.4,26.2,31.4]),
    ('Gemini3.8-flash', None, 'Audio + V', 'additional', [53.7,34.1,22.1,31.3,26.0,32.0]),
    ('muse-spark-1.2', None, 'Audio + V', 'additional', [55.0,34.2,19.6,29.6,25.2,31.3]),
    ('doubao-seed-2-0-lite-260428', None, 'Audio + V', 'additional', [34.3,30.6,22.1,29.6,25.3,28.3]),
    ('Qwen3.8-omni-flash', None, 'Audio + V', 'additional', [74.1,47.9,28.6,37.4,39.5,43.9]),
]
models = []
for name, base, setting, source, values in raw:
    models.append(dict(model=name, baseModel=base, setting=setting, source=source, **dict(zip([x[0] for x in columns], values))))
models.sort(key=lambda m: -m['overall'])
for index, model in enumerate(models, 1):
    model['rank'] = index
metadata = {
    'benchmark': 'OmniEchoBench-QA', 'track': 'Audio & Vision', 'updated': '2026-10-18',
    'unit': 'percent', 'higherIsBetter': True,
    'notes': [
        'Paper Audio & Vision results combined with the additional results supplied by the project maintainer.',
        'OmniEcho uses the paper values: Camera Rotation 41.7 and 3D Localization 14.2, as confirmed by the project maintainer.',
        'Audio channel configuration and base models for the additional results were not specified.',
        'Overall scores are reproduced as reported; they are not recalculated as the mean of the five displayed task scores.',
        'Ranks are based on Overall descending. No audio-only results are included.',
    ], 'models': models,
}
(SITE / 'assets' / 'leaderboard-data.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8', newline='\n')
with (SITE / 'assets' / 'leaderboard.csv').open('w', encoding='utf-8', newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=['rank','model','baseModel','source']+[x[0] for x in columns],extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(models)

best = {key:max(m[key] for m in models) for key,_ in columns}
rows=[]
for m in models:
    data=' '.join(f'data-{key.replace("R", "-r").replace("D", "-d").replace("M", "-m")}="{value}"' for key,value in m.items() if key in best)
    ours=m['model']=='OmniEcho'
    name=html.escape(m['model']);base=html.escape(m['baseModel'] or '')
    label=' <span class="ours-badge">OURS</span>' if ours else ''
    subline=f'<small>Base: {base}</small>' if base else ''
    setting=m['setting']+(' <span aria-label="Audio channel configuration not specified">*</span>' if m['source']=='additional' else '')
    cells=[]
    for key,_ in columns:
        classes=('overall-cell ' if key=='overall' else '')+('best' if m[key]==best[key] else '')
        cells.append(f'<td class="{classes.strip()}"'+(' aria-label="'+str(m[key])+' percent, best in column"' if m[key]==best[key] else '')+f'>{m[key]:.1f}</td>')
    rows.append(f'<tr class="{"our-model" if ours else ""}" data-model="{name}" data-rank="{m["rank"]}" {data}><td>{m["rank"]:02}</td><td class="model-cell"><div class="model-name">{name}{label}</div>{subline}</td>{"".join(cells)}</tr>')

headers=''.join(f'<th scope="col" data-sort="{key}" aria-sort="{"descending" if key=="overall" else "none"}"><button type="button" aria-label="Sort by {title.replace("<br>", " ")}">{title} <span class="sort-arrow" aria-hidden="true">{"↓" if key=="overall" else "↕"}</span></button></th>' for key,title in columns)
template = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Leaderboard — OmniEchoBench</title>
<meta name="description" content="Compare 10 models on OmniEchoBench-QA. Explore Audio & Vision results across camera rotation, source direction, 3D localization, source motion, and cognitive map.">
<meta name="theme-color" content="#ffffff"><meta property="og:title" content="OmniEchoBench — Audio & Vision Leaderboard"><meta property="og:type" content="website"><meta property="og:image" content="https://pku-value-lab.github.io/OmniEcho/assets/social-card.png">
<link rel="canonical" href="https://pku-value-lab.github.io/OmniEcho/leaderboard/">
<link rel="icon" href="../assets/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="../assets/fonts.css"><link rel="stylesheet" href="../assets/style.css"><link rel="stylesheet" href="../assets/light.css">
<script src="../assets/cosmic.js" defer></script><script src="../assets/site.js" defer></script><script src="../assets/leaderboard.js" defer></script>
</head>
<body class="leaderboard-page">
<a class="skip-link" href="#main">Skip to content</a>
<header class="site-header"><div class="nav-wrap"><a class="brand" href="../" aria-label="OmniEcho home"><span class="brand-mark" aria-hidden="true">◎</span> OmniEcho<span class="brand-dot">.</span></a><nav aria-label="Main navigation"><a href="../">Overview</a><a href="../#demo">Demo</a><a href="../#benchmark">Benchmark</a><a class="nav-active" href="./" aria-current="page">Leaderboard <span aria-hidden="true">↗</span></a></nav><a class="nav-github" href="https://github.com/PKU-VaLuE-Lab/OmniEcho" target="_blank" rel="noopener noreferrer">GitHub <span aria-hidden="true">↗</span></a></div></header>
<main id="main" class="container board-main">
<div class="breadcrumbs"><a href="../">OmniEcho</a><span>/</span><span>Benchmark leaderboard</span></div>
<div class="board-heading"><div><p class="eyebrow">OMNIECHOBENCH-QA</p><h1>Spatial intelligence.<br><span>Measured.</span></h1><p>How well can a model understand the world through sound and sight?<br>Compare Audio & Vision performance across five spatial reasoning tasks.</p></div><div class="board-status"><strong><i class="legend-dot teal"></i> Audio & Vision</strong><span>Updated October 18, 2026</span></div></div>
<section class="board-summary" aria-label="Leaderboard summary"><div><small>MODELS EVALUATED</small><strong>10</strong><span>models</span><p>One unified Audio & Vision table</p></div><div><small>HIGHEST OVERALL</small><strong>43.9<em>%</em></strong><p>Qwen3.8-omni-flash</p></div><div><small>OMNIECHO · OVERALL</small><strong class="mint">28.5<em>%</em></strong><p>FOA spatial audio + vision</p></div><div><small>BEST COGNITIVE MAP</small><strong class="mint">47.5<em>%</em></strong><p>OmniEcho</p></div></section>
<section class="table-panel" aria-labelledby="table-title"><div class="table-toolbar"><div class="table-title"><h2 id="table-title">Audio & Vision</h2><span class="count-badge">10 MODELS</span></div><div class="toolbar-actions"><label class="search-wrap"><span aria-hidden="true">⌕</span><span class="sr-only">Search models</span><input id="model-search" type="search" placeholder="Search models…" autocomplete="off"></label><a class="download-button" href="../assets/leaderboard.csv" download="OmniEchoBench-Audio-Vision.csv">Download CSV <span aria-hidden="true">↓</span></a></div></div>
<div class="table-help"><span>Accuracy (%) · <b>Higher is better</b></span><span>Click a metric to sort · Scroll horizontally on smaller screens</span></div>
<noscript><p class="noscript-note">All results are shown below. Enable JavaScript to search and sort.</p></noscript>
<div class="table-scroll" tabindex="0" role="region" aria-label="Scrollable Audio and Vision results"><table class="leaderboard-table"><caption class="sr-only">OmniEchoBench-QA Audio and Vision accuracy. Ten models ranked by reported overall accuracy. Percentages, higher is better.</caption><colgroup><col class="rank-col"><col class="model-col"><col class="score-col" span="5"><col class="overall-col"></colgroup><thead><tr><th scope="col">#</th><th scope="col" class="model-head" data-sort="model" aria-sort="none"><button type="button" aria-label="Sort by model">Model / Base Model <span class="sort-arrow" aria-hidden="true">↕</span></button></th>__HEADERS__</tr></thead><tbody>__ROWS__</tbody></table></div>
<div class="table-bottom"><span id="visible-count">10 of 10 models</span><span><span class="legend-best">● &nbsp; Best in column</span><span>Highlighted row: OmniEcho</span><span>Rank: Overall ↓</span></span></div><span id="filter-status" class="sr-only" role="status" aria-live="polite"></span></section>
<div class="data-notes"><p><strong>Reading the results.</strong> All entries use audio and vision. Overall is reproduced as reported, rather than calculated as a simple mean of the five columns. Ranks remain based on Overall when sorting other metrics.</p><p><strong>Data provenance.</strong> The Audio & Vision results from the <a href="https://arxiv.org/abs/2609.23407" target="_blank" rel="noopener noreferrer">OmniEcho paper</a> are combined with additional results supplied by the project maintainer. OmniEcho uses the paper values (41.7 Camera Rotation; 14.2 3D Localization). <a href="../assets/leaderboard-data.json" download>Download the full data and notes ↗</a></p></div>
<section class="board-explainer" aria-labelledby="metrics-title"><p class="eyebrow">BEHIND THE NUMBERS</p><h2 id="metrics-title">Five ways to understand space.</h2><div class="metric-grid"><article class="metric-card"><span>01 / VIEWPOINT</span><h3>Camera Rotation</h3><p>Infer how the camera rotates from synchronized audio and visual cues.</p></article><article class="metric-card"><span>02 / DIRECTION</span><h3>Source Direction</h3><p>Identify where a sound comes from, including sources outside the frame.</p></article><article class="metric-card"><span>03 / POSITION</span><h3>3D Localization</h3><p>Estimate a sounding source’s position in three-dimensional space.</p></article><article class="metric-card"><span>04 / MOVEMENT</span><h3>Source Motion</h3><p>Reason about the heading and motion of a source over time.</p></article><article class="metric-card"><span>05 / SCENE LAYOUT</span><h3>Cognitive Map</h3><p>Connect heard directions to positions on a bird’s-eye map.</p></article></div></section>
<div class="board-bottom-cta"><div><h3>Discover the research behind the benchmark.</h3><p>Real-world spatial audio, a dedicated FOA encoder, and embodied reasoning.</p></div><a class="button secondary" href="../">Explore OmniEcho <span aria-hidden="true">↗</span></a></div>
</main><footer class="site-footer container"><a class="brand" href="../"><span class="brand-mark" aria-hidden="true">◎</span> OmniEcho<span class="brand-dot">.</span></a><span>Spatial audio. Embodied intelligence.</span><a href="https://github.com/PKU-VaLuE-Lab" target="_blank" rel="noopener noreferrer">PKU VaLuE Lab <span aria-hidden="true">↗</span></a></footer>
</body></html>
'''
(SITE/'leaderboard'/'index.html').write_text(template.replace('__HEADERS__',headers).replace('__ROWS__','\n'.join(rows)),encoding='utf-8', newline='\n')
assert len(models)==10
assert next(m for m in models if m['model']=='OmniEcho')['localization']==14.2
print('Built 10-model leaderboard, JSON and CSV. OmniEcho uses confirmed paper values.')
