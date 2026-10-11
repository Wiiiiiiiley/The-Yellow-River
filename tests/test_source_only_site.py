"""Regression checks for the observed two-station model publication."""
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
CJK = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]')
STUDY_ANCHORS = {
    'study-hydrology', 'study-sediment', 'study-dams', 'study-flood',
    'study-ecology', 'study-livelihood', 'study-culture',
}


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()
        self.language = None
        self.visible_text = []
        self.accessible_text = []
        self.excluded_depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'html':
            self.language = attrs.get('lang')
        if tag in {'script', 'style'}:
            self.excluded_depth += 1
        for key in ['aria-label', 'aria-description', 'aria-valuetext', 'alt', 'title', 'placeholder']:
            if attrs.get(key):
                self.accessible_text.append(attrs[key])
        if tag == 'meta' and attrs.get('name') == 'description':
            self.accessible_text.append(attrs.get('content', ''))
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        for key in ['href', 'src']:
            if key in attrs:
                self.links.append(attrs[key])

    def handle_endtag(self, tag):
        if tag in {'script', 'style'}:
            self.excluded_depth = max(0, self.excluded_depth - 1)

    def handle_data(self, data):
        if not self.excluded_depth:
            self.visible_text.append(data)

    def text(self):
        return ' '.join(self.visible_text)


class ObservedModelSiteTests(unittest.TestCase):
    def test_reviewed_original_archives(self):
        data = json.loads((ROOT / 'experiments/observed_rivers/provenance.json').read_text())
        self.assertEqual(data['data_status'], 'observed')
        self.assertEqual(data['model_status'], 'trained_observed_daily')
        self.assertEqual({s['river_id'] for s in data['sources']}, {'yellow_river', 'mississippi'})
        for source in data['sources']:
            self.assertIs(source['redistribution_allowed'], True)
            self.assertEqual(hashlib.sha256((ROOT / source['raw_file']).read_bytes()).hexdigest(), source['raw_sha256'])

    def test_observed_results_and_download_paths(self):
        results = json.loads((ROOT / 'experiments/observed_rivers/website_data.json').read_text())
        self.assertEqual(results['model_status'], 'trained_observed_daily')
        self.assertEqual(set(results['rivers']), {'yellow_river', 'mississippi'})
        self.assertEqual(results['protocol']['seeds'], [42, 43, 44])
        for river in results['rivers'].values():
            self.assertEqual(len(river['observations']), 2557)
            self.assertEqual(len(river['backtest']), 384)
            self.assertEqual(len(river['forecast']), 6)
            self.assertEqual(river['forecast'][0]['date'], '2022-01-01')
            self.assertEqual(river['forecast'][-1]['date'], '2022-01-06')
        for artifact in results['artifacts']:
            path = (ROOT / artifact['path']).resolve()
            self.assertTrue(path.is_relative_to(ROOT / 'experiments/observed_rivers'))
            self.assertTrue(path.is_file(), artifact['path'])

    def test_no_withdrawn_results_or_synthetic_asset(self):
        source = ''.join((ROOT / 'website' / f).read_text() for f in ['index.html', 'app.js'])
        for token in ['assets/river_lab_data.js', 'window.RIVER_LAB_DATA', 'experiments/multi_river/', '重新认识河流']:
            self.assertNotIn(token, source)
        self.assertIn('assets/observed_river_data.js', source)

    def test_comparison_is_editorial_not_model_transfer(self):
        html = (ROOT / 'website/index.html').read_text()
        app = (ROOT / 'website/app.js').read_text()
        for removed in ['源河的学习，是否帮助目标河', 'transfer-results', 'model-profile', 'metric-rows']:
            self.assertNotIn(removed, html + app)
        self.assertNotIn('data.transfer', app)
        heading_expectations = {
            'study-hydrology': r'\b(?:basin|watershed)\b.*\bhydrology\b',
            'study-sediment': r'\bsediment\b',
            'study-dams': r'\bdams\b.*\bhydropower\b',
            'study-flood': r'\bflood\b',
            'study-ecology': r'\bwater\b.*\becology\b',
        }
        for anchor, expected in heading_expectations.items():
            heading = re.search(r'<h3\b[^>]*\bid="' + anchor + r'"[^>]*>(.*?)</h3>', html, re.S)
            self.assertIsNotNone(heading, anchor)
            self.assertRegex(heading.group(1).lower(), expected)

    def test_deeper_comparison_and_humanities(self):
        html = (ROOT / 'website/index.html').read_text()
        doc = (ROOT / 'docs/river_comparative_study.md').read_text()
        self.assertEqual(html.count('class="study-card"'), 7)
        subjects = [
            r'\blanzhou\b.{0,40}\bwaterwheels?\b',
            r'\bhua[’\']?er\b', r'\b(?:the )?yellow river cantata\b',
            r'\bbdote\b', r'\bdakota\b', r'\bjim\b', r'\bdelta blues\b',
        ]
        for subject in subjects:
            self.assertRegex(html.lower(), subject)
            self.assertRegex(doc.lower(), subject)
        articles = re.findall(r'<article class="study-card".*?</article>', html, re.S)
        for article in articles:
            self.assertGreaterEqual(article.count('<p>'), 4)
            self.assertGreater(len(re.sub(r'<[^>]+>', '', article)), 400)
        parser = LinkParser()
        parser.feed(html)
        self.assertTrue(STUDY_ANCHORS.issubset(parser.ids))
        for anchor in STUDY_ANCHORS:
            self.assertIn('#' + anchor, parser.links)
        self.assertTrue(any('comparison' in label.lower() or 'study' in label.lower()
                            for label in parser.accessible_text))
        citations = re.findall(r'^\d+\. \[.*?\]\((https://[^)]+)\)', doc, re.M)
        self.assertEqual(len(citations), 20)
        self.assertTrue(set(citations).issubset(set(parser.links)))
        model_section = html.split('<section id="prediction"', 1)[1].split('</section>', 1)[0]
        self.assertEqual(model_section.count('class="model-card"'), 2)
        model_text = LinkParser()
        model_text.feed(model_section)
        self.assertLessEqual(len(re.findall(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*", model_text.text())), 120)

    def test_presentation_and_chart_are_fully_english(self):
        for relative in ['website/index.html', 'cloudflare/404.html']:
            with self.subTest(page=relative):
                parser = LinkParser()
                parser.feed((ROOT / relative).read_text(encoding='utf-8'))
                self.assertEqual(parser.language, 'en')
                self.assertIsNone(CJK.search(parser.text()))
                self.assertIsNone(CJK.search(' '.join(parser.accessible_text)))
                if relative == 'website/index.html':
                    self.assertIn('Between Rivers', parser.text())
        app = (ROOT / 'website/app.js').read_text(encoding='utf-8')
        self.assertIsNone(CJK.search(app), 'Dynamic chart labels must be English too.')
        for label in ['Yellow River', 'Mississippi', 'Lijin', 'St. Louis']:
            self.assertIn(label, app)
        self.assertNotIn('"zh-CN"', app)

    def test_two_river_models_and_public_navigation_remain_available(self):
        parser = LinkParser()
        parser.feed((ROOT / 'website/index.html').read_text(encoding='utf-8'))
        for anchor in STUDY_ANCHORS | {'top', 'comparison', 'culture', 'prediction', 'sources'}:
            self.assertIn(anchor, parser.ids)
        for river in ['yellow_river', 'mississippi']:
            self.assertIn('../experiments/observed_rivers/models/' + river + '_gru_seed42.pt', parser.links)
            self.assertIn('../experiments/observed_rivers/data/' + river + '.csv', parser.links)
        for link in ['../src/models/observed_rivers.py', '../docs/observed_model_study.md',
                     '../LICENSE.observed-models', '../docs/river_comparative_study.md']:
            self.assertIn(link, parser.links)

    def test_current_page_local_links(self):
        page = ROOT / 'website/index.html'
        parser = LinkParser()
        parser.feed(page.read_text())
        for value in parser.links:
            parts = urlsplit(value)
            if parts.scheme or parts.netloc:
                continue
            if not parts.path:
                if parts.fragment:
                    self.assertIn(parts.fragment, parser.ids)
                continue
            path = (page.parent / unquote(parts.path)).resolve()
            self.assertTrue(path.is_relative_to(ROOT), value)
            self.assertTrue(path.is_file(), value)

    def test_portable_export_contains_only_reviewed_files(self):
        with tempfile.TemporaryDirectory(prefix='river-source-site-') as directory:
            bundle = Path(directory) / 'site'
            subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/export_river_site.py'), '--output', str(bundle)], check=True, capture_output=True)
            manifest = json.loads((bundle / 'publication_manifest.json').read_text())
            self.assertEqual(manifest['model_status'], 'trained_observed_daily')
            self.assertTrue(any(name.endswith('.pt') for name in manifest['files']))
            self.assertTrue(any(name.endswith('.csv') for name in manifest['files']))
            for relative, entry in manifest['files'].items():
                self.assertNotIn('multi_river', relative)
                file = bundle / relative
                self.assertEqual(file.stat().st_size, entry['bytes'])
                self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(), entry['sha256'])
            parser = LinkParser()
            parser.feed((bundle / 'website/index.html').read_text())
            for value in parser.links:
                parts = urlsplit(value)
                if not parts.scheme and not parts.netloc and parts.path:
                    self.assertTrue((bundle / 'website' / unquote(parts.path)).resolve().is_file(), value)
            failed = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/export_river_site.py'), '--output', str(bundle)], capture_output=True)
            self.assertNotEqual(failed.returncode, 0)


if __name__ == '__main__':
    unittest.main()
