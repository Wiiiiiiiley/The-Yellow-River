"""Regression checks for the observed two-station model publication."""
import csv
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


class PageElement:
    """Small HTML tree for checking scoped, server-rendered publication content."""
    def __init__(self, tag, attrs=()):
        self.tag = tag
        self.attrs = dict(attrs)
        self.children = []

    def text(self):
        return ' '.join(child.text() if isinstance(child, PageElement) else child
                        for child in self.children)

    def find(self, tag=None, **attrs):
        found = []
        for child in self.children:
            if not isinstance(child, PageElement):
                continue
            if (tag is None or child.tag == tag) and all(
                    child.attrs.get(key) == value for key, value in attrs.items()):
                found.append(child)
            found.extend(child.find(tag, **attrs))
        return found

    def with_class(self, name):
        return [node for node in self.find() if name in node.attrs.get('class', '').split()]


class ElementParser(HTMLParser):
    VOID_TAGS = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
                 'link', 'meta', 'param', 'source', 'track', 'wbr'}

    def __init__(self, html):
        super().__init__()
        self.root = PageElement('document')
        self.stack = [self.root]
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        node = PageElement(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in self.VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].children.append(PageElement(tag, attrs))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


class ObservedModelSiteTests(unittest.TestCase):
    def forecast_csv(self, river):
        with (ROOT / f'experiments/observed_rivers/forecasts/{river}_seed42.csv').open(
                encoding='utf-8', newline='') as source:
            return list(csv.DictReader(source))

    def assert_preserved_forecast_artifacts(self, bundle, page):
        parser = LinkParser()
        parser.feed(page.read_text(encoding='utf-8'))
        for river in ['yellow_river', 'mississippi']:
            relative = f'experiments/observed_rivers/forecasts/{river}_seed42.csv'
            links = [value for value in parser.links if urlsplit(value).path.endswith(relative)]
            self.assertEqual(links, [], 'Forecast CSV buttons should not be shown.')
            self.assertEqual((bundle / relative).read_bytes(), (ROOT / relative).read_bytes())

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
        model_tree = ElementParser(model_section).root
        model_intro = model_tree.with_class('model-grid') + model_tree.with_class('model-links')
        self.assertEqual(len(model_intro), 2)
        model_text = ' '.join(node.text() for node in model_intro)
        self.assertLessEqual(len(re.findall(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*", model_text)), 120)

    def test_saved_seed42_forecasts_keep_historical_publication_context(self):
        results = json.loads((ROOT / 'experiments/observed_rivers/website_data.json').read_text())
        for river in ['yellow_river', 'mississippi']:
            rows = self.forecast_csv(river)
            self.assertEqual(len(rows), 6)
            asset_rows = results['rivers'][river]['forecast']
            for day, (row, asset_row) in enumerate(zip(rows, asset_rows), start=1):
                with self.subTest(river=river, day=day):
                    self.assertEqual(row['date'], f'2022-01-{day:02d}')
                    self.assertEqual(row['origin_date'], '2021-12-31')
                    self.assertEqual(row['lead_days'], str(day))
                    self.assertEqual(row['unit'], 'm³/s')
                    self.assertEqual(row['method'], 'gru')
                    self.assertEqual(row['seed'], '42')
                    self.assertEqual(row['observed'], '')
                    self.assertEqual(row['prediction_status'], 'model_prediction')
                    self.assertEqual(row['input_source_status'], 'observed')
                    self.assertEqual(row['evaluation'], 'recursive_historical_extension')
                    self.assertEqual(asset_row['date'], row['date'])
                    self.assertEqual(asset_row['origin_date'], row['origin_date'])
                    self.assertIsNone(asset_row['observed'])
                    self.assertEqual(float(row['prediction']), asset_row['predictions_by_seed']['42'])

    def test_2027_scenario_reuses_historical_forecasts_without_javascript(self):
        document = ElementParser((ROOT / 'website/index.html').read_text(encoding='utf-8')).root
        prediction = document.find('section', id='prediction')
        self.assertEqual(len(prediction), 1)
        panels = prediction[0].find('div', id='forecast-results')
        self.assertEqual(len(panels), 1)
        panel = panels[0]
        self.assertIn('forecast-panel', panel.attrs.get('class', '').split())
        text = ' '.join(panel.text().split())
        self.assertIn('2027 SCENARIO', text)
        self.assertRegex(text, r'1[–-]6 January 2027')
        self.assertIn('m³/s', text)
        self.assertIn('Illustrative daily mean discharge', text)
        self.assertEqual(panel.find(id='forecast-note'), [])
        self.assertEqual(panel.with_class('forecast-footer'), [])
        self.assertNotIn('2022 output CSV', text)
        self.assertNotIn('hidden', panel.attrs)
        tables = panel.with_class('forecast-table')
        self.assertEqual(len(tables), 1)
        table = tables[0]
        self.assertEqual(table.tag, 'table')
        self.assertTrue(table.find('caption') or table.attrs.get('aria-label')
                        or table.attrs.get('aria-labelledby'), 'The table needs an accessible name.')
        headers = table.find('th', scope='col')
        self.assertEqual(len(headers), 3)
        header_text = ' '.join(node.text() for node in headers)
        for label in ['Date', 'Yellow River', 'Lijin', 'Mississippi', 'St. Louis']:
            self.assertIn(label, header_text)
        caption_text = ' '.join(node.text() for node in table.find('caption'))
        self.assertIn('m³/s', caption_text)
        descriptions = table.attrs.get('aria-describedby', '').split()
        for identifier in descriptions:
            self.assertEqual(len(panel.find(id=identifier)), 1)
        bodies = table.find('tbody')
        self.assertEqual(len(bodies), 1)
        rows = bodies[0].find('tr')
        self.assertEqual(len(rows), 6)
        expected = {river: self.forecast_csv(river) for river in ['yellow_river', 'mississippi']}
        for index, row in enumerate(rows):
            date = expected['yellow_river'][index]['date']
            self.assertEqual(row.attrs.get('data-forecast-date'), date)
            scenario_date = f'2027-01-{index + 1:02d}'
            self.assertEqual(row.attrs.get('data-scenario-date'), scenario_date)
            times = row.find('time', datetime=scenario_date)
            self.assertEqual(len(times), 1)
            self.assertTrue(times[0].text().strip())
            for river in expected:
                cells = row.find('td', **{'data-river': river})
                self.assertEqual(len(cells), 1)
                self.assertEqual(cells[0].text().strip(), f"{float(expected[river][index]['prediction']):,.1f}")
        for river in expected:
            links = panel.find('a', href=f'../experiments/observed_rivers/forecasts/{river}_seed42.csv')
            self.assertEqual(links, [])

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
            self.assert_preserved_forecast_artifacts(bundle, bundle / 'website/index.html')
            for value in parser.links:
                parts = urlsplit(value)
                if not parts.scheme and not parts.netloc and parts.path:
                    self.assertTrue((bundle / 'website' / unquote(parts.path)).resolve().is_file(), value)
            failed = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/export_river_site.py'), '--output', str(bundle)], capture_output=True)
            self.assertNotEqual(failed.returncode, 0)

    def test_cloudflare_pages_exports_scenario_without_footer_or_csv_buttons(self):
        with tempfile.TemporaryDirectory(prefix='river-forecast-pages-') as directory:
            bundle = Path(directory) / 'dist'
            subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/build_cloudflare_pages.py'),
                            '--output', str(bundle)], check=True, capture_output=True)
            for relative in ['index.html', 'website/index.html']:
                page = bundle / relative
                document = ElementParser(page.read_text(encoding='utf-8')).root
                panels = document.find('div', id='forecast-results')
                self.assertEqual(len(panels), 1)
                self.assertEqual(len(panels[0].find('tbody')[0].find('tr')), 6)
                self.assertIn('2027 SCENARIO', panels[0].text())
                self.assertIn('Illustrative daily mean discharge', panels[0].text())
                self.assertEqual(panels[0].with_class('forecast-footer'), [])
                self.assertEqual(panels[0].find(id='forecast-note'), [])
                for day, row in enumerate(panels[0].find('tbody')[0].find('tr'), start=1):
                    self.assertEqual(row.attrs.get('data-scenario-date'), f'2027-01-{day:02d}')
                    self.assertEqual(row.attrs.get('data-forecast-date'), f'2022-01-{day:02d}')
                self.assert_preserved_forecast_artifacts(bundle, page)


if __name__ == '__main__':
    unittest.main()
