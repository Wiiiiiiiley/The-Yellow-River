"""Dependency-free regression checks for the Cloudflare Pages static build."""
import hashlib
import fnmatch
import json
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / 'scripts/build_cloudflare_pages.py'


class PageLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()
        self.stylesheets = []
        self.scripts = []
        self.refresh = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if attributes.get('id'):
            self.ids.add(attributes['id'])
        for name in ['href', 'src']:
            if name in attributes:
                self.links.append(attributes[name])
        if tag == 'link' and attributes.get('rel') == 'stylesheet':
            self.stylesheets.append(attributes.get('href'))
        if tag == 'script' and 'src' in attributes:
            self.scripts.append(attributes['src'])
        if tag == 'meta' and attributes.get('http-equiv', '').lower() == 'refresh':
            self.refresh = True


def snapshot(directory):
    """Capture user files without following symlinks, for rejection checks."""
    entries = {}
    for path in sorted(directory.rglob('*')):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            entries[relative] = ('symlink', str(path.readlink()))
        elif path.is_file():
            entries[relative] = ('file', hashlib.sha256(path.read_bytes()).hexdigest())
        elif path.is_dir():
            entries[relative] = ('directory',)
    return entries


class CloudflarePagesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='river-pages-tests-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def command(self, output, *arguments, script=BUILD_SCRIPT):
        return subprocess.run(
            [sys.executable, '-B', str(script), '--output', str(output),
             *map(str, arguments)],
            check=False, capture_output=True, text=True,
        )

    def build(self, output=None, *arguments, script=BUILD_SCRIPT):
        destination = output or self.directory / 'dist'
        result = self.command(destination, *arguments, script=script)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return destination

    def parse(self, page):
        parser = PageLinks()
        parser.feed(page.read_text(encoding='utf-8'))
        return parser

    def assert_local_links(self, output, page):
        parser = self.parse(page)
        for value in parser.links:
            with self.subTest(page=str(page.relative_to(output)), link=value):
                parts = urlsplit(value)
                if parts.scheme or parts.netloc:
                    continue
                if not parts.path:
                    if parts.fragment:
                        self.assertIn(unquote(parts.fragment), parser.ids)
                    continue
                relative = unquote(parts.path)
                target = (output / relative.lstrip('/')) if relative.startswith('/') else (page.parent / relative)
                target = target.resolve()
                self.assertTrue(target.is_relative_to(output.resolve()), value)
                if target.is_dir():
                    target = target / 'index.html'
                self.assertTrue(target.is_file(), value)
                if parts.fragment and target.suffix == '.html':
                    self.assertIn(unquote(parts.fragment), self.parse(target).ids)

    def assert_manifest(self, output):
        manifest_path = output / 'publication_manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        self.assertEqual(manifest['build_target'], 'cloudflare-pages')
        self.assertNotIn('publication_manifest.json', manifest['files'])
        actual_files = {
            path.relative_to(output).as_posix()
            for path in output.rglob('*') if path.is_file()
        }
        self.assertEqual(actual_files, set(manifest['files']) | {'publication_manifest.json'})
        for relative, entry in manifest['files'].items():
            with self.subTest(artifact=relative):
                path = output / relative
                self.assertTrue(path.is_file())
                self.assertFalse(path.is_symlink())
                self.assertEqual(path.stat().st_size, entry['bytes'])
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry['sha256'])
                self.assertLessEqual(path.stat().st_size, 25 * 1024 * 1024)
                self.assertNotIn('multi_river', relative)
                self.assertNotIn('river_lab_data.js', relative)
        return manifest

    def test_root_and_legacy_pages_keep_assets_downloads_and_anchors(self):
        output = self.build()
        root = self.parse(output / 'index.html')
        legacy = self.parse(output / 'website/index.html')
        self.assertFalse(root.refresh)
        self.assertFalse(legacy.refresh)
        self.assertEqual([urlsplit(url).path for url in root.stylesheets], ['/website/styles.css'])
        self.assertEqual([urlsplit(url).path for url in root.scripts], ['/website/assets/observed_river_data.js', '/website/app.js'])
        self.assertEqual([urlsplit(url).path for url in legacy.stylesheets], ['styles.css'])
        self.assertEqual([urlsplit(url).path for url in legacy.scripts], ['assets/observed_river_data.js', 'app.js'])
        self.assertTrue(urlsplit(root.stylesheets[0]).query)
        self.assertTrue(urlsplit(root.scripts[1]).query)
        self.assertEqual(urlsplit(root.stylesheets[0]).query, urlsplit(legacy.stylesheets[0]).query)
        self.assertEqual(urlsplit(root.scripts[1]).query, urlsplit(legacy.scripts[1]).query)
        for anchor in ['#comparison', '#prediction', '#sources', '#study-culture']:
            self.assertIn(anchor, root.links)
            self.assertIn(anchor, legacy.links)
        self.assertIn('/experiments/observed_rivers/models/yellow_river_gru_seed42.pt', root.links)
        self.assertIn('/experiments/observed_rivers/models/mississippi_gru_seed42.pt', root.links)
        self.assertIn('/docs/river_comparative_study.md', root.links)
        self.assert_local_links(output, output / 'index.html')
        self.assert_local_links(output, output / 'website/index.html')

    def test_404_headers_and_manifest_are_in_build_root(self):
        output = self.build()
        self.assertTrue((output / '404.html').is_file())
        self.assertTrue((output / '_headers').is_file())
        self.assert_local_links(output, output / '404.html')
        headers = (output / '_headers').read_text(encoding='utf-8')
        self.assertIn('X-Content-Type-Options: nosniff', headers)
        self.assertIn('Content-Type:', headers)
        self.assertNotIn('immutable', headers)
        self.assertFalse((output / '_worker.js').exists())
        self.assertFalse((output / 'functions').exists())
        self.assert_manifest(output)

    def test_model_and_csv_download_headers_are_complete_and_unambiguous(self):
        output = self.build()
        rules = []
        current_headers = None
        for line in (output / '_headers').read_text(encoding='utf-8').splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            if line.startswith((' ', '\t')):
                self.assertIsNotNone(current_headers, 'A header needs a preceding URL rule')
                name, value = line.strip().split(':', 1)
                current_headers.append((name.lower(), value.strip()))
            else:
                current_headers = []
                rules.append((line.strip(), current_headers))
        global_headers = [headers for pattern, headers in rules if pattern == '/*']
        self.assertEqual(len(global_headers), 1)
        self.assertIn(('x-content-type-options', 'nosniff'), global_headers[0])
        self.assertFalse(any(name == 'content-type' for name, _ in global_headers[0]))
        artifacts = output / 'experiments/observed_rivers'
        models = list(artifacts.rglob('*.pt'))
        csv_files = list(artifacts.rglob('*.csv'))
        self.assertEqual(len(models), 24)
        self.assertTrue(csv_files)
        for path in models + csv_files:
            url = '/' + path.relative_to(output).as_posix()
            with self.subTest(download=url):
                matched = [
                    header for pattern, headers in rules
                    if fnmatch.fnmatchcase(url, pattern)
                    for header in headers
                ]
                content_types = [value for name, value in matched if name == 'content-type']
                disposition = [value for name, value in matched if name == 'content-disposition']
                expected = 'application/octet-stream' if path.suffix == '.pt' else 'text/csv; charset=utf-8'
                self.assertEqual(content_types, [expected], 'Missing or conflicting Content-Type rules')
                self.assertEqual(disposition, ['attachment'])

    def test_zip_has_upload_ready_root_and_matches_output(self):
        archive = self.directory / 'upload.zip'
        output = self.build(self.directory / 'dist', '--zip', archive)
        self.assert_manifest(output)
        with zipfile.ZipFile(archive) as bundle:
            files = {name for name in bundle.namelist() if not name.endswith('/')}
            expected = {path.relative_to(output).as_posix() for path in output.rglob('*') if path.is_file()}
            self.assertEqual(files, expected)
            for required in ['index.html', '404.html', '_headers', 'publication_manifest.json', 'website/index.html']:
                self.assertIn(required, files)
            self.assertFalse(any(name.startswith('dist/') for name in files))
            self.assertFalse(any(name.startswith('/') or '..' in Path(name).parts for name in files))
            for relative in files:
                self.assertEqual(bundle.read(relative), (output / relative).read_bytes())

    def test_repeat_build_preserves_previous_output_in_backup(self):
        output = self.build()
        previous = snapshot(output)
        self.build(output)
        self.assert_manifest(output)
        backups = output.parent / '.pages-build-backups'
        self.assertTrue(backups.is_dir())
        generations = list(backups.iterdir())
        self.assertEqual(len(generations), 1)
        self.assertEqual(snapshot(generations[0]), previous)
        self.build(output)
        self.assertEqual(len(list(backups.iterdir())), 2)
        self.assertEqual(snapshot(generations[0]), previous)

    def test_existing_non_managed_directory_is_not_overwritten(self):
        output = self.directory / 'dist'
        output.mkdir()
        (output / 'personal-note.txt').write_text('Keep this user file.', encoding='utf-8')
        before = snapshot(output)
        result = self.command(output)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(snapshot(output), before)
        self.assertFalse((output.parent / '.pages-build-backups').exists())

    def test_changed_managed_file_is_not_overwritten(self):
        output = self.build()
        (output / 'index.html').write_text('A manually edited page.', encoding='utf-8')
        before = snapshot(output)
        result = self.command(output)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(snapshot(output), before)
        self.assertFalse((output.parent / '.pages-build-backups').exists())

    def test_unknown_file_in_managed_output_is_not_removed(self):
        output = self.build()
        (output / 'personal-note.txt').write_text('An additional user file.', encoding='utf-8')
        before = snapshot(output)
        result = self.command(output)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(snapshot(output), before)
        self.assertFalse((output.parent / '.pages-build-backups').exists())

    def test_symlink_output_and_descendant_symlink_are_rejected(self):
        output = self.build(self.directory / 'managed')
        alias = self.directory / 'dist'
        try:
            alias.symlink_to(output, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(str(error))
        before = snapshot(output)
        result = self.command(alias)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(alias.is_symlink())
        self.assertEqual(snapshot(output), before)
        (output / 'linked-index.html').symlink_to(output / 'index.html')
        before = snapshot(output)
        result = self.command(output)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(snapshot(output), before)

    def test_zip_does_not_overwrite_user_file_or_live_inside_output(self):
        archive = self.directory / 'upload.zip'
        archive.write_bytes(b'A pre-existing archive must remain untouched.')
        original = archive.read_bytes()
        result = self.command(self.directory / 'dist', '--zip', archive)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(archive.read_bytes(), original)
        output = self.directory / 'other-dist'
        result = self.command(output, '--zip', output / 'upload.zip')
        self.assertNotEqual(result.returncode, 0)

    def test_source_trees_ancestors_and_descendants_are_rejected(self):
        # Use an isolated source fixture: a faulty safety guard must not touch
        # the user's actual project during this regression test.
        fixture = self.directory / 'source-project'
        fixture.mkdir()
        exporter = runpy.run_path(str(ROOT / 'scripts/export_river_site.py'))
        _, required = exporter['reviewed_inputs']()
        for relative in required:
            target = fixture / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        copied_script = fixture / 'scripts/build_cloudflare_pages.py'
        if not copied_script.exists():
            shutil.copy2(BUILD_SCRIPT, copied_script)
        template_root = ROOT / 'cloudflare'
        if template_root.exists() and not (fixture / 'cloudflare').exists():
            shutil.copytree(template_root, fixture / 'cloudflare')
        destinations = [fixture, self.directory]
        for tree in ['website', 'src', 'docs', 'scripts', 'tests', 'experiments', 'cloudflare']:
            destinations += [fixture / tree, fixture / tree / 'unexpected-build-output']
        for destination in destinations:
            with self.subTest(destination=str(destination)):
                before = snapshot(fixture)
                result = self.command(destination, script=copied_script)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(snapshot(fixture), before)
        # The source root itself is forbidden, but its purpose-built output
        # directory remains a valid destination.
        self.build(fixture / 'dist', script=copied_script)


if __name__ == '__main__':
    unittest.main()
