#!/usr/bin/env python3
"""Build the reviewed river website for Cloudflare Pages using only the stdlib."""
import argparse
import hashlib
import html
import json
import re
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from export_river_site import ROOT, reviewed_inputs

MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_FILES = 20_000
MANIFEST = 'publication_manifest.json'
TARGET = 'cloudflare-pages'
URL_ATTRIBUTE = re.compile(r'\b(href|src)(\s*=\s*)([\"\'])(.*?)\3', re.I | re.S)


class RootPageLinks(HTMLParser):
    """Change URL attributes without reserializing the SVG or page markup."""

    def __init__(self, source):
        super().__init__(convert_charrefs=False)
        self.source = source
        self.offsets = [0]
        for line in source.splitlines(keepends=True):
            self.offsets.append(self.offsets[-1] + len(line))
        self.edits = []

    def handle_starttag(self, tag, attrs):
        raw = self.get_starttag_text()

        def replace(match):
            value = html.unescape(match.group(4))
            parts = urlsplit(value)
            if not value or value.startswith('#') or parts.scheme or parts.netloc:
                return match.group(0)
            resolved = urljoin('/website/', value)
            return match.group(1) + match.group(2) + match.group(3) + html.escape(resolved, quote=True) + match.group(3)

        rewritten = URL_ATTRIBUTE.sub(replace, raw)
        if rewritten != raw:
            line, column = self.getpos()
            self.edits.append((self.offsets[line - 1] + column, len(raw), rewritten))

    handle_startendtag = handle_starttag

    def render(self):
        self.feed(self.source)
        self.close()
        result = self.source
        for offset, length, replacement in reversed(self.edits):
            result = result[:offset] + replacement + result[offset + length:]
        return result


def file_entry(path):
    return {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def validate_destination(requested, required):
    if requested.is_symlink():
        raise ValueError('Output must not be a symbolic link')
    destination = requested.resolve()
    if destination == ROOT or destination in ROOT.parents:
        raise ValueError('Output must not be the project or an ancestor')
    for relative in required:
        protected = ROOT / relative.parts[0]
        if destination == protected or (protected.is_dir() and destination.is_relative_to(protected)):
            raise ValueError('Output overlaps publication source files: ' + str(protected))
    if destination.exists():
        validate_managed_output(destination)
    return destination


def validate_managed_output(destination):
    """Never replace an unrelated, changed, or symlink-containing directory."""
    if not destination.is_dir():
        raise ValueError('Output already exists and is not a managed Pages directory')
    manifest_path = destination / MANIFEST
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError('Existing output is not a managed Pages build; choose a new --output')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('build_target') != TARGET or not isinstance(manifest.get('files'), dict):
        raise ValueError('Existing output is not a managed Pages build; choose a new --output')
    actual = set()
    for path in destination.rglob('*'):
        if path.is_symlink():
            raise ValueError('Existing output contains a symbolic link; it was not changed')
        if path.is_file():
            actual.add(path.relative_to(destination).as_posix())
    if actual != set(manifest['files']) | {MANIFEST}:
        raise ValueError('Existing output contains missing or extra files; it was not changed')
    for relative, expected in manifest['files'].items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or not (destination / path).is_file():
            raise ValueError('Invalid existing build manifest')
        if file_entry(destination / path) != expected:
            raise ValueError('Existing output was modified; choose a new --output to preserve it')


def build_payload(destination, provenance, required):
    destination.mkdir()
    for relative in required:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    source = (ROOT / 'website/index.html').read_text(encoding='utf-8')
    (destination / 'index.html').write_text(RootPageLinks(source).render(), encoding='utf-8')
    for name in ['404.html', '_headers']:
        shutil.copy2(ROOT / 'cloudflare' / name, destination / name)
    manifest = {
        'build_target': TARGET,
        'data_status': provenance['data_status'],
        'model_status': provenance['model_status'],
        'snapshot_date': provenance['snapshot_date'],
        'excludes': 'Legacy synthetic results, database, environments, credentials, backups and training dependencies',
        'files': {},
    }
    for path in sorted(destination.rglob('*')):
        if path.is_file():
            if path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError('Asset exceeds the Pages 25 MiB limit: ' + str(path))
            manifest['files'][path.relative_to(destination).as_posix()] = file_entry(path)
    if len(manifest['files']) + 1 > MAX_FILES:
        raise ValueError('Build exceeds the Pages Free-plan 20,000-file limit')
    manifest_path = destination / MANIFEST
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if manifest_path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError('Manifest exceeds the Pages 25 MiB limit')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist')
    parser.add_argument('--zip', dest='zip_path', type=Path, help='Optional fresh Dashboard-upload ZIP, with no outer directory')
    args = parser.parse_args()
    try:
        provenance, required = reviewed_inputs()
        destination = validate_destination(args.output, required)
        archive = None
        if args.zip_path:
            if args.zip_path.is_symlink() or args.zip_path.exists():
                raise ValueError('ZIP already exists; choose a new --zip path to preserve it')
            archive = args.zip_path.resolve()
            if archive == destination or archive.is_relative_to(destination):
                raise ValueError('ZIP must be outside the output directory')
            for relative in required:
                protected = ROOT / relative.parts[0]
                if archive == protected or (protected.is_dir() and archive.is_relative_to(protected)):
                    raise ValueError('ZIP overlaps publication source files')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.river-pages-build-', dir=destination.parent) as scratch:
            payload = Path(scratch) / 'site'
            manifest = build_payload(payload, provenance, required)
            zip_payload = None
            if archive:
                zip_payload = Path(scratch) / 'upload.zip'
                with zipfile.ZipFile(zip_payload, 'w', zipfile.ZIP_DEFLATED) as package:
                    for path in sorted(payload.rglob('*')):
                        if path.is_file():
                            package.write(path, path.relative_to(payload).as_posix())
            backup = None
            if destination.exists():
                validate_managed_output(destination)
                backup_root = destination.parent / '.pages-build-backups'
                if backup_root.is_symlink():
                    raise ValueError('Build backup directory must not be a symbolic link')
                backup_root.mkdir(exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                backup = backup_root / (destination.name + '-' + stamp + '-' + uuid.uuid4().hex[:8])
                destination.rename(backup)
            try:
                payload.rename(destination)
            except OSError:
                if backup:
                    backup.rename(destination)
                raise
            if archive:
                archive.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive creation also protects against a file appearing during the build.
                with archive.open('xb') as output, zip_payload.open('rb') as source:
                    shutil.copyfileobj(source, output)
        print(f'Cloudflare Pages build: {destination}\nFiles: {len(manifest["files"]) + 1}')
        if archive:
            print(f'Dashboard upload ZIP: {archive}')
        if backup:
            print(f'Previous generated build retained: {backup}')
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
