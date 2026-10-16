"""Adversarial checks for the publication guard; all values are synthetic."""
import importlib.util
from pathlib import Path
import tarfile
import io
import zipfile

import pytest

SPEC = importlib.util.spec_from_file_location('release_check', Path(__file__).resolve().parents[1] / 'tools/check_release.py')
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


@pytest.mark.parametrize('text', [
    'sk-' + 'x' * 30,
    'api_key' + ' = "' + 'synthetic-value' + '"',
    'Authorization: ' + 'Bearer ' + 'x' * 20,
    '/' + 'Users' + '/example/project/file.py',
    'endpoint=' + '192.' + '168.1.1',
    'address=' + 'person' + '@' + 'private.invalid',
])
def test_sensitive_source_values_are_rejected_without_echoing_them(text):
    errors = []
    release.check_bytes('example.py', text.encode(), errors)
    assert errors
    assert all(text not in error for error in errors)


def test_ssh_clone_command_and_synthetic_example_contact_are_allowed():
    errors = []
    release.check_bytes('README.md', b'git clone git@github.com:owner/repository.git\nuser@example.com\n', errors)
    assert errors == []


@pytest.mark.parametrize('name', ['.env', 'credentials/record.json', 'model.bin', '../outside.py'])
def test_sensitive_or_unexpected_names_are_rejected(name):
    errors = []
    release.check_bytes(name, b'synthetic', errors)
    assert errors


def test_wheel_hidden_environment_file_is_detected(tmp_path):
    path = tmp_path / 'test.whl'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('package/.env', 'synthetic')
    errors = []
    release.check_distributions(tmp_path, errors)
    assert any('sensitive filename' in error for error in errors)


def test_source_archive_symlink_is_detected(tmp_path):
    path = tmp_path / 'test.tar.gz'
    with tarfile.open(path, 'w:gz') as archive:
        info = tarfile.TarInfo('package/link.py')
        info.type = tarfile.SYMTYPE
        info.linkname = 'outside.py'
        archive.addfile(info)
        info = tarfile.TarInfo('package/example.py')
        data = b'print("synthetic")\n'
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    errors = []
    release.check_distributions(tmp_path, errors)
    assert any('archive link' in error for error in errors)
