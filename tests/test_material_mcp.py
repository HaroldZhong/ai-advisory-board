"""A real SDK client discovers/calls a separate local stdio process. No LLM or network."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

mcp = pytest.importorskip('mcp', reason='Install the optional mcp extra to validate stdio')
from mcp import Client, StdioServerParameters


@pytest.mark.asyncio
async def test_stdio_discovery_call_and_revalidation(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    root = tmp_path / 'demo'
    env = {**os.environ, 'AAB_DATA_DIR': str(root), 'OPENROUTER_API_KEY': ''}
    subprocess.run([sys.executable, 'scripts/create_research_demo.py', str(root)], cwd=repo, env=env, check=True, capture_output=True)
    manifest = json.loads((root / 'demo-materials.json').read_text())
    first = manifest['sources'][0]
    parameters = StdioServerParameters(command=sys.executable, args=['-m', 'backend.material_mcp', '--demo-root', str(root)], cwd=repo, env=env)
    async with Client(parameters, read_timeout_seconds=10) as client:
        tools = await client.list_tools()
        assert tools.next_cursor is None
        assert [tool.name for tool in tools.tools] == ['aab_read_material']
        assert tools.tools[0].annotations.read_only_hint
        resources = await client.list_resources()
        assert resources.next_cursor is None
        assert [str(resource.uri) for resource in resources.resources] == ['aab://materials']
        catalog = await client.read_resource('aab://materials')
        assert json.loads(catalog.contents[0].text) == manifest
        args = {key: first[key] for key in ('source_id', 'version_id')}
        valid = await client.call_tool('aab_read_material', args)
        assert not valid.is_error
        assert valid.structured_content['source_id'] == first['source_id']
        assert '合成演示' in valid.structured_content['chunks'][0]['text']
        chunk = await client.call_tool('aab_read_material', {**args, 'chunk_id': 2})
        assert not chunk.is_error and [part['ordinal'] for part in chunk.structured_content['chunks']] == [2]
        for invalid in [{'chunk_id': 0}, {'chunk_id': True}, {'chunk_id': 999}, {'source_id': '../private'}, {'source_id': 'not-approved'}, {'version_id': '0' * 64}]:
            result = await client.call_tool('aab_read_material', {**args, **invalid})
            assert result.is_error, invalid
        # A valid pinned version cannot read content re-extracted under the same ID.
        text_path = root / 'data/conversations/attachments/text' / f"{first['source_id']}.txt"
        text_path.rename(text_path.with_suffix('.original'))
        text_path.write_text('Changed after server start', encoding='utf-8')
        assert (await client.call_tool('aab_read_material', args)).is_error
        # A deleted source remains refused even though the startup catalog still names it.
        meta_path = root / 'data/conversations/attachments/meta' / f"{first['source_id']}.json"
        meta_path.rename(meta_path.with_suffix('.removed'))
        assert (await client.call_tool('aab_read_material', args)).is_error
