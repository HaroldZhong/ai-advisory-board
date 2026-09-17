"""Offline catalog checks: discovery, refresh, isolation, and money/selection boundaries."""
import asyncio
import httpx
import pytest
from backend import config, openrouter_client as catalog


@pytest.fixture(autouse=True)
def isolated_catalog(monkeypatch):
    catalog.clear_cache()
    monkeypatch.setenv('OPENROUTER_API_KEY', 'catalog-test-key')
    monkeypatch.setattr(catalog, 'provider_is_openrouter', lambda: True)
    yield
    catalog.clear_cache()


def transport(monkeypatch, handler):
    monkeypatch.setattr(catalog, '_catalog_transport', httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_discovery_updates_prices_adds_models_and_marks_removed(monkeypatch):
    rows = [{'id': 'new/reasoner', 'supported_parameters': ['reasoning'],
             'pricing': {'prompt': '0', 'completion': '0'}, 'context_length': 32000}]
    def serve(request):
        assert request.headers['authorization'] == 'Bearer catalog-test-key'
        return httpx.Response(200, json={'data': [] if request.url.path.endswith('/zdr') else rows})
    transport(monkeypatch, serve)
    monkeypatch.setattr(config, 'CURATED_MODELS', [{'id': 'removed/model', 'type': 'both'}])
    models = await catalog.get_enriched_models(config.CURATED_MODELS)
    assert [(m['id'], m['available']) for m in models] == [('removed/model', False), ('new/reasoner', True)]
    from backend import main, openrouter, budget_router
    main.validate_model_selection([], 'new/reasoner')
    with pytest.raises(main.HTTPException):
        main.validate_model_selection([], 'removed/model')
    assert openrouter.resolve_model_reasoning('new/reasoner', 'low', capability_records={}) == ({'effort': 'low'}, None)
    assert budget_router._model_call_cost('new/reasoner', 1000000, 1000000) == 0
    rows[0]['pricing'] = {'prompt': '0.000003', 'completion': '0.000008'}
    catalog._cache['last_attempt'] = 0
    await catalog.get_openrouter_models_cached(force=True)
    assert main.calculate_cost({'prompt_tokens': 1000000, 'completion_tokens': 1000000}, 'new/reasoner') == 11
    assert main.calculate_cost({'cost': .01}, 'new/reasoner') == .01
    assert budget_router._model_call_cost('new/reasoner', 1000000, 1000000) == 11


@pytest.mark.asyncio
async def test_failure_preserves_catalog_and_empty_success_is_authoritative(monkeypatch):
    response = {'data': [{'id': 'a'}]}
    def serve(request):
        return httpx.Response(200, json={'data': []} if request.url.path.endswith('/zdr') else response)
    transport(monkeypatch, serve)
    assert 'a' in await catalog.get_openrouter_models_cached()
    response = {'error': 'not a models catalog'}
    catalog._cache['last_attempt'] = 0
    assert 'a' in await catalog.get_openrouter_models_cached(force=True)
    assert catalog.catalog_status()['stale'] and catalog.catalog_status()['error']
    response = {'data': []}
    catalog._cache['last_attempt'] = 0
    assert await catalog.get_openrouter_models_cached(force=True) == {}
    assert not catalog.catalog_status()['stale']


@pytest.mark.asyncio
async def test_zdr_partial_failure_is_visible_and_retries_without_waiting_for_ttl(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(catalog.time, 'time', lambda: now[0])
    zdr_ok = True
    requests = []
    def serve(request):
        requests.append(request.url.path)
        if request.url.path.endswith('/zdr'):
            return httpx.Response(200, json={'data': [{'model_id': 'model/a'}]}) if zdr_ok else httpx.Response(503)
        return httpx.Response(200, json={'data': [{'id': 'model/a'}]})
    transport(monkeypatch, serve)
    monkeypatch.setattr(config, 'CURATED_MODELS', [{'id': 'model/a', 'supports_zdr': True}])
    from backend import main
    await catalog.get_openrouter_models_cached()
    main.ensure_zdr_compatible_models('model/a', ['model/a'])
    zdr_ok = False
    now[0] += 30
    rows = await catalog.get_openrouter_models_cached(force=True)
    assert rows['model/a']['supports_zdr'] is None
    assert catalog.catalog_status()['stale']
    assert 'ZDR availability could not be verified' in catalog.catalog_status()['error']
    assert catalog.catalog_status()['next_refresh_at'] == 1060
    assert catalog.catalog_status()['refresh_cooldown_seconds'] == 30
    main.validate_model_selection([], 'model/a')  # Ordinary model selection still works.
    with pytest.raises(main.HTTPException):
        main.ensure_zdr_compatible_models('model/a', ['model/a'])
    zdr_ok = True
    now[0] += 29
    await catalog.get_openrouter_models_cached(force=True)
    assert len(requests) == 4  # Even manual retries respect the advertised cooldown.
    now[0] += 1
    await catalog.get_openrouter_models_cached()  # Normal polling recovers before the one-hour TTL.
    assert len(requests) == 6
    assert not catalog.catalog_status()['stale']
    assert catalog.catalog_status()['error'] is None
    main.ensure_zdr_compatible_models('model/a', ['model/a'])

    # The first fetch failing ZDR must not restore the curated True either.
    catalog.clear_cache()
    zdr_ok = False
    models = await catalog.get_enriched_models(config.CURATED_MODELS)
    assert models[0]['supports_zdr'] is None and catalog.catalog_status()['stale']


@pytest.mark.asyncio
async def test_single_flight_cooldown_and_generation_change(monkeypatch):
    calls = 0
    entered, release = asyncio.Event(), asyncio.Event()
    async def fetch():
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return [{'id': 'old-account'}]
    async def zdr(): return set()
    monkeypatch.setattr(catalog, 'fetch_openrouter_models', fetch)
    monkeypatch.setattr(catalog, 'fetch_openrouter_zdr_model_ids', zdr)
    a = asyncio.create_task(catalog.get_openrouter_models_cached())
    await entered.wait()
    b = asyncio.create_task(catalog.get_openrouter_models_cached(force=True))
    await asyncio.sleep(0)
    release.set()
    assert await a == await b
    await catalog.get_openrouter_models_cached(force=True)
    assert calls == 1
    # Credential replacement invalidates even a fresh cache and an in-flight generation.
    monkeypatch.setenv('OPENROUTER_API_KEY', 'second-key')
    assert catalog.get_model_metadata('old-account') is None
    entered.clear(); release.clear()
    a = asyncio.create_task(catalog.get_openrouter_models_cached())
    await entered.wait()
    catalog.clear_cache()
    release.set()
    assert await a is None
    assert catalog.get_model_metadata('old-account') is None


@pytest.mark.asyncio
async def test_pagination_completeness_and_no_redirect_credentials(monkeypatch):
    requests = []
    def serve(request):
        requests.append(request)
        if request.url.params.get('after') == 'a':
            return httpx.Response(200, json={'data': [{'id': 'b'}], 'has_more': False, 'total': 2})
        return httpx.Response(200, json={'data': [{'id': 'a'}], 'has_more': True, 'last_id': 'a'})
    transport(monkeypatch, serve)
    assert [row['id'] for row in await catalog.fetch_openrouter_models()] == ['a', 'b']
    assert len(requests) == 2
    requests.clear()
    def redirect(request):
        requests.append(request)
        return httpx.Response(302, headers={'Location': 'https://untrusted.example/models'})
    transport(monkeypatch, redirect)
    assert await catalog.fetch_openrouter_models() is None
    assert len(requests) == 1
    for body in ({'data': [], 'total': 1}, {'data': [{'id': 'a'}], 'next_cursor': 'x'},
                 {'data': [{'id': 'a'}, {'id': 'a'}]}, {'data': [None]}):
        transport(monkeypatch, lambda request: httpx.Response(200, json=body))
        assert await catalog.fetch_openrouter_models() is None


@pytest.mark.asyncio
async def test_provider_count_and_links_cannot_publish_an_incomplete_catalog(monkeypatch):
    response = {'data': [{'id': 'kept'}], 'total_count': 1, 'links': {'next': None}}
    requests = []
    def serve(request):
        requests.append(request)
        assert str(request.url) in (catalog.OPENROUTER_MODELS_URL, catalog.OPENROUTER_ZDR_ENDPOINTS_URL)
        return httpx.Response(200, json={'data': []} if request.url.path.endswith('/zdr') else response)
    transport(monkeypatch, serve)
    assert set(await catalog.get_openrouter_models_cached()) == {'kept'}
    for fields in ({'total_count': 1}, {'total_count': True}, {'total_count': '0'},
                   {'total_count': None}, {'total_count': -1},
                   {'total_count': 0, 'total': 1},
                   {'links': {'next': 'https://untrusted.example/models'}},
                   {'links': {'next': ''}}, {'links': []}, {'links': None}):
        response = {'data': [], **fields}
        requests.clear()
        catalog._cache['last_attempt'] = 0
        assert set(await catalog.get_openrouter_models_cached(force=True)) == {'kept'}
        assert catalog.catalog_status()['stale'] and catalog.catalog_status()['error']
        assert len(requests) == 1  # Never follow a provider link or publish partial rows.
    response = {'data': [], 'total_count': 0, 'links': {'next': None}}
    catalog._cache['last_attempt'] = 0
    assert await catalog.get_openrouter_models_cached(force=True) == {}
    assert not catalog.catalog_status()['stale']

    # A count on an earlier page still binds the final result and cannot change.
    for last_page in ({'data': []}, {'data': [{'id': 'b'}], 'total_count': 3},
                      {'data': [{'id': 'b'}]}):
        def paginated(request):
            return httpx.Response(200, json=last_page if request.url.params else {
                'data': [{'id': 'a'}], 'has_more': True, 'last_id': 'a', 'total_count': 2})
        transport(monkeypatch, paginated)
        rows = await catalog.fetch_openrouter_models()
        assert (rows is not None) == (last_page == {'data': [{'id': 'b'}]})


def test_price_unknown_is_distinct_from_zero_and_does_not_cross_providers(monkeypatch):
    for invalid in (None, '', ' ', True, -1, 'NaN', 'Infinity', '1e308'):
        parsed = catalog.parse_openrouter_model({'id': 'free-name:free', 'pricing': {'prompt': invalid, 'completion': '0'}})
        assert parsed['pricing_source'] == 'unknown'
        assert parsed['pricing']['input'] is None
    parsed = catalog.parse_openrouter_model({'id': 'free', 'pricing': {'prompt': '0', 'completion': '0'}})
    assert parsed['pricing_source'] == 'provider' and parsed['pricing'] == {'input': 0, 'output': 0}
    monkeypatch.setattr(catalog, 'provider_is_openrouter', lambda: False)
    parsed = catalog.parse_openrouter_model({'id': 'generic', 'pricing': {'prompt': '1', 'completion': '2'}})
    assert parsed['pricing_source'] == 'unknown'
    curated = [{'id': 'same-id', 'pricing': {'input': 1, 'output': 2}, 'supports_zdr': True}]
    row = catalog._merge_models(curated, None)[0]
    assert row['supports_zdr'] is False and row['pricing']['input'] is None


@pytest.mark.asyncio
async def test_generic_authenticated_discovery_skips_zdr_and_nontext_is_not_selectable(monkeypatch):
    monkeypatch.setattr(catalog, 'provider_is_openrouter', lambda: False)
    def serve(request):
        assert request.headers['authorization'] == 'Bearer catalog-test-key'
        assert not request.url.path.endswith('/zdr')
        return httpx.Response(200, json={'data': [{'id': 'local/new'}, {'id': 'image/only', 'architecture': {'output_modalities': ['image']}}]})
    transport(monkeypatch, serve)
    models = await catalog.get_enriched_models([])
    assert models[0]['type'] == 'both' and models[0]['supports_zdr'] is False
    assert models[1]['type'] == 'other'


@pytest.mark.asyncio
async def test_automatic_ttl_refresh_and_size_limit(monkeypatch):
    count = 0
    def serve(request):
        nonlocal count
        count += 1
        return httpx.Response(200, json={'data': []})
    transport(monkeypatch, serve)
    await catalog.get_openrouter_models_cached()
    assert count == 2
    catalog._cache.update(last_fetched=0, last_attempt=0)
    await catalog.get_openrouter_models_cached()
    assert count == 4
    monkeypatch.setattr(catalog, 'MAX_CATALOG_BYTES', 2)
    assert await catalog.fetch_openrouter_models() is None
