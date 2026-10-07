def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_index_renders_shell(client):
    response = client.get("/")
    assert response.status_code == 200

    body = response.text
    assert 'id="chat-log"' in body
    assert 'id="wheel-pane"' in body
    assert 'id="picked-list"' in body
    assert 'id="go-form"' in body
    assert "htmx.min.js" in body
    assert "wheel.js" in body
    assert "type a list or upload an image" in body

    # Interfaces: banner, canvas + wheel-data, go controls, chat form wiring.
    assert 'id="banner"' in body
    assert 'id="wheel"' in body
    assert 'data-entries="[]"' in body
    assert 'id="remove-picked"' in body
    assert 'id="chat-form"' in body
    assert 'hx-post="/chat"' in body
    assert 'name="token"' in body
    assert 'name="files"' in body
