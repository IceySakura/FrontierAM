from nexus.runtime.executors.terminal.screen import VirtualScreen


async def test_feed_and_display_visible():
    s = VirtualScreen(rows=10, cols=40)
    try:
        s.feed(b'hello world\r\n')
        # display() awaits the same single-worker executor, so it observes the
        # feed submitted just above (FIFO ordering).
        assert 'hello world' in await s.display()
    finally:
        s.close()


async def test_blank_trailing_lines_stripped():
    s = VirtualScreen(rows=10, cols=40)
    try:
        s.feed(b'one line\r\n')
        out = await s.display()
        assert out == 'one line'
    finally:
        s.close()


async def test_wide_character_displayed_without_stub():
    s = VirtualScreen(rows=2, cols=10)
    try:
        s.feed('中A'.encode())
        assert await s.display() == '中A'
    finally:
        s.close()


async def test_overwritten_wide_character_preserves_cell_position():
    s = VirtualScreen(rows=2, cols=10)
    try:
        s.feed('中\rA\x1b[4GZ'.encode())
        assert await s.display() == 'A  Z'
    finally:
        s.close()


async def test_resize_changes_dims():
    s = VirtualScreen(rows=10, cols=40)
    try:
        await s.resize(20, 80)
        s.feed(b'x' * 60 + b'\r\n')
        assert 'x' * 60 in await s.display()
    finally:
        s.close()
