"""Read-only VLC discovery; executed inside the desktop user's session."""

import json


def collect():
    import dbus

    bus = dbus.SessionBus()
    players = []
    for name in bus.list_names():
        if str(name) != 'org.mpris.MediaPlayer2.vlc' and not str(name).startswith(
            'org.mpris.MediaPlayer2.vlc.instance'
        ):
            continue
        props = dbus.Interface(bus.get_object(name, '/org/mpris/MediaPlayer2'), 'org.freedesktop.DBus.Properties')
        state = props.GetAll('org.mpris.MediaPlayer2.Player', timeout=5)
        metadata = state.get('Metadata', {})
        players.append({
            'instance': str(name),
            'state': str(state.get('PlaybackStatus', '')).lower(),
            'url': str(metadata.get('xesam:url', '')),
            'title': str(metadata.get('xesam:title', '')),
        })
    return {'players': players}


if __name__ == '__main__':
    try:
        print(json.dumps(collect()))
    except Exception as exc:
        print(json.dumps({'probe_error': type(exc).__name__}))
