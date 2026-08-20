# Zenith remote bot for Akagi

This Akagi plugin forwards MJAI JSONL batches to a remote
`zenith-akagi-server` over one stateful WebSocket connection. It contains no
model or checkpoint.

Copy this directory to `<akagi>/mjai_bot/zenith-remote`, open Akagi's Bots
page, click **Install environment**, configure the server URL and token, and
activate it for 4-player games.

Use `wss://` with a valid certificate when crossing the public internet. A
plain `ws://` URL is appropriate only over a trusted private VPN or an SSH
tunnel.
