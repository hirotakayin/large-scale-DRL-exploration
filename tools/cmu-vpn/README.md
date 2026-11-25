# CMU VPN from the laptop, without a terminal

Reaching the training server needs the CMU VPN. These scripts let the Mission Control GUI (Training tab) or a
shell bring it up with one Duo approval and no sudo prompt.

| File | Runs as | What |
|---|---|---|
| `cmu-vpn` | root, through sudo | `connect <fingerprint>` (session cookie on stdin) / `disconnect` / `status`. Server and all openconnect options are fixed; the fingerprint must match a strict pattern, so the sudoers rule cannot be used to pass `--script` or another server. Tunnel interface `cmuvpn0`, logs to syslog. |
| `cmu-vpn.sudoers.in` | | Sudoers rule template: only `/usr/local/sbin/cmu-vpn`, no password. |
| `cmu-vpn-up` | the user | Logs in with openconnect-sso (CMU login window, password, Duo push), then passes the cookie to `sudo -n cmu-vpn connect`. Exits 0 once the tunnel is up. `--hidden` only if the login page fills in a saved password itself. The user is `$CMU_VPN_USER`, else `$USER`. |
| `cmu-vpn-down` | the user | Disconnects. |

Needs openconnect (`apt install openconnect`) and openconnect-sso in `~/.venvs/cmu-vpn`
(`python3 -m venv ~/.venvs/cmu-vpn && ~/.venvs/cmu-vpn/bin/pip install openconnect-sso`).

## Install (once, on the laptop)

```bash
mkdir -p ~/.local/share/cmu-vpn && cp cmu-vpn-up cmu-vpn-down ~/.local/share/cmu-vpn/   # where the GUI looks
sudo install -o root -g root -m 0755 cmu-vpn /usr/local/sbin/cmu-vpn
sed "s/^USER /$USER /" cmu-vpn.sudoers.in > /tmp/cmu-vpn.sudoers
sudo install -o root -g root -m 0440 /tmp/cmu-vpn.sudoers /etc/sudoers.d/cmu-vpn
sudo visudo -cf /etc/sudoers.d/cmu-vpn      # must print "parsed OK"
~/.local/share/cmu-vpn/cmu-vpn-up           # first login; leave the "TOTP secret" prompt blank (Duo stays the second factor)
```

Never store an OTP secret: the second factor must stay the Duo push.
