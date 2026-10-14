"""``NETWORK_ALLOWLIST`` (PRD §5.2, A40): the declared-endpoint check.

The VM network policy is the authoritative control (A40); this list is
defense in depth. Every declared endpoint — a model ``base_url``
(``models.yaml``), a resource ``url``/``uri`` and an MCP ``url``
(``resources.yaml``) — must match it at config load (:meth:`check_config`,
no DNS) and again at connect (:meth:`check_connect`, which resolves names).
Unset or empty, nothing is checked.
"""

from __future__ import annotations

import asyncio
import fnmatch
import ipaddress
import os
import socket
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from vdreamer.resources import ResourcesError

NETWORK_ALLOWLIST_ENV: Final = "NETWORK_ALLOWLIST"

_Network = ipaddress.IPv4Network | ipaddress.IPv6Network


class EndpointError(ResourcesError):
    """A declared endpoint is malformed or carries credentials (A23)."""


class EndpointNotAllowed(EndpointError):
    """A declared endpoint's host is not on ``NETWORK_ALLOWLIST`` (config load)."""


class EndpointUnavailable(RuntimeError):
    """An endpoint cannot be used now: it does not resolve, or resolves off the list."""


@dataclass(frozen=True)
class NetworkAllowlist:
    """Comma-separated host globs (``*.internal``) and IPs/CIDRs (``10.20.0.0/16``).

    A host passes when it matches a host glob, or when it is (or, at connect,
    resolves only to) addresses inside an allowed network.
    """

    hosts: tuple[str, ...] = ()
    networks: tuple[_Network, ...] = ()

    @classmethod
    def parse(cls, text: str) -> NetworkAllowlist:
        hosts: list[str] = []
        networks: list[_Network] = []
        for raw in text.split(","):
            entry = raw.strip().lower()
            if not entry:
                continue
            try:
                networks.append(ipaddress.ip_network(entry, strict=False))
            except ValueError:
                hosts.append(entry)
        return cls(tuple(hosts), tuple(networks))

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> NetworkAllowlist | None:
        """The allowlist, or ``None`` when ``NETWORK_ALLOWLIST`` is unset or empty."""
        text = (environ if environ is not None else os.environ).get(NETWORK_ALLOWLIST_ENV, "")
        allowlist = cls.parse(text)
        return allowlist if allowlist.hosts or allowlist.networks else None

    def _host_allowed(self, host: str) -> bool:
        return any(fnmatch.fnmatchcase(host.lower(), pattern) for pattern in self.hosts)

    def _address_allowed(self, address: str) -> bool:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            return False
        return any(ip in network for network in self.networks)

    def check_config(self, host: str) -> None:
        """Config-load check: no DNS. A name not on the list may still resolve into
        an allowed network, so it is deferred to :meth:`check_connect` when the
        list has networks."""
        if self._host_allowed(host) or self._address_allowed(host):
            return
        try:
            ipaddress.ip_address(host)
        except ValueError:
            if self.networks:
                return
        raise EndpointNotAllowed(f"host {host!r} is not on {NETWORK_ALLOWLIST_ENV}")

    async def check_connect(self, host: str, port: int) -> None:
        """Connect-time check: a name off the host list must resolve only into
        allowed networks."""
        if self._host_allowed(host) or self._address_allowed(host):
            return
        loop = asyncio.get_running_loop()
        try:
            infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise EndpointUnavailable(f"cannot resolve {host!r}: {exc}") from exc
        addresses = {str(info[4][0]) for info in infos}
        if not addresses or not all(self._address_allowed(address) for address in addresses):
            raise EndpointUnavailable(
                f"host {host!r} resolves to {sorted(addresses)}, outside {NETWORK_ALLOWLIST_ENV}"
            )


def check_url(
    label: str,
    url: str,
    allowlist: NetworkAllowlist | None,
    *,
    schemes: Sequence[str] = ("http", "https"),
) -> urllib.parse.SplitResult:
    """A URL with a host, no inline credentials (A23), and an allowed host (A40)."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in schemes or not parsed.hostname:
        raise EndpointError(f"{label}: {url!r} must be a {'/'.join(schemes)} URL with a host")
    if parsed.username or parsed.password:
        raise EndpointError(
            f"{label}: credentials in the URL put a secret in YAML (A23); use the *_env keys"
        )
    if allowlist is not None:
        try:
            allowlist.check_config(parsed.hostname)
        except EndpointNotAllowed as exc:
            raise EndpointNotAllowed(f"{label}: {exc}") from exc
    return parsed


def default_port(parsed: urllib.parse.SplitResult, default: int | None = None) -> int:
    """The URL's port, else the scheme's usual one."""
    if parsed.port is not None:
        return parsed.port
    if default is not None:
        return default
    return 443 if parsed.scheme.endswith("s") else 80


__all__ = [
    "NETWORK_ALLOWLIST_ENV",
    "EndpointError",
    "EndpointNotAllowed",
    "EndpointUnavailable",
    "NetworkAllowlist",
    "check_url",
    "default_port",
]
